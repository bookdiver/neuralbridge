import jax
import jax.numpy as jnp
import jax.scipy as jsp

from diffrax import ODETerm, Dopri5, diffeqsolve, SaveAt

class BackwardInformationFilter:
    # unconditional & auxiliary process
    b: callable           
    B: callable                
    beta: callable
    a_tilde: callable
    a_tilde_0: jnp.ndarray
    a_tilde_T: jnp.ndarray
    dim: int
    
    # observations
    vT: jnp.ndarray
    LT: jnp.ndarray
    SigmaT: jnp.ndarray

    def __init__(self, 
                 sde,
                 aux_sde,
                 obs_params):
        self.b = lambda t, x: sde.b(t, x)
        self.B = lambda t: aux_sde.B(t)
        self.beta = lambda t: aux_sde.beta(t)
        self.a_tilde = lambda t: aux_sde.a(t, x=None)
        self.a_tilde_0 = aux_sde._a_tilde_0
        self.a_tilde_T = aux_sde._a_tilde_T
        self.dim = sde.dim
        
        self.vT = obs_params["vT"]
        self.LT = obs_params["LT"]
        self.SigmaT = obs_params["SigmaT"]
    
    def _initialize_cFH_ode(self):        
        inv_SigmaT = jnp.linalg.inv(self.SigmaT)
        HT = jnp.dot(self.LT, inv_SigmaT @ self.LT)   # d x d
        FT = jnp.dot(self.vT, inv_SigmaT @ self.LT)   # d x 1
        cT = jsp.stats.multivariate_normal.logpdf(
            self.vT,    
            mean=jnp.zeros_like(self.vT),
            cov=self.SigmaT
        )
        return HT, FT, cT
    
    def _initialize_uLM_ode(self):
        MT_dag = self.SigmaT
        LT = self.LT
        muT = jnp.zeros_like(self.vT)
        return LT, MT_dag, muT
        
    def _solve_cFH(self, ts):
        HT, FT, cT = self._initialize_cFH_ode()
        y0 = jnp.concatenate([HT.reshape(-1), FT, jnp.array([cT])])
        
        def vector_field_cFH(t, y, args):
            dim = args["dim"]
            B_t = self.B(t)
            beta_t = self.beta(t)
            a_tilde_t = self.a_tilde(t)
            
            H_t_flat = y[:(dim*dim)]
            H_t = H_t_flat.reshape(dim, dim)
            F_t = y[(dim*dim):(dim*dim + dim)]
            # c_t = y[-1]
            
            dH_dt = (-B_t.T @ H_t - H_t @ B_t + H_t @ a_tilde_t @ H_t)
            dF_dt = (-B_t.T @ F_t + H_t @ a_tilde_t @ F_t + H_t @ beta_t)
            dc_dt = (jnp.dot(beta_t, F_t) \
                     + 0.5 * jnp.dot(F_t, a_tilde_t @ F_t) \
                     - 0.5 * jnp.trace(H_t @ a_tilde_t))

            dH_dt_flat = dH_dt.reshape(-1)
            dF_dt_flat = dF_dt
            dc_dt_flat = jnp.array([dc_dt])
            return jnp.concatenate([dH_dt_flat, dF_dt_flat, dc_dt_flat])
        
        term = ODETerm(vector_field_cFH)
        solver = Dopri5()
        saveat = SaveAt(ts=jnp.flip(ts))
        
        sol = diffeqsolve(
            term,
            solver,
            t0=ts[-1],
            t1=ts[0],
            dt0=ts[0]-ts[1],
            y0=y0,
            args={"dim": self.dim},
            saveat=saveat,
        )
        
        Hs = sol.ys[:, :(self.dim*self.dim)].reshape(-1, self.dim, self.dim) # (n_steps, d, d)
        Fs = sol.ys[:, (self.dim*self.dim):(self.dim*self.dim + self.dim)] # (n_steps, d)
        cs = sol.ys[:, -1] # (n_steps,)
        return Hs[::-1], Fs[::-1], cs[::-1]
    
    def _solve_uLM(self, ts):
        LT, MT_dag, muT = self._initialize_uLM_ode()
        y0 = jnp.concatenate([LT.reshape(-1), MT_dag.reshape(-1), jnp.array(muT)])
        
        def vector_field_uLM(t, y, args):
            dim = args["dim"]
            B_t = self.B(t)
            beta_t = self.beta(t)
            a_tilde_t = self.a_tilde(t)
            
            L_t_flat = y[:(dim*dim)]
            L_t = L_t_flat.reshape(dim, dim)
            # M_t_dag_flat = y[(dim*dim):(2*dim*dim)]
            # M_t_dag = M_t_dag_flat.reshape(dim, dim)
            # mu_t = y[(2*dim*dim):]
            
            dL_dt = (-L_t @ B_t)
            dM_dag_dt = (-L_t @ a_tilde_t @ L_t.T)
            dmu_dt = (-L_t @ beta_t)
            
            dL_dt_flat = dL_dt.reshape(-1)
            dM_dag_dt_flat = dM_dag_dt.reshape(-1)
            dmu_dt_flat = dmu_dt
            return jnp.concatenate([dL_dt_flat, dM_dag_dt_flat, dmu_dt_flat])
        
        term = ODETerm(vector_field_uLM)
        solver = Dopri5()
        saveat = SaveAt(ts=jnp.flip(ts))
        
        sol = diffeqsolve(
            term,
            solver,
            t0=ts[-1],
            t1=ts[0],
            dt0=ts[0] - ts[1],
            y0=y0,
            args={"dim": self.dim},
            saveat=saveat
        )
        
        Ls = sol.ys[:, :(self.dim*self.dim)].reshape(-1, self.dim, self.dim)
        Ms_dag = sol.ys[:, (self.dim*self.dim):(2*self.dim*self.dim)].reshape(-1, self.dim, self.dim)
        Ms = jax.vmap(jnp.linalg.inv)(Ms_dag)
        mus = sol.ys[:, 2*self.dim*self.dim:]
        
        H_fn = lambda L, M: L.T @ M @ L
        F_fn = lambda L, M, mu: L.T @ M @ (self.vT - mu)
        Hs = jax.vmap(H_fn)(Ls, Ms)
        Fs = jax.vmap(F_fn)(Ls, Ms, mus)
        return Hs[::-1], Fs[::-1], mus[::-1]
    
    def _solve_close_form(self, ts):
        t0, T = ts[0], ts[-1]
        assert jnp.isclose(self.B(t0), jnp.zeros((self.dim, self.dim))).all()
        assert jnp.isclose(self.beta(t0), jnp.zeros((self.dim, ))).all()
        
        HT, FT, cT = self._initialize_cFH_ode()
        
        # some helper functions
        def quadratic(x, A):
            """ Compute the quadratic form: x^T A x"""
            return x.T @ A @ x
        
        def log_omega_H(H):
            """ Compute the normalization constant for a Gaussian with precision matrix H."""
            abs_log_det = jnp.linalg.slogdet(H)[1]
            return 0.5 * (abs_log_det - jnp.log(2.0 * jnp.pi) * self.dim)
        
        def log_psi_H(x, mu, H):
            """ Log Gaussian (observation distribution) pdf with mean of mu and precision matrix of H """
            log_norm_const = log_omega_H(H)
            return log_norm_const - 0.5 * quadratic(x-mu, H)
        
        def inv_Phi(t):
            if self.a_tilde_0 is None:
                return jnp.eye(self.dim) + HT @ self.a_tilde_T * (T - t)
            else:
                return jnp.eye(self.dim) \
                    + HT @ (-(t**2 - T**2) / (2.0 * T) * self.a_tilde_T \
                            + (T-t)**2 / (2.0 * T) * self.a_tilde_0)
        
        def H(t):
            return jnp.linalg.solve(inv_Phi(t), HT).reshape(HT.shape)
        
        def F(t):
            return jnp.linalg.solve(inv_Phi(t), FT).reshape(FT.shape)
        
        def c(t, H):
            return log_psi_H(jnp.zeros(self.dim), mu=self.vT, H=H)
        
        Hs = jax.vmap(H)(ts)
        Fs = jax.vmap(F)(ts)
        cs = jax.vmap(c)(ts, Hs)
        return Hs, Fs, cs
    
    def solve(self, ts, eq_type="cFH"):
        if eq_type.lower() == "cfh":
            return self._solve_cFH(ts)
        elif eq_type.lower() == "ulm":
            return self._solve_uLM(ts)
        elif eq_type.lower() == "close":
            return self._solve_close_form(ts)
        else:
            raise ValueError(f"eq_type {eq_type} is not supported")