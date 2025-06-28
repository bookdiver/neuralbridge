import jax
import jax.numpy as jnp
import jax.scipy as jsp

from diffrax import ODETerm, Dopri5, diffeqsolve, SaveAt

class BackwardInformationFilter:

    def __init__(self, 
                 prior_sde,
                 auxiliary_sde,
                 observations):
        self.prior_sde = prior_sde
        self.auxiliary_sde = auxiliary_sde

        self.v = observations["v"]
        self.L = observations["L"]
        self.Sigma = observations["Sigma"]
        
        self.dim_x = prior_sde.dim_x
        self.dim_v = self.v.shape[0]
    
    def _initialize_cFH_ode(self):        
        inv_Sigma = jnp.linalg.solve(self.Sigma, jnp.eye(self.dim_v))
        H = self.L @ inv_Sigma @ self.L.T   # dim_x x dim_x
        F = self.v @ inv_Sigma @ self.L.T   # dim_x x 1
        c = jsp.stats.multivariate_normal.logpdf(
            self.v,    
            mean=jnp.zeros_like(self.v),
            cov=self.Sigma
        )                       # scalar
        return H, F, c
    
    def _initialize_uLM_ode(self):
        M_dag = self.Sigma   # dim_v x dim_v
        L = self.L           # dim_v x dim_x
        u = jnp.zeros_like(self.v)  # dim_v
        return L, M_dag, u
        
    def _solve_cFH(self, ts):
        H, F, c = self._initialize_cFH_ode()
        init_state = jnp.concatenate([H.reshape(-1), F, jnp.array([c])])
        
        def vector_field_cFH(t, y, args):
            dim_x, dim_v = args["dim_x"], args["dim_v"]
            B_t = self.auxiliary_sde.B(t)
            beta_t = self.auxiliary_sde.beta(t)
            a_tilde_t = self.auxiliary_sde.a(t)
            
            H_t_flat = y[:(dim_x * dim_x)]
            H_t = H_t_flat.reshape(dim_x, dim_x)
            F_t = y[(dim_x * dim_x):(dim_x * dim_x + dim_x)]
            c_t = y[-1]
            
            dH_dt = (-B_t.T @ H_t - H_t @ B_t + H_t @ a_tilde_t @ H_t)
            dF_dt = (-B_t.T @ F_t + H_t @ a_tilde_t @ F_t + H_t @ beta_t)
            dc_dt = (beta_t @ F_t
                     + 0.5 * F_t @ a_tilde_t @ F_t
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
            y0=init_state,
            args={"dim_x": self.dim_x, "dim_v": self.dim_v},
            saveat=saveat,
        )
        
        Hs = sol.ys[:, :(self.dim_x*self.dim_x)].reshape(-1, self.dim_x, self.dim_x) # (n_steps, d, d)
        Fs = sol.ys[:, (self.dim_x*self.dim_x):(self.dim_x*self.dim_x + self.dim_x)] # (n_steps, d)
        cs = sol.ys[:, -1] # (n_steps,)
        return Hs[::-1], Fs[::-1], cs[::-1]
    
    def _solve_uLM(self, ts):
        L, M_dag, u = self._initialize_uLM_ode()
        init_state = jnp.concatenate([L.reshape(-1), M_dag.reshape(-1), u])
        
        def vector_field_uLM(t, y, args):
            dim_x, dim_v = args["dim_x"], args["dim_v"]
            B_t = self.auxiliary_sde.B(t)
            beta_t = self.auxiliary_sde.beta(t)
            a_tilde_t = self.auxiliary_sde.a(t)
            
            L_t_flat = y[:(dim_v*dim_x)]
            L_t = L_t_flat.reshape(dim_v, dim_x)
            M_t_dag_flat = y[(dim_v*dim_x):(dim_v*dim_x + dim_v*dim_v)]
            M_t_dag = M_t_dag_flat.reshape(dim_v, dim_v)
            u_t = y[(dim_v*dim_x + dim_v*dim_v):]
            
            dL_dt = (-L_t @ B_t)
            dM_dag_dt = (-L_t @ a_tilde_t @ L_t.T)
            du_dt = (-L_t @ beta_t)
            
            dL_dt_flat = dL_dt.reshape(-1)
            dM_dag_dt_flat = dM_dag_dt.reshape(-1)
            du_dt_flat = du_dt
            return jnp.concatenate([dL_dt_flat, dM_dag_dt_flat, du_dt_flat])
        
        term = ODETerm(vector_field_uLM)
        solver = Dopri5()
        saveat = SaveAt(ts=jnp.flip(ts))
        
        sol = diffeqsolve(
            term,
            solver,
            t0=ts[-1],
            t1=ts[0],
            dt0=ts[0] - ts[1],
            y0=init_state,
            args={"dim_x": self.dim_x, "dim_v": self.dim_v},
            saveat=saveat
        )
        
        Ls = sol.ys[:, :(self.dim_v*self.dim_x)].reshape(-1, self.dim_v, self.dim_x)
        Ms_dag = sol.ys[:, (self.dim_v*self.dim_x):(self.dim_v*self.dim_x + self.dim_v*self.dim_v)].reshape(-1, self.dim_v, self.dim_v)
        Ms = jax.vmap(jnp.linalg.inv)(Ms_dag)
        us = sol.ys[:, (self.dim_v*self.dim_x + self.dim_v*self.dim_v):]
        
        H_fn = lambda L, M: L.T @ M @ L            # dim_x x dim_x
        F_fn = lambda L, M, u: L.T @ M @ (self.v - u)  # dim_x
        Hs = jax.vmap(H_fn)(Ls, Ms)
        Fs = jax.vmap(F_fn)(Ls, Ms, us)
        return Hs[::-1], Fs[::-1], us[::-1] 
    
    # def _solve_close_form(self, ts):
    #     t0, T = ts[0], ts[-1]
    #     assert jnp.isclose(self.B(t0), jnp.zeros((self.dim_x, self.dim_x))).all()
    #     assert jnp.isclose(self.beta(t0), jnp.zeros((self.dim_x, ))).all()
        
    #     HT, FT, cT = self._initialize_cFH_ode()
        
    #     # some helper functions
    #     def quadratic(x, A):
    #         """ Compute the quadratic form: x^T A x"""
    #         return x.T @ A @ x
        
    #     def log_omega_H(H):
    #         """ Compute the normalization constant for a Gaussian with precision matrix H."""
    #         abs_log_det = jnp.linalg.slogdet(H)[1]
    #         return 0.5 * (abs_log_det - jnp.log(2.0 * jnp.pi) * self.dim_x)
        
    #     def log_psi_H(x, mu, H):
    #         """ Log Gaussian (observation distribution) pdf with mean of mu and precision matrix of H """
    #         log_norm_const = log_omega_H(H)
    #         return log_norm_const - 0.5 * quadratic(x-mu, H)
        
    #     def inv_Phi(t):
    #         if self.a_tilde_0 is None:
    #             return jnp.eye(self.dim_x) + HT @ self.a_tilde_T * (T - t)
    #         else:
    #             return jnp.eye(self.dim_x) \
    #                 + HT @ (-(t**2 - T**2) / (2.0 * T) * self.a_tilde_T \
    #                         + (T-t)**2 / (2.0 * T) * self.a_tilde_0)
        
    #     def H(t):
    #         return jnp.linalg.solve(inv_Phi(t), HT).reshape(HT.shape)
        
    #     def F(t):
    #         return jnp.linalg.solve(inv_Phi(t), FT).reshape(FT.shape)
        
    #     def c(t, H):
    #         return log_psi_H(jnp.zeros(self.dim_x), mu=self.vT, H=H)
        
    #     Hs = jax.vmap(H)(ts)
    #     Fs = jax.vmap(F)(ts)
    #     cs = jax.vmap(c)(ts, Hs)
    #     return Hs, Fs, cs
    
    def solve(self, ts, eq_type="cFH"):
        if eq_type.lower() == "cfh":
            return self._solve_cFH(ts)
        elif eq_type.lower() == "ulm":
            return self._solve_uLM(ts)
        # elif eq_type.lower() == "close":
        #     return self._solve_close_form(ts)
        else:
            raise ValueError(f"eq_type {eq_type} is not supported")