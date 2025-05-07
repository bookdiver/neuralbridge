import jax
import jax.random as jr
import diffrax as dfx

class SDESolver:
    """ Normal SDE Euler solver for most of cases, unable to backpropagate through the solution,
        used for inference.
    """
    dim: int
    drift: callable
    diffusion: callable
    solver: dfx.AbstractItoSolver
    
    def __init__(self, sde, method="euler"):
        self.dim = sde.dim
        self.drift = lambda t, y, _: sde.b(t, y)
        self.diffusion = lambda t, y, _: sde.sigma(t, y)
        
        if method == "euler":
            self.solver = dfx.Euler()
        elif method == "milstein":
            self.solver = dfx.ItoMilstein()
        else:
            raise ValueError(f"Solver type {method} not supported")
        
    def solve(self, rng_key, y0, ts):
        bm = dfx.UnsafeBrownianPath(
            shape=(self.dim, ),
            key=rng_key
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(self.drift),
            dfx.ControlTerm(self.diffusion, bm)
        )
        saveat = dfx.SaveAt(ts=ts)
        sol = dfx.diffeqsolve(
            terms, self.solver, ts[0], ts[-1], None, y0, 
            saveat=saveat, 
            adjoint=dfx.DirectAdjoint(),
            stepsize_controller=dfx.StepTo(ts)
        )
        return sol.ys
    
    def batch_solve(self, rng_key, y0, ts, batch_size=1):
        sub_keys = jr.split(rng_key, batch_size)
        yss = jax.vmap(
            self.solve,
            in_axes=(0, None, None)
        )(sub_keys, y0, ts)
        return yss