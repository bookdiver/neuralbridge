import jax
import jax.numpy as jnp
import jax.random as jr
import tqdm
from functools import partial

from .base import GuidedProposalSDE

class PCNSampler:
    def __init__(self, guided_proposal_sde, *, eta):
        self.guided_proposal_sde = guided_proposal_sde
        self.eta = eta
        
    def sample_dWs(self, ts, rng_key):
        dts = jnp.diff(ts)
        n_steps = len(dts)
        return jr.normal(rng_key, (n_steps, self.guided_proposal_sde.dim_w)) * jnp.sqrt(dts[:, None])
        
    def solve(self, ts, x0, dWs):
        dts = jnp.diff(ts)
        
        def scan_body(carry, vals):
            t, x, ll = carry
            dt, dW = vals
            drift = self.guided_proposal_sde.f(t, x) * dt
            diffusion = self.guided_proposal_sde.g(t, x) @ dW
            ll_increment = self.guided_proposal_sde.G(t, x) * dt
            x_new = x + drift + diffusion
            ll_new = ll + ll_increment
            t_new = t + dt
            return (t_new, x_new, ll_new), (x_new)
        
        (_, _, ll), xs = jax.lax.scan(
            scan_body,
            init=(0.0, x0, 0.0),
            xs=(dts, dWs),
            length=len(dts)
        )
        xs = jnp.concatenate([x0[None, :], xs], axis=0)
        return xs, ll
    
    @partial(jax.jit, static_argnums=(0,))
    def update_step(self, xs, ll, dWs, x0, ts, rng_key):
        sub_key1, sub_key2 = jr.split(rng_key)
        dZs = self.sample_dWs(ts, sub_key1)
        dWs_proposed = self.eta * dWs + jnp.sqrt(1.0 - self.eta ** 2) * dZs

        xs_proposed, ll_proposed = self.solve(ts, x0, dWs_proposed)
        ll_ratio = ll_proposed - ll
        
        log_u = jnp.log(jr.uniform(sub_key2))
        accept = log_u < ll_ratio
        
        xs_updated, ll_updated, dWs_updated = jax.lax.cond(
            accept,
            lambda: (xs_proposed, ll_proposed, dWs_proposed),
            lambda: (xs, ll, dWs)
        )
        return xs_updated, ll_updated, dWs_updated, accept
    
    def sample(self, x0, ts, *, n_iters, n_samples, n_burn_in, rng_key):
        pbar = tqdm.tqdm(range(n_iters), total=n_iters)
        
        n_sampling_after_burn_in = n_iters - n_burn_in
        
        sampling_interval = n_sampling_after_burn_in // n_samples
        
        if n_sampling_after_burn_in <= 0:
            print(f"Warning: n_iters ({n_iters}) is not greater than n_burn_in ({n_burn_in}). "
                  "No samples will be collected after burn-in. Returning empty samples.")
            return jnp.empty((0, len(ts), self.guided_proposal_sde.dim_x)), jnp.zeros((n_iters,))

        # Array to store all samples collected after the burn-in period
        collected_xs_after_burn_in = jnp.zeros(
            (n_samples, len(ts), self.guided_proposal_sde.dim_x)
        )
        
        lls = jnp.zeros((n_iters,)) # Log-likelihoods for all iterations
        
        # Initial state
        current_dWs = self.sample_dWs(ts, rng_key)
        current_xs, current_ll = self.solve(ts, x0, current_dWs)
        
        n_accepted_post_burn_in = 0 # Counter for acceptance rate after burn-in

        # Main MCMC loop
        iter_rng_key = rng_key # Use a separate key for the loop that gets folded
        j = 0
        for i in pbar:
            iter_rng_key = jr.fold_in(iter_rng_key, i) # Update RNG key for this iteration
            
            xs_updated, ll_updated, dWs_updated, accepted = self.update_step(
                current_xs, current_ll, current_dWs, x0, ts, iter_rng_key
            )
            
            lls = lls.at[i].set(ll_updated) # Store log-likelihood of the (potentially) new state
            
            if i >= n_burn_in:
                if i % sampling_interval == 0:
                    collected_xs_after_burn_in = collected_xs_after_burn_in.at[j].set(xs_updated)
                    j += 1
                    
                if accepted:
                    n_accepted_post_burn_in += 1
            
            # Update current state for the next iteration
            current_xs, current_ll, current_dWs = xs_updated, ll_updated, dWs_updated

        if n_sampling_after_burn_in > 0:
            acceptance_rate = n_accepted_post_burn_in / n_sampling_after_burn_in
            print(f"acceptance rate (post-burn-in): {acceptance_rate:.2%}")
        else:
            print("acceptance rate (post-burn-in): N/A (no post-burn-in samples)")
            
        return collected_xs_after_burn_in, lls
        
        


    