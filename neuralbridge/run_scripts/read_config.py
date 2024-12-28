from omegaconf import OmegaConf

from neuralbridge.setups import *

def read_config(config_path):
    config = OmegaConf.load(config_path)
    # Convert v values from strings to float/nan
    if 'sde' in config:
        u_values = [float('nan') if x.lower() == 'nan' else float(x) 
                    for x in config.sde.u]
        v_values = [float('nan') if x.lower() == 'nan' else float(x) 
                    for x in config.sde.v]
        config.sde.u = u_values
        config.sde.v = v_values
        
    return config