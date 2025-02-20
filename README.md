# Neural Guided Diffusion Bridge
Code implementation for https://arxiv.org/html/2502.11909v2, where we proposed a learnable conditioned diffusion process simulation scheme, building upon the _guided proposal_ (https://arxiv.org/abs/1311.3606) but bypassing the canonical MCMC and score matching. Our method especially excels in conditioning on rare events and multi-modal events, while being efficient in generating independent conditioned samples.

## Installation
The whole codebase is implemented with the backbone of JAX and FLAX, to install CPU version (sufficient to reproduce all the experiments in the paper), run:
```bash
pip install .
```
Or GPU version
```bash
pip install .[gpu]
```

## Usage
### Project structure
```
neuralbridge/                  
├── neuralbridge/              
│   ├── models/                
│   ├── networks/              
│   ├── stochastic_processes/  
│   ├── solvers/               
│   ├── utils/                
│   ├── run_scripts/          
│   └── setups.py            
│
├── notebooks/               
├── legacies/                
├── tests/                
├── configs/                 
├── pyproject.toml          
└── README.md                
```
### Examples
All the examples are provided in the forms of Jupyter Notebooks, presenting under `./notebooks` folder.

## Citation
Please consider citing our paper:
```
@misc{yang2025neuralguideddiffusionbridges,
      title={Neural Guided Diffusion Bridges}, 
      author={Gefan Yang and Frank van der Meulen and Stefan Sommer},
      year={2025},
      eprint={2502.11909},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2502.11909}, 
}
```

