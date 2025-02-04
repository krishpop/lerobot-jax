# lerobot-jax

lerobot-jax is a JAX-based library for training multi-task policies using a combination of offline RL and imitation learning techniques. 
It is designed to facilitate research and development in reinforcement learning and related fields.

## Installation

To install lerobot-jax, clone the repository and run:
```bash
pip install -e .
```

When installing packages, you can install miniconda to setup a conda environment with all other dependencies:
`curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
`bash Miniconda3-latest-Linux-x86_64.sh`


### Download lerobot, jaxrl_m, and related dependencies

To setup `multi_task_experts`,
`cd $HOME`
`git clone --recursive --branch tdmpc-jax git@github.com:krishpop/multi_task_experts.git`

To install `lerobot`
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
conda create -n lerobot python=3.10
pip install -e .[pusht]
```

To install `d3il` environments
```bash
git clone https://github.com/ALRhub/d3il.git
cd d3il/environments/d3il/envs/gym_stacking
pip install -e .
cd ../gym_sorting
pip install -e .
conda env config vars set PYTHONPATH=[D3IL INSTALL PATH]
```

To install `jaxrl_m`
```bash
git clone https://github.com/krishpop/jaxrl_m.git
pip install -e .
conda install -c conda-forge gin-config gym==0.21.0 -y
conda install conda-forge::pinocchio
pip install pybullet==3.2.6 --no-deps
pip install mujoco==2.3.2
```

To install jax[tpu], flax, ml-collections, diffusers
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax ml-collections diffusers distrax
pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html
```

To download pretrained encoders:
```bash
cd $HOME
git clone git@github.com:krishpop/pretrained_vision.git
python pretrained_vision/pretrained_vision/resnetv1.py \
    --pretrained_path=imagenet --prefix=imagenet-resnetv1-18 \
    --encoder resnetv1-18 --save_dir [jaxrl_m INSTALL PATH]/pretrained_encoders_new/
```

## Usage

To use lerobot-jax, import the necessary modules and start training your models. Here is a basic example:

```python
from lerobot_jax import diffusion_jax, tdmpc2_jax

# Initialize your model and start training
model = diffusion_jax.create_simple_diffusion_learner(...)
model.train(...)
```


## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details. 

