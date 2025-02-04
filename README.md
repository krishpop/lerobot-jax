# lerobot-jax

lerobot-jax is a JAX-based library for training multi-task experts using diffusion models and TDMPC2. It is designed to facilitate research and development in reinforcement learning and related fields.

## Installation

To install lerobot-jax, clone the repository and run:

```bash
pip install .
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