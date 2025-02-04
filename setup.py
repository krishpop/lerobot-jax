from setuptools import setup, find_packages

setup(
    name="lerobot-jax",
    version="0.1.0",
    description="A JAX-based library for multi-task experts training using diffusion and TDMPC2.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "flax",
        "ml-collections",
        "numpy",
        "diffusers",
        "optax",
        "absl-py",
        "wandb",
        "torch",
        "tqdm",
        "hydra-core",
        "omegaconf",
        "deepdiff",
        "termcolor",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
) 