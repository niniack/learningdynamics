# Representing Neural Network Layers as Linear Operations via Koopman Operator Theory

##### Spinning up

This project primarily uses the Jupyter notebook environment via `nbdev`, but using `nbdev` functionality is optional.

1. Install `uv`
```
curl -LsSf https://astral.sh/uv/install.sh | less
```
There are other methods, check out: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer.

1. Set the cache in `pyproject.toml`(where packages are installed)
```
[tool.uv]
cache-dir = "/scratch/nsa325/uv_cache/"
link-mode = "symlink"
```

1. Install packages

```
uv sync
```

That should create a virtual environment. To use the virtual environment on the command line prepend `uv run`. In notebooks, set the kernel to `learningdynamics`.

1. Install local package

```
uv pip install -e .
```

If you would like to view this project as a website with
`nbdev`, install the library and then run `nbdev_preview`.

##### Reproducibility

All the code for the project is available in `nbs` directory. It is
organized by the steps taken in the methodology and available for both
datasets.

All the models/tensors used in the notebooks are available in the
`saved_weights` directory.

All the figures produced in the paper and the raw MNIST accuracies are
available in the `figures` directory and were produced by the notebooks
presented here.

