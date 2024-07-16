import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation


def extract_layer_weights(model, state_dicts, layer_num):
    """Efficiently extracts weights of a specific layer from multiple state dicts.

    Args:
        model: The PyTorch model.
        state_dicts: A list of model state dictionaries.
        layer_num: The index of the layer to extract weights from (0-based).

    Returns:
        A tensor containing the weights of the specified layer across all state dicts.
    """

    device = next(model.parameters()).device  # Get model's device

    # Identify the target layer's weight key
    target_weight_key = None
    for i, (key, _) in enumerate(model.state_dict().items()):
        if i == layer_num:
            target_weight_key = key
            break

    if target_weight_key is None:
        raise ValueError(f"Layer {layer_num} not found in the model.")

    # Efficiently extract weights using a list comprehension and torch.stack
    weights = torch.stack(
        [state_dict[target_weight_key].to(device) for state_dict in state_dicts], dim=0
    )

    weights = torch.flatten(weights, start_dim=1)

    return weights.cpu()  # Move back to CPU for further analysis


# Remove the state_dict_generator here, it's now integrated into plot_decision_boundary_movie
def plot_decision_boundary_movie(
    model, state_dicts, X, y, figsize=(8, 8), labels=[0, 1]
):
    """Creates a matplotlib animation of the decision boundary evolution.

    Args:
        model: PyTorch model.
        state_dicts: list of PyTorch model state dicts.
        X: The feature matrix (torch.Tensor).
        y: The label vector (torch.Tensor).
    """
    # Initialization
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_in = np.c_[xx.ravel(), yy.ravel()]
    x_in = torch.tensor(x_in, dtype=torch.float32).to(
        next(model.parameters()).device
    )  # Move to GPU

    # Figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        X[y == labels[0], 0],
        X[y == labels[0], 1],
        c="crimson",
        marker="o",
        s=50,
        label="Class -1",
    )
    ax.scatter(
        X[y == labels[1], 0],
        X[y == labels[1], 1],
        c="royalblue",
        marker="o",
        s=50,
        label="Class 1",
    )

    # Initialize `contour` outside of the update function
    contour = ax.contourf(
        xx,
        yy,
        np.zeros_like(xx),
        cmap="coolwarm_r",
        alpha=0.3,
        levels=np.linspace(labels[0], labels[1], 3),
    )

    # Precompute predictions for all state_dicts
    y_preds = []
    for model_state_dict in state_dicts:
        model.load_state_dict(model_state_dict)
        model.eval()
        out = model(x_in)
        out = torch.argmax(out, dim=1)
        with torch.no_grad():
            y_preds.append(out.cpu().numpy().reshape(xx.shape))

    # Update function with precomputed predictions
    def update(frame):
        nonlocal contour  # Use the outer scope's contour variable

        y_pred = y_preds[frame]

        for coll in contour.collections:
            coll.remove()
        contour = ax.contourf(
            xx,
            yy,
            y_pred,
            cmap="coolwarm_r",
            alpha=0.3,
            levels=np.linspace(labels[0], labels[1], 3),
        )
        return contour.collections

    # Animation
    ani = FuncAnimation(fig, update, frames=len(state_dicts), blit=True, repeat=False)
    return ani
