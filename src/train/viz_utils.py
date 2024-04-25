import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_bivariate_distributions(
    predictions: np.array, targets: np.array
) -> plt.Figure:
    """
    Plot bivariate distributions of predictions and targets

    Args:
        predictions: np.array, predictions
        targets: np.array, targets

    Returns:
        fig: plt.Figure, figure of the plot
    """
    sns.set_theme(style="darkgrid")

    axes = sns.histplot(
        data={"targets": targets, "predictions": predictions},
        x="targets",
        y="predictions",
        stat="density",
        bins=list(np.arange(-1, 1.1, 0.1)),
        cbar=True,
    )

    return axes.figure
