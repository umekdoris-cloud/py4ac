"""
mytools.py — Helper Functions for the "Python & ML for Actuaries" Workshop
==========================================================================

This module provides reusable helper functions for data generation and
visualization that are used throughout the course notebooks. By centralising
complex code here, the notebooks stay clean and focused on the concepts being
taught.

Functions are organised into the following groups:
    • Data generators   – synthetic datasets for teaching examples
    • Visualisation      – plots that illustrate ML concepts
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# 1. DATA GENERATORS
# ---------------------------------------------------------------------------

def generate_polynomial_data(n: int = 20, noise: float = 1.0,
                             seed: int = 42) -> tuple:
    """Generate noisy data from a cubic polynomial.

    Used in Notebook 2 to demonstrate under-fitting vs. over-fitting.

    Parameters
    ----------
    n : int
        Number of data points.
    noise : float
        Standard deviation of Gaussian noise added to y.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n,)
        Sorted x values in [-3, 3].
    y : np.ndarray, shape (n,)
        Noisy cubic polynomial values.
    """
    rng = np.random.RandomState(seed)
    X = np.sort(rng.uniform(-3, 3, n))
    # True function: 0.5 x^3 - 1.5 x + 1  (a recognisable cubic)
    y_true = 0.5 * X**3 - 1.5 * X + 1.0
    y = y_true + rng.randn(n) * noise
    return X, y


def generate_two_blobs(n: int = 300, seed: int = 42, cluster_std: float = 5.0) -> tuple:
    """Generate a simple two-class blob dataset.

    Used in Notebook 2 to illustrate classification decision boundaries.

    Parameters
    ----------
    n : int
        Total number of samples (split evenly between the two classes).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n, 2)
        Feature matrix with two columns.
    y : np.ndarray, shape (n,)
        Binary class labels (0 or 1).
    """
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, centers=2, cluster_std=cluster_std,
                       random_state=seed)
    return X, y


def generate_sample_insurance_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a small synthetic motor-insurance dataset.

    Used in Notebook 1 for Pandas and plotting exercises.

    Parameters
    ----------
    n : int
        Number of rows.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        policyholder_id, age, gender, region, vehicle_type,
        annual_premium, claims_count, claim_amount
    """
    rng = np.random.RandomState(seed)

    regions = ["North", "South", "East", "West", "Central"]
    vehicle_types = ["Compact", "Mid-range", "SUV", "Luxury"]

    age = rng.randint(18, 76, n)
    gender = rng.choice(["Female", "Male", "Non-binary"], n,
                        p=[0.48, 0.48, 0.04])
    region = rng.choice(regions, n)
    vehicle_type = rng.choice(vehicle_types, n, p=[0.35, 0.30, 0.20, 0.15])
    annual_premium = (rng.gamma(shape=3.0, scale=500.0, size=n) + 300).round(2)
    claims_count = rng.poisson(lam=1.2, size=n)
    claim_amount = np.where(
        claims_count > 0,
        (rng.lognormal(mean=7.0, sigma=0.8, size=n)).round(2),
        0.0
    )

    df = pd.DataFrame({
        "policyholder_id": np.arange(1, n + 1),
        "age": age,
        "gender": gender,
        "region": region,
        "vehicle_type": vehicle_type,
        "annual_premium": annual_premium,
        "claims_count": claims_count,
        "claim_amount": claim_amount,
    })
    return df


def generate_multiclass_data(n: int = 300, seed: int = 42) -> tuple:
    """Generate a three-class dataset for classification exercises.

    Parameters
    ----------
    n : int
        Total number of samples.
    seed : int
        Random seed.

    Returns
    -------
    X : np.ndarray, shape (n, 2)
    y : np.ndarray, shape (n,)   – labels in {0, 1, 2}
    """
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, centers=3, cluster_std=1.5,
                       random_state=seed)
    return X, y


# ---------------------------------------------------------------------------
# 2. VISUALISATION HELPERS
# ---------------------------------------------------------------------------

def plot_model_selection_regression(X: np.ndarray, y: np.ndarray) -> None:
    """Show under-fitting, good fit, and over-fitting side by side.

    Creates a 1×3 figure:
        – Left:   linear fit  (high bias / under-fitting)
        – Centre: cubic fit   (good generalisation)
        – Right:  degree-19 fit (high variance / over-fitting)

    Parameters
    ----------
    X : np.ndarray, shape (n,)
    y : np.ndarray, shape (n,)
    """
    x_plot = np.linspace(X.min() - 0.5, X.max() + 0.5, 500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = [
        "Linear Model (Degree 1)\n→ Under-fitting (High Bias)",
        "Polynomial Model (Degree 3)\n→ Good Fit",
        f"Polynomial Model (Degree {len(X) - 1})\n→ Over-fitting (High Variance)",
    ]
    degrees = [1, 3, len(X) - 1]
    colours = ["#e74c3c", "#27ae60", "#8e44ad"]

    for ax, title, deg, col in zip(axes, titles, degrees, colours):
        coeffs = np.polyfit(X, y, deg)
        y_plot = np.polyval(coeffs, x_plot)

        ax.scatter(X, y, c="steelblue", edgecolors="k", s=50, zorder=3)
        ax.plot(x_plot, y_plot, color=col, linewidth=2.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    # Share the same y-axis limits across all three subplots
    y_margin = (y.max() - y.min()) * 0.2
    shared_ylim = (y.min() - y_margin, y.max() + y_margin)
    for ax in axes:
        ax.set_ylim(shared_ylim)

    plt.tight_layout()
    plt.show()


def plot_classification_boundaries(X: np.ndarray, y: np.ndarray,
                                   classifiers: list, titles: list,
                                   figsize: tuple = (16, 4)) -> None:
    """Plot decision boundaries for several classifiers in one row.

    Parameters
    ----------
    X : np.ndarray, shape (n, 2)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Binary labels.
    classifiers : list
        Fitted or unfitted sklearn classifier instances.
    titles : list of str
        Subplot titles (same length as *classifiers*).
    figsize : tuple
        Figure size.
    """
    h = 0.05  # mesh step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    cmap_bg = ListedColormap(["#FFCCCC", "#CCE5FF"])
    cmap_pts = ListedColormap(["#e74c3c", "#2980b9"])

    fig, axes = plt.subplots(1, len(classifiers), figsize=figsize)
    if len(classifiers) == 1:
        axes = [axes]

    for ax, clf, title in zip(axes, classifiers, titles):
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts,
                   edgecolors="k", s=25, linewidths=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_custom(y_true, y_pred, labels=None,
                                 title="Confusion Matrix") -> None:
    """Plot a simple confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Class labels for axis ticks.
    title : str
        Plot title.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(cm.shape[0])
    if labels is None:
        labels = tick_marks
    ax.set(xticks=tick_marks, yticks=tick_marks,
           xticklabels=labels, yticklabels=labels,
           ylabel="True Label", xlabel="Predicted Label",
           title=title)

    # Write numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 10,
                            title: str = "Feature Importance") -> None:
    """Horizontal bar chart of feature importances from a tree-based model.

    Parameters
    ----------
    model : fitted sklearn estimator with ``feature_importances_`` attribute.
    feature_names : list of str
    top_n : int
        Show only the top-n most important features.
    title : str
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 0.5 * top_n + 1))
    ax.barh(range(top_n), importances[indices[::-1]],
            color="steelblue", edgecolor="k")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def load_titanic_data() -> tuple:
    """Load and prepare the Titanic dataset for classification.

    Loads the Titanic dataset via seaborn, selects relevant features,
    encodes categorical variables (sex, embarked), and drops rows with
    missing values.

    Returns
    -------
    X : np.ndarray, shape (n, 7)
        Feature matrix with columns: pclass, sex, age, sibsp, parch,
        fare, embarked.
    y : np.ndarray, shape (n,)
        Binary survival labels (0 = died, 1 = survived).
    feature_names : list of str
        Human-readable names for the seven features.
    """
    import seaborn as sns

    titanic = sns.load_dataset('titanic')

    df = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp',
                   'parch', 'fare', 'embarked']].copy()
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df = df.dropna()

    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].values
    y = df['survived'].values

    return X, y, feature_names
