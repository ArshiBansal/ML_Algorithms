# dim_reduction_suite.py
# Run: python dim_reduction_suite.py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Try to import UMAP; fall back gracefully if not installed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def build_dataset():
    """
    Use digits dataset (1797 samples, 64D).
    Good testbed for dimension reduction.
    """
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    y = digits.target  # for coloring
    return X, y


def run_all_algorithms(X, y):
    reducers = []

    # 1) PCA
    reducers.append(("PCA", PCA(n_components=2, random_state=42)))

    # 2) Kernel PCA (RBF kernel)
    reducers.append(("Kernel PCA (RBF)", KernelPCA(n_components=2, kernel="rbf", gamma=0.03, random_state=42)))

    # 3) t-SNE
    reducers.append(("t-SNE", TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)))

    # 4) UMAP (if installed)
    if HAS_UMAP:
        reducers.append(("UMAP", umap.UMAP(n_components=2, random_state=42)))
    else:
        reducers.append(("UMAP (not installed)", None))

    # 5) LDA (supervised)
    reducers.append(("LDA", LinearDiscriminantAnalysis(n_components=2)))

    # 6) ICA
    reducers.append(("ICA", FastICA(n_components=2, random_state=42, max_iter=500)))

    # 7) Factor Analysis
    reducers.append(("Factor Analysis", FactorAnalysis(n_components=2, random_state=42)))

    results = []
    for name, reducer in reducers:
        if reducer is None:
            results.append((name, np.zeros((X.shape[0], 2))))
            continue
        try:
            if isinstance(reducer, LinearDiscriminantAnalysis):
                X_new = reducer.fit_transform(X, y)
            else:
                X_new = reducer.fit_transform(X)
            results.append((name, X_new))
        except Exception as e:
            print(f"{name} failed: {e}")
            results.append((f"{name} (failed)", np.zeros((X.shape[0], 2))))
    return results


def plot_results(results, y):
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 14))
    axes = axes.ravel()

    for ax, (name, X_new) in zip(axes, results):
        sc = ax.scatter(X_new[:, 0], X_new[:, 1], c=y, s=10, cmap="tab10")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(results), rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    X, y = build_dataset()
    results = run_all_algorithms(X, y)
    plot_results(results, y)


if __name__ == "__main__":
    main()
