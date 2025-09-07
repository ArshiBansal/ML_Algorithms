import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    MeanShift,
    SpectralClustering,
    AffinityPropagation,
)

from sklearn.mixture import GaussianMixture

RNG = np.random.RandomState(42)

# --- Try to import KMedoids from sklearn_extra; if missing, use a simple PAM fallback.
USING_SKLEARN_EXTRA = True
try:
    from sklearn_extra.cluster import KMedoids  # type: ignore
except Exception:
    USING_SKLEARN_EXTRA = False

    class KMedoids:
        """
        Simple PAM (Partitioning Around Medoids) fallback.
        Euclidean distance, O(k*(n-k)*n) per iteration, ok for small demos.
        API mimics sklearn estimator minimally: fit(X), labels_, medoid_indices_
        """
        def __init__(self, n_clusters=3, max_iter=300, random_state=None):
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.random_state = random_state
            self.labels_ = None
            self.medoid_indices_ = None

        def _total_cost(self, D, medoids, labels):
            return np.sum(D[np.arange(D.shape[0]), medoids[labels]])

        def fit(self, X, y=None):
            n = X.shape[0]
            D = pairwise_distances(X, metric="euclidean")
            rng = np.random.RandomState(self.random_state)
            medoids = rng.choice(n, self.n_clusters, replace=False)
            # Initial assignment
            labels = np.argmin(D[:, medoids], axis=1)
            best_cost = self._total_cost(D, medoids, labels)

            for _ in range(self.max_iter):
                improved = False
                for mi in range(self.n_clusters):
                    for candidate in range(n):
                        if candidate in medoids:
                            continue
                        new_medoids = medoids.copy()
                        new_medoids[mi] = candidate
                        new_labels = np.argmin(D[:, new_medoids], axis=1)
                        new_cost = self._total_cost(D, new_medoids, new_labels)
                        if new_cost + 1e-12 < best_cost:
                            medoids = new_medoids
                            labels = new_labels
                            best_cost = new_cost
                            improved = True
                if not improved:
                    break

            self.labels_ = labels
            self.medoid_indices_ = medoids
            return self


def build_dataset(n_samples=900, random_state=42):
    """
    Build a 2D dataset mixing easy blobs + non-spherical shapes.
    This stresses the algorithms differently.
    """
    X1, _ = make_blobs(
        n_samples=n_samples // 2,
        centers=[(-6, 0), (0, 6), (6, 0)],
        cluster_std=[1.1, 1.0, 1.2],
        random_state=random_state,
    )
    X2, _ = make_moons(n_samples=n_samples // 2, noise=0.06, random_state=random_state)
    X2 *= 6.0  # spread the moons
    X = np.vstack([X1, X2])
    X = StandardScaler().fit_transform(X)
    return X


def safe_silhouette(X, labels):
    """Compute silhouette safely (needs at least 2 clusters and no single-cluster labeling)."""
    labels = np.asarray(labels)
    unique = np.unique(labels[labels >= 0])  # ignore noise labels like -1
    if unique.size < 2:
        return np.nan
    try:
        return silhouette_score(X, labels)
    except Exception:
        return np.nan


def run_all_algorithms(X, k=3, random_state=42):
    algos = []

    # 1) K-Means
    algos.append(("K-Means", KMeans(n_clusters=k, n_init="auto", random_state=random_state)))

    # 2) K-Medoids (PAM)
    if USING_SKLEARN_EXTRA:
        algos.append(("K-Medoids (sklearn-extra)", KMedoids(n_clusters=k, random_state=random_state)))
    else:
        algos.append(("K-Medoids (PAM - fallback)", KMedoids(n_clusters=k, random_state=random_state)))

    # 3) Hierarchical (Agglomerative - Ward)
    algos.append(("Hierarchical (Ward)", AgglomerativeClustering(n_clusters=k, linkage="ward")))

    # 4) DBSCAN
    algos.append(("DBSCAN", DBSCAN(eps=0.3, min_samples=8)))

    # 5) OPTICS
    algos.append(("OPTICS", OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)))

    # 6) Gaussian Mixture Models
    algos.append(("GMM (EM)", GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)))

    # 7) Mean-Shift
    algos.append(("Mean-Shift", MeanShift(bin_seeding=True)))

    # 8) Spectral Clustering
    algos.append(("Spectral Clustering", SpectralClustering(n_clusters=k, assign_labels="kmeans", affinity="nearest_neighbors", random_state=random_state)))

    # 9) Affinity Propagation
    algos.append(("Affinity Propagation", AffinityPropagation(random_state=random_state, damping=0.9)))

    results = []
    for name, model in algos:
        try:
            if isinstance(model, GaussianMixture):
                model.fit(X)
                labels = model.predict(X)
            else:
                model.fit(X)
                labels = getattr(model, "labels_", None)
                if labels is None and hasattr(model, "predict"):
                    labels = model.predict(X)
            sil = safe_silhouette(X, labels)
            results.append((name, labels, sil))
        except Exception as e:
            # If an algo fails (e.g., Spectral if graph not connected with chosen params), mark it.
            results.append((f"{name} (failed: {type(e).__name__})", np.full(X.shape[0], -1), np.nan))
    return results


def plot_results(X, results):
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 14))
    axes = axes.ravel()

    for ax, (name, labels, sil) in zip(axes, results):
        # Map noise labels (-1) to a distinct marker edge.
        # Let matplotlib pick colors automatically.
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=12)
        ax.set_title(f"{name}\nSilhouette: {sil:.3f}" if not np.isnan(sil) else f"{name}\nSilhouette: n/a")
        ax.set_xticks([])
        ax.set_yticks([])
        # Optionally, show legend of unique labels count
        uniq = len(np.unique(labels))
        ax.text(0.02, 0.97, f"k*={uniq}", transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0))

    # Hide any unused subplots (in case fewer results)
    for j in range(len(results), rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    X = build_dataset(n_samples=900, random_state=42)
    results = run_all_algorithms(X, k=3, random_state=42)
    plot_results(X, results)

    # Print a quick summary table
    print("\n=== Silhouette Summary ===")
    for name, _, sil in results:
        print(f"{name:26s}  ->  {('n/a' if np.isnan(sil) else f'{sil:.3f}')}")


if __name__ == "__main__":
    main()
