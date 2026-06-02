import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


def unified_scoring(method, fitted_model, X):
    if method in ["DBSCAN", "Deadwood"]:
        return None
    if method == "LocalOutlierFactor":
        scores = fitted_model.negative_outlier_factor_
    else:
        scores = fitted_model.decision_function(X)
    scores *= -1
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).flatten().tolist()


def find_epsilon(data: np.ndarray, k: int = 4):
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.sort(distances[:, -1])[::-1]

    kneedle = KneeLocator(
        x=range(len(distances)),
        y=distances,
        S=1.0,
        curve="convex",
        direction="decreasing",
    )
    knee = kneedle.knee or len(distances) // 2
    return distances[knee]

