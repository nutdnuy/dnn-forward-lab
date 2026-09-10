"""Rolling OHLC estimators and a clearly separate cross-asset clustering extension."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import validate_ohlc


def historical_volatility(
    frame: pd.DataFrame, window: int = 20, annualization: int = 252
) -> pd.DataFrame:
    frame = validate_ohlc(frame)
    if window < 2 or annualization < 1:
        raise ValueError("window >= 2 and annualization >= 1 are required")
    hl = np.log(frame.high / frame.low)
    co = np.log(frame.close / frame.open)
    oc = np.log(frame.open / frame.close.shift(1))
    rs = np.log(frame.high / frame.close) * np.log(frame.high / frame.open) + np.log(
        frame.low / frame.close
    ) * np.log(frame.low / frame.open)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    variances = pd.DataFrame(
        {
            "parkinson": (hl**2).rolling(window).mean() / (4 * np.log(2)),
            "garman_klass": (0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).rolling(window).mean(),
            "rogers_satchell": rs.rolling(window).mean(),
            "yang_zhang": oc.rolling(window).var(ddof=1)
            + k * co.rolling(window).var(ddof=1)
            + (1 - k) * rs.rolling(window).mean(),
        }
    )
    return np.sqrt(variances.clip(lower=0) * annualization)


def cluster_assets(
    assets: dict[str, pd.DataFrame], cutoff: str, clusters: int = 3, seed: int = 42
) -> dict:
    """One asset = one mean-volatility feature vector, using only common pre-cutoff dates."""
    if type(clusters) is not int or not 1 <= clusters <= len(assets):
        raise ValueError("clusters must be between 1 and the asset count")
    bounded = {
        name: validate_ohlc(frame).loc[: pd.Timestamp(cutoff)] for name, frame in assets.items()
    }
    common = None
    for frame in bounded.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if len(common) < 40:
        raise ValueError("Need at least 40 common observations on or before cutoff")
    features = pd.DataFrame(
        {
            name: historical_volatility(frame.loc[common]).dropna().mean()
            for name, frame in bounded.items()
        }
    ).T.sort_index()
    if len(features.drop_duplicates()) < clusters:
        raise ValueError("Too few distinct volatility vectors for the requested clusters")
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    model = KMeans(n_clusters=clusters, init="k-means++", n_init=20, random_state=seed).fit(x)
    # Relabel by mean raw volatility: 0 is the lowest observed cluster, not a market-wide category.
    order = sorted(
        range(clusters), key=lambda k: features.iloc[model.labels_ == k].to_numpy().mean()
    )
    labels = [order.index(int(label)) for label in model.labels_]
    output = features.copy()
    output["cluster"] = labels
    return {
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "common_start": str(common[0].date()),
        "common_end": str(common[-1].date()),
        "observations": len(common),
        "seed": seed,
        "method": "cross-asset mean volatility extension; not paper observation clustering",
        "assets": output.reset_index(names="asset").to_dict(orient="records"),
        "inertia": float(model.inertia_),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
