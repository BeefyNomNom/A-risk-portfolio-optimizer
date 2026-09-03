import numpy as np


def generate_asset_returns(
    mean_a,
    volatility_a,
    mean_b,
    volatility_b,
    mean_c,
    volatility_c,
    correlation_matrix,
    simulations=10_000
):
    """
    Generate correlated returns for the cool three assets.

    The same simulated scenarios are used for the differing
    portfolio combinations so the optimisation is nice and fair.
    """

    rng = np.random.default_rng(seed=42)

    random_returns = rng.multivariate_normal(
        mean=[0, 0, 0],
        cov=correlation_matrix,
        size=simulations
    )

    returns_a = (
        mean_a
        + volatility_a * random_returns[:, 0]
    )

    returns_b = (
        mean_b
        + volatility_b * random_returns[:, 1]
    )

    returns_c = (
        mean_c
        + volatility_c * random_returns[:, 2]
    )

    return returns_a, returns_b, returns_c
