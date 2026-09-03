import numpy as np


def simulate_portfolio(
    mean_a,
    volatility_a,
    mean_b,
    volatility_b,
    correlation,
    weight_a=0.6,
    weight_b=0.4,
    simulations=10_000
):
    rng = np.random.default_rng(seed=42)

    correlation_matrix = np.array([
        [1.0, correlation],
        [correlation, 1.0]
    ])

    random_returns = rng.multivariate_normal(
        mean=[0, 0],
        cov=correlation_matrix,
        size=simulations
    )

    returns_a = (
        mean_a +
        volatility_a * random_returns[:, 0]
    )

    returns_b = (
        mean_b +
        volatility_b * random_returns[:, 1]
    )

    portfolio_returns = (
        weight_a * returns_a +
        weight_b * returns_b
    )

    return portfolio_returns


if __name__ == "__main__":

    correlations = [-1.0, 0.0, 0.5, 1.0]

    print("Correlation v Volatility")
    print("-----------------------------------")

    for correlation in correlations:

        portfolio_returns = simulate_portfolio(
            mean_a=0.0006,
            volatility_a=0.02,
            mean_b=0.0003,
            volatility_b=0.01,
            correlation=correlation
        )

        volatility = portfolio_returns.std()

        print(
            f"Correlation: {correlation:>4.1f} "
            f"| Volatility: {volatility:.4%}"
        )
