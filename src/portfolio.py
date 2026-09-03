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

    # Correlation stuff
    correlation_matrix = np.array([
        [1.0, correlation],
        [correlation, 1.0]
    ])

    # Generate random stuff
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

    # Combine the assets by weight
    portfolio_returns = (
        weight_a * returns_a +
        weight_b * returns_b
    )

    return portfolio_returns


if __name__ == "__main__":

    portfolio_returns = simulate_portfolio(
        mean_a=0.0006,
        volatility_a=0.02,
        mean_b=0.0003,
        volatility_b=0.01,
        correlation=0.5
    )

    print("First 10 portfolio returns:")
    print(portfolio_returns[:10])

    print("\nMean portfolio return:")
    print(portfolio_returns.mean())

    print("\nPortfolio volatility:")
    print(portfolio_returns.std())
