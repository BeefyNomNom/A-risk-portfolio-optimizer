import numpy as np


def simulate_returns(mean, volatility, simulations=10_000):
    rng = np.random.default_rng(seed=42)
    returns = rng.normal(
        loc=mean,
        scale=volatility,
        size=simulations
    )
    return returns
if __name__ == "__main__":
    simulated_returns = simulate_returns(
        mean=0.0005,
        volatility=0.02
    )

    print("First 10 simulated returns:")
    print(simulated_returns[:10])

    print("\nMean simulated return:")
    print(simulated_returns.mean())

    print("\nStandard deviation:")
    print(simulated_returns.std())
