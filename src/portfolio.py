import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path

from simulation import generate_asset_returns
from risk import calculate_var, calculate_expected_shortfall
from optimisation import (
    calculate_portfolio_returns,
    calculate_sharpe_ratio,
    find_efficient_frontier
)


if __name__ == "__main__":

    mean_a = 0.0006
    volatility_a = 0.02

    mean_b = 0.0003
    volatility_b = 0.01

    mean_c = 0.0008
    volatility_c = 0.015

    correlation_matrix = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])


    returns_a, returns_b, returns_c = generate_asset_returns(
        mean_a=mean_a,
        volatility_a=volatility_a,
        mean_b=mean_b,
        volatility_b=volatility_b,
        mean_c=mean_c,
        volatility_c=volatility_c,
        correlation_matrix=correlation_matrix
    )


    portfolios = []

    for weight_a in np.arange(0, 1.01, 0.01):

        for weight_b in np.arange(0, 1.01, 0.01):

            weight_c = 1 - weight_a - weight_b

            if weight_c < 0:
                continue

            portfolio_returns = calculate_portfolio_returns(
                returns_a,
                returns_b,
                returns_c,
                weight_a,
                weight_b,
                weight_c
            )

            mean_return = portfolio_returns.mean()

            volatility = portfolio_returns.std()

            sharpe_ratio = calculate_sharpe_ratio(
                portfolio_returns
            )

            portfolios.append({
                "weight_a": weight_a,
                "weight_b": weight_b,
                "weight_c": weight_c,
                "return": mean_return,
                "volatility": volatility,
                "sharpe": sharpe_ratio
            })

    minimum_volatility = min(
        portfolios,
        key=lambda portfolio: portfolio["volatility"]
    )

    maximum_sharpe = max(
        portfolios,
        key=lambda portfolio: portfolio["sharpe"]
    )

    maximum_return = max(
        portfolios,
        key=lambda portfolio: portfolio["return"]
    )

    efficient_frontier = find_efficient_frontier(
        portfolios
    )


    best_returns = calculate_portfolio_returns(
        returns_a,
        returns_b,
        returns_c,
        maximum_sharpe["weight_a"],
        maximum_sharpe["weight_b"],
        maximum_sharpe["weight_c"]
    )

    var_95 = calculate_var(best_returns)

    expected_shortfall = calculate_expected_shortfall(
        best_returns
    )

    print()
    print("3-Asset Portfolio Optimisation")
    print("==============================")
    print()

    print(f"Portfolios tested: {len(portfolios):,}")
    print(
        f"Efficient portfolios: "
        f"{len(efficient_frontier):,}"
    )


    print()
    print("Minimum-Volatility Portfolio")
    print("----------------------------")

    print(f"Asset A: {minimum_volatility['weight_a']:.0%}")
    print(f"Asset B: {minimum_volatility['weight_b']:.0%}")
    print(f"Asset C: {minimum_volatility['weight_c']:.0%}")

    print(
        f"Mean return: "
        f"{minimum_volatility['return']:.4%}"
    )

    print(
        f"Volatility: "
        f"{minimum_volatility['volatility']:.4%}"
    )


    print()
    print("Maximum-Sharpe Portfolio")
    print("------------------------")

    print(f"Asset A: {maximum_sharpe['weight_a']:.0%}")
    print(f"Asset B: {maximum_sharpe['weight_b']:.0%}")
    print(f"Asset C: {maximum_sharpe['weight_c']:.0%}")

    print(
        f"Sharpe ratio: "
        f"{maximum_sharpe['sharpe']:.4f}"
    )

    print(
        f"Mean return: "
        f"{maximum_sharpe['return']:.4%}"
    )

    print(
        f"Volatility: "
        f"{maximum_sharpe['volatility']:.4%}"
    )

    print(
        f"95% VaR: "
        f"{var_95:.4%}"
    )

    print(
        f"95% Expected Shortfall: "
        f"{expected_shortfall:.4%}"
    )


    print()
    print("Maximum-Return Portfolio")
    print("------------------------")

    print(f"Asset A: {maximum_return['weight_a']:.0%}")
    print(f"Asset B: {maximum_return['weight_b']:.0%}")
    print(f"Asset C: {maximum_return['weight_c']:.0%}")

    print(
        f"Mean return: "
        f"{maximum_return['return']:.4%}"
    )

    print(
        f"Volatility: "
        f"{maximum_return['volatility']:.4%}"
    )

    project_root = Path(__file__).resolve().parent.parent

    figures_directory = (
        project_root
        / "results"
        / "figures"
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    all_volatility = [
        portfolio["volatility"]
        for portfolio in portfolios
    ]

    all_returns = [
        portfolio["return"]
        for portfolio in portfolios
    ]

    frontier_volatility = [
        portfolio["volatility"]
        for portfolio in efficient_frontier
    ]

    frontier_returns = [
        portfolio["return"]
        for portfolio in efficient_frontier
    ]

    plt.figure(figsize=(11, 7))


    plt.scatter(
        all_volatility,
        all_returns,
        s=8,
        alpha=0.15,
        label="Feasible portfolios"
    )


    plt.plot(
        frontier_volatility,
        frontier_returns,
        linewidth=2,
        label="Efficient frontier"
    )


    plt.scatter(
        minimum_volatility["volatility"],
        minimum_volatility["return"],
        s=140,
        marker="X",
        label="Minimum volatility"
    )


    plt.scatter(
        maximum_sharpe["volatility"],
        maximum_sharpe["return"],
        s=160,
        marker="*",
        label="Maximum Sharpe"
    )


    plt.scatter(
        maximum_return["volatility"],
        maximum_return["return"],
        s=120,
        marker="D",
        label="Maximum return"
    )

    plt.annotate(
        "Minimum volatility",
        (
            minimum_volatility["volatility"],
            minimum_volatility["return"]
        ),
        xytext=(15, -25),
        textcoords="offset points"
    )


    plt.annotate(
        "Maximum Sharpe",
        (
            maximum_sharpe["volatility"],
            maximum_sharpe["return"]
        ),
        xytext=(15, 15),
        textcoords="offset points"
    )

    plt.annotate(
        "Maximum return",
        (
            maximum_return["volatility"],
            maximum_return["return"]
        ),
        xytext=(15, -25),
        textcoords="offset points"
    )
    plt.gca().xaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    plt.gca().yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )


    plt.xlabel("Portfolio Volatility")
    plt.ylabel("Mean Return")

    plt.title(
        "3-Asset Portfolio Efficient Frontier"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    output_file = (
        figures_directory
        / "efficient_frontier.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print()
    print(
        f"Graph saved to: "
        f"{output_file}"
    )

    plt.show()
