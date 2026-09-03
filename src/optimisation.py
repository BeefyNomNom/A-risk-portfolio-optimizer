def calculate_portfolio_returns(
    returns_a,
    returns_b,
    returns_c,
    weight_a,
    weight_b,
    weight_c
):
    """
    Calculate portfolio returns from cool three assets
    """

    return (
        weight_a * returns_a
        + weight_b * returns_b
        + weight_c * returns_c
    )


def calculate_sharpe_ratio(
    returns,
    risk_free_rate=0.0
):
    """
    Calculate the Sharpe ratio.

    The Sharpe ratio measures return relative
    to volatility.
    """

    mean_return = returns.mean()
    volatility = returns.std()

    return (
        (mean_return - risk_free_rate)
        / volatility
    )


def find_efficient_frontier(portfolios):
    """
    Identify portfolios for the efficient frontier

    A portfolio is efficient if no portfolio with
    lower volatility has an >= return
    """

    sorted_portfolios = sorted(
        portfolios,
        key=lambda portfolio: portfolio["volatility"]
    )

    efficient = []

    highest_return = float("-inf")

    for portfolio in sorted_portfolios:

        if portfolio["return"] > highest_return:

            efficient.append(portfolio)

            highest_return = portfolio["return"]

    return efficient
