import numpy as np


def calculate_var(returns, confidence_level=0.95):
    """
    Calculate Risk using percentile method.
    """

    percentile = (1 - confidence_level) * 100

    return np.percentile(
        returns,
        percentile
    )


def calculate_expected_shortfall(
    returns,
    confidence_level=0.95
):
    """
    Calculate Shortfall

    This is the average return among outcomes
    worse than the VaR threshold.
    """

    var = calculate_var(
        returns,
        confidence_level
    )

    tail_losses = returns[returns <= var]

    return tail_losses.mean()
