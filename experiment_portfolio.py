"""Experiment-portfolio simulator and decision rules (ECON 5200 PS2)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def simulate_portfolio(n: int, seed: int, **kwargs):
    """Return the experiment log with the true effect attached."""
    rn = np.random.default_rng(seed)

    share = kwargs.get("share", 0.25)
    effect_mean = kwargs.get("effect_mean", 0.30)
    effect_sd = kwargs.get("effect_sd", 0.10)
    sample_size_low = kwargs.get("sample_size_low", 40)
    sample_size_high = kwargs.get("sample_size_high", 400)
    outcome_sd = kwargs.get("outcome_sd", 1.0)

    real = rn.random(n) < share

    truth = np.where(
        real,
        rn.normal(effect_mean, effect_sd, size=n),
        0.0
    )

    sample_size = rn.integers(
        sample_size_low, sample_size_high + 1, size=n
    )
    standard_error = outcome_sd * np.sqrt(2 / sample_size)

    estimate = truth + rn.normal(0, standard_error)

    log = pd.DataFrame({
        "Truth": truth,
        "Sample_Size": sample_size,
        "Outcome_SD": outcome_sd,
        "Standard_Error": standard_error,
        "Effect_Estimate": estimate
    })

    z_score = log["Effect_Estimate"] / log["Standard_Error"]
    log["p_value"] = 2 * stats.norm.sf(np.abs(z_score))
    log["Cohens_d"] = log["Effect_Estimate"] / outcome_sd

    alpha = 0.05
    minimum_effect = 0.30
    z_critical = stats.norm.ppf(1 - alpha / 2)

    noncentrality = minimum_effect / log["Standard_Error"]

    log["Power"] = (
        stats.norm.cdf(-z_critical - noncentrality)
        + stats.norm.sf(z_critical - noncentrality)
    )

    return log


def apply_rule(log, rule):
    """Ship decisions under `rule`: the p < 0.05 threshold and at least one other."""
    if rule == "p":
        return log["p_value"] < 0.05

    if rule == "cohens_d":
        return log["Cohens_d"] > 0.2

    if rule == "power":
        return (log["p_value"] < 0.05) & (log["Power"] >= 0.80)

    raise ValueError(
        "Unknown rule. Use 'p' or 'cohens_d'."
    )


def evaluate(log, rule):
    """Realised true return, ship count and false-ship rate under `rule`."""
    shipped = log[apply_rule(log, rule)]
    ship_count = len(shipped)

    if ship_count == 0:
        return {
            "Realized Return": 0.0,
            "Number Shipped": 0,
            "False-Ship Rate": np.nan,
            "Average True Effect": np.nan,
            "Average Estimated Effect": np.nan
        }

    return {
        "Realized Return": shipped["Truth"].sum(),
        "Number Shipped": ship_count,
        "False-Ship Rate": (shipped["Truth"] == 0).mean(),
        "Average True Effect": shipped["Truth"].mean(),
        "Average Estimated Effect": shipped["Effect_Estimate"].mean()
    }



if __name__ == "__main__":
    log = simulate_portfolio(200, seed=3)

    print("Client rule:", evaluate(log, "p"))
    print("Cohen's d rule:", evaluate(log, "cohens_d"))

    same_log = simulate_portfolio(200, seed=3)
    assert log.equals(same_log), "Same seed should reproduce the results."

    client_ships = apply_rule(log, "p")
    assert (client_ships == (log["p_value"] < 0.05)).all()

    d_ships = apply_rule(log, "cohens_d")
    assert (d_ships == (log["Cohens_d"] > 0.2)).all()

    assert 0 <= client_ships.sum() <= 200

    print("All assertions passed.")
