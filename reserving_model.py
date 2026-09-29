"""
Chain-Ladder & Bornhuetter-Ferguson Claims Reserving Model
=============================================================

A from-scratch implementation of the two classic General Insurance claims-
reserving methods on the industry-standard "RAA" triangle (Reinsurance
Association of America, accident years 1981-1990): chain-ladder projects
each accident year to its ultimate cost using historical development
factors; Bornhuetter-Ferguson improves on it for the newest, thinnest years
by blending in an independent pricing-based prior. Both are cross-checked
against the industry-standard `chainladder` Python package, and reserve
uncertainty is quantified with a residual-resampling bootstrap, compared
against the package's full over-dispersed-Poisson bootstrap.

Built step by step as a learning project.

Author: Jabir Dhalla
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chainladder as cl

np.random.seed(42)
pd.set_option("display.float_format", lambda x: f"{x:,.1f}")

# ---------------------------------------------------------------------------
# 1. Load the triangle
# ---------------------------------------------------------------------------

raa = cl.load_sample("raa")
tri = raa.to_frame(origin_as_datetime=False)
cols = list(tri.columns)  # [12, 24, 36, ..., 120]

print("=== Raw cumulative claims triangle ===")
print(tri, "\n")

# ---------------------------------------------------------------------------
# 2. Chain-ladder, by hand
# ---------------------------------------------------------------------------

def dev_factors(triangle):
    """Volume-weighted average development factor for each column-to-column step."""
    f = {}
    for i in range(len(cols) - 1):
        c_from, c_to = cols[i], cols[i + 1]
        pair = triangle[[c_from, c_to]].dropna()
        f[(c_from, c_to)] = pair[c_to].sum() / pair[c_from].sum()
    return f


def project_ultimate(triangle, f):
    """Fill in the blanks of a triangle using development factors, and return
    the resulting total reserve (ultimate minus paid-to-date, summed)."""
    proj = triangle.copy()
    for row in proj.index:
        for i in range(len(cols) - 1):
            c_from, c_to = cols[i], cols[i + 1]
            if pd.isna(proj.loc[row, c_to]) and not pd.isna(proj.loc[row, c_from]):
                proj.loc[row, c_to] = proj.loc[row, c_from] * f[(c_from, c_to)]
    paid = triangle.apply(lambda r: r.dropna().iloc[-1], axis=1)
    return proj, paid


factors = dev_factors(tri)
projected, paid_to_date = project_ultimate(tri, factors)
ultimate = projected[cols[-1]]
reserve_cl = ultimate - paid_to_date

print("=== Chain-ladder development factors ===")
for (c_from, c_to), factor in factors.items():
    print(f"  {c_from:>3} -> {c_to:>3} months: {factor:.4f}")

print("\n=== Chain-ladder reserve by accident year ===")
print(pd.DataFrame({
    "Paid to date": paid_to_date,
    "Projected ultimate": ultimate,
    "Reserve": reserve_cl,
}))
print(f"\nTOTAL chain-ladder reserve: GBP {reserve_cl.sum():,.0f}\n")

# ---------------------------------------------------------------------------
# 3. Cross-check against the chainladder package
# ---------------------------------------------------------------------------

dev = cl.Development().fit_transform(raa)
cl_model = cl.Chainladder().fit(dev)
ibnr_package = cl_model.ibnr_.to_frame(origin_as_datetime=False).sum().sum()

print("=== Cross-check against the chainladder package ===")
print(f"By-hand total reserve:  GBP {reserve_cl.sum():,.0f}")
print(f"Package total reserve:  GBP {ibnr_package:,.0f}\n")

# ---------------------------------------------------------------------------
# 4. Bornhuetter-Ferguson
# ---------------------------------------------------------------------------

# "% developed" at each age = 1 / (product of all remaining development
# factors from that age to ultimate) -- derived purely from the pattern,
# never from an individual year's own paid amount, keeping the outside
# prior genuinely independent.
cdf_to_ultimate = {}
running = 1.0
for c in reversed(cols[:-1]):
    idx = cols.index(c)
    running *= factors[(cols[idx], cols[idx + 1])]
    cdf_to_ultimate[c] = running
cdf_to_ultimate[cols[-1]] = 1.0
pct_developed = pd.Series(cdf_to_ultimate).sort_index().rdiv(1)

current_age = tri.apply(lambda r: r.dropna().index[-1], axis=1)
pct_dev_by_year = current_age.map(pct_developed)

# Illustrative premiums and a 75% expected loss ratio, standing in for real
# pricing-team figures (the RAA dataset ships no premium data).
premium = pd.Series({
    1981: 22_000, 1982: 21_500, 1983: 24_500, 1984: 28_500, 1985: 29_500,
    1986: 22_000, 1987: 20_500, 1988: 26_500, 1989: 19_500, 1990: 21_000,
})
premium.index = tri.index
EXPECTED_LOSS_RATIO = 0.75
a_priori_ultimate = premium * EXPECTED_LOSS_RATIO

reserve_bf = a_priori_ultimate * (1 - pct_dev_by_year)

print("=== Bornhuetter-Ferguson reserve by accident year ===")
print(pd.DataFrame({
    "% developed": pct_dev_by_year,
    "A priori ultimate": a_priori_ultimate,
    "BF reserve": reserve_bf,
    "Chain-ladder reserve": reserve_cl,
}))
print(f"\nTOTAL Bornhuetter-Ferguson reserve: GBP {reserve_bf.sum():,.0f}\n")

# ---------------------------------------------------------------------------
# 5. Bootstrap -- quantifying reserve uncertainty
# ---------------------------------------------------------------------------

# Step 1: the "fitted" triangle -- anchor each year on its last known real
# value, then project forward (future columns) and backward (already-
# observed columns) using the same development factors, so every cell has a
# smooth, noise-free counterpart to compare against.
fitted_cum = pd.DataFrame(index=tri.index, columns=cols, dtype=float)
for row in tri.index:
    known = tri.loc[row].dropna()
    last_known_col = known.index[-1]
    anchor = known.iloc[-1]
    fitted_cum.loc[row, last_known_col] = anchor
    idx = cols.index(last_known_col)
    val = anchor
    for i in range(idx, len(cols) - 1):
        val = val * factors[(cols[i], cols[i + 1])]
        fitted_cum.loc[row, cols[i + 1]] = val
    val = anchor
    for i in range(idx, 0, -1):
        val = val / factors[(cols[i - 1], cols[i])]
        fitted_cum.loc[row, cols[i - 1]] = val

# Step 2: incremental values and Pearson residuals, on observed cells only.
actual_incr = tri.copy()
fitted_incr = fitted_cum.copy()
for i in range(len(cols) - 1, 0, -1):
    actual_incr[cols[i]] = tri[cols[i]] - tri[cols[i - 1]]
    fitted_incr[cols[i]] = fitted_cum[cols[i]] - fitted_cum[cols[i - 1]]

observed_mask = tri.notna()
residuals = ((actual_incr - fitted_incr) / np.sqrt(fitted_incr.clip(lower=1)))[observed_mask]
resid_pool = residuals.values.flatten()
resid_pool = resid_pool[~pd.isna(resid_pool)]

# Step 3-5: resample residuals, rebuild a pseudo triangle, re-run
# chain-ladder, repeat thousands of times.
N_SIMS = 5_000
results_by_hand = np.zeros(N_SIMS)
fitted_incr_arr = fitted_incr.values
observed_arr = observed_mask.values

for s in range(N_SIMS):
    sampled = np.random.choice(resid_pool, size=fitted_incr_arr.shape, replace=True)
    pseudo_incr = fitted_incr_arr + sampled * np.sqrt(np.clip(fitted_incr_arr, 0, None))
    pseudo_incr = np.where(observed_arr, pseudo_incr, np.nan)
    pseudo_incr_df = pd.DataFrame(pseudo_incr, index=tri.index, columns=cols)
    pseudo_cum = pseudo_incr_df.cumsum(axis=1).where(observed_mask)
    f_pseudo = dev_factors(pseudo_cum)
    proj_pseudo, paid_pseudo = project_ultimate(tri, f_pseudo)
    results_by_hand[s] = (proj_pseudo[cols[-1]] - paid_pseudo).sum()

print("=== Bootstrap -- parameter risk only (by hand) ===")
print(f"Mean reserve:    GBP {results_by_hand.mean():,.0f}")
print(f"5th percentile:  GBP {np.percentile(results_by_hand, 5):,.0f}")
print(f"95th percentile: GBP {np.percentile(results_by_hand, 95):,.0f}\n")

# Cross-check: the package's full over-dispersed-Poisson bootstrap, which
# adds process risk (the randomness of individual claim outcomes) on top of
# parameter risk.
sims = cl.BootstrapODPSample(n_sims=N_SIMS, random_state=42).fit_transform(raa)
odp_model = cl.Chainladder().fit(sims)
results_package = odp_model.ibnr_.sum("origin").to_frame().values.flatten()

print("=== Bootstrap -- parameter + process risk (package's full ODP bootstrap) ===")
print(f"Mean reserve:    GBP {results_package.mean():,.0f}")
print(f"5th percentile:  GBP {np.percentile(results_package, 5):,.0f}")
print(f"95th percentile: GBP {np.percentile(results_package, 95):,.0f}\n")

# ---------------------------------------------------------------------------
# 6. Chart: both bootstrap distributions, side by side
# ---------------------------------------------------------------------------

plt.figure(figsize=(9, 5))
plt.hist(results_by_hand, bins=60, alpha=0.6, color="#4C72B0",
         label="Parameter risk only (by hand)")
plt.hist(results_package, bins=60, alpha=0.6, color="#DD8452",
         label="Parameter + process risk (package)")
plt.axvline(reserve_cl.sum(), color="black", linestyle="--", linewidth=2,
            label=f"Chain-ladder point estimate: GBP {reserve_cl.sum():,.0f}")
plt.xlabel("Simulated total reserve (GBP)")
plt.ylabel("Number of simulations")
plt.title("Bootstrap distribution of the total reserve (5,000 simulations)")
plt.legend()
plt.tight_layout()
plt.savefig("reserving_bootstrap_chart.png", dpi=150)
print("Saved chart to reserving_bootstrap_chart.png")
