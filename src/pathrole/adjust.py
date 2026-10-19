"""Composition-adjusted group effects.

For each programme the group effect is estimated twice: once from group label
alone, and once with one or more composition covariates added. Both estimates
are reported, together with the attenuation between them.

Reporting both is the point. A programme whose effect survives adjustment is a
difference in the cells' state; one whose effect disappears is a difference in
which cells are present. Neither is uninteresting, but they are different
claims, and an unadjusted analysis cannot tell them apart.

Attenuation is undefined when the unadjusted effect is near zero - the ratio of
two small numbers is noise - so it is returned as NaN below a threshold rather
than printed as a large meaningless percentage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares with standard errors and two-sided t p values."""
    from scipy import stats

    n, k = X.shape
    if n <= k:
        raise ValueError(f'{n} samples cannot fit {k} parameters')
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, beta / se, 0.0)
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, p


@dataclass
class AdjustedEffects:
    """Result of :func:`adjusted_effects`."""

    table: pd.DataFrame
    group_levels: tuple[str, str]
    covariates: tuple[str, ...]
    n_per_group: tuple[int, int]

    def significant(self, alpha: float = 0.05, adjusted: bool = True) -> pd.DataFrame:
        col = 'fdr_adjusted' if adjusted else 'fdr_unadjusted'
        return self.table[self.table[col] < alpha]

    def composition_driven(self, alpha: float = 0.05,
                           min_attenuation: float = 50.0) -> pd.DataFrame:
        """Programmes significant before adjustment that do not survive it,
        or that lose more than `min_attenuation` percent of their effect."""
        t = self.table
        lost = (t['fdr_unadjusted'] < alpha) & (t['fdr_adjusted'] >= alpha)
        shrunk = t['pct_attenuation'] >= min_attenuation
        return t[lost | (shrunk & (t['fdr_unadjusted'] < alpha))]

    def __repr__(self) -> str:
        a, b = self.group_levels
        return (f'AdjustedEffects({len(self.table)} programmes, '
                f'{a} n={self.n_per_group[0]} vs {b} n={self.n_per_group[1]}, '
                f'covariates={list(self.covariates)})')


def adjusted_effects(scores: pd.DataFrame,
                     groups: pd.Series,
                     composition: pd.DataFrame | pd.Series | None = None,
                     *,
                     reference: str | None = None,
                     extra_covariates: pd.DataFrame | None = None,
                     attenuation_floor: float = 0.1) -> AdjustedEffects:
    """Estimate each programme's group effect with and without composition.

    Parameters
    ----------
    scores
        Samples in rows, programmes in columns (see :func:`score_programmes`).
    groups
        One label per sample. Exactly two distinct labels must remain after
        dropping missing values.
    composition
        Samples in rows, composition covariates in columns - for example the
        estimated fraction of the compartment that expresses the programmes.
        A Series is treated as a single covariate. If None, only the
        unadjusted model is fitted and the adjusted columns repeat it.
    reference
        The group level to treat as baseline; effects are reported as
        other minus reference. Defaults to the alphabetically first level.
    extra_covariates
        Further per-sample covariates to include in BOTH models - use this for
        nuisance variables such as batch or purity that are not the
        composition question under test.
    attenuation_floor
        Attenuation is reported as NaN when the unadjusted coefficient's
        absolute value is below this, because the ratio is then meaningless. The default of 0.1 is
        conservative: an effect smaller than that is not one anybody reports.

    Returns
    -------
    AdjustedEffects with one row per programme.
    """
    if scores.empty:
        raise ValueError('no programme scores supplied')

    groups = pd.Series(groups).dropna().astype(str)
    idx = scores.index.intersection(groups.index)
    if composition is not None:
        comp = composition.to_frame() if isinstance(composition, pd.Series) \
            else composition.copy()
        idx = idx.intersection(comp.dropna().index)
    else:
        comp = None
    if extra_covariates is not None:
        idx = idx.intersection(extra_covariates.dropna().index)
    if len(idx) == 0:
        raise ValueError('scores, groups and composition share no samples; '
                         'check that the indices use the same identifiers')

    g = groups.loc[idx]
    levels = sorted(g.unique())
    if len(levels) != 2:
        raise ValueError(f'exactly two groups required, found {levels}')
    if reference is None:
        reference = levels[0]
    if reference not in levels:
        raise ValueError(f'reference {reference!r} is not one of {levels}')
    other = [l for l in levels if l != reference][0]

    is_other = (g == other).to_numpy(dtype=float)
    n_ref, n_other = int((~g.eq(other)).sum()), int(g.eq(other).sum())
    if min(n_ref, n_other) < 3:
        raise ValueError(f'each group needs at least 3 samples, '
                         f'got {reference}={n_ref}, {other}={n_other}')

    base_cols = [np.ones(len(idx)), is_other]
    if extra_covariates is not None:
        extra = extra_covariates.loc[idx].apply(pd.to_numeric, errors='coerce')
        base_cols += [extra[c].to_numpy(dtype=float) for c in extra.columns]
    X_un = np.column_stack(base_cols)

    if comp is not None:
        comp = comp.loc[idx].apply(pd.to_numeric, errors='coerce')
        cov_names = tuple(comp.columns)
        X_adj = np.column_stack(base_cols + [comp[c].to_numpy(dtype=float)
                                             for c in comp.columns])
    else:
        cov_names = ()
        X_adj = X_un

    rows = []
    for prog in scores.columns:
        y = scores.loc[idx, prog].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError(f'programme {prog!r} has non-finite scores')
        b_un, se_un, p_un = _ols(y, X_un)
        b_ad, se_ad, p_ad = _ols(y, X_adj)
        beta0, beta1 = float(b_un[1]), float(b_ad[1])
        att = (100.0 * (1 - beta1 / beta0)
               if abs(beta0) >= attenuation_floor else np.nan)
        row = {'programme': prog,
               'beta_unadjusted': beta0, 'se_unadjusted': float(se_un[1]),
               'p_unadjusted': float(p_un[1]),
               'beta_adjusted': beta1, 'se_adjusted': float(se_ad[1]),
               'p_adjusted': float(p_ad[1]),
               'pct_attenuation': att}
        for j, name in enumerate(cov_names):
            k = X_adj.shape[1] - len(cov_names) + j
            row[f'beta_{name}'] = float(b_ad[k])
            row[f'p_{name}'] = float(p_ad[k])
        rows.append(row)

    table = pd.DataFrame(rows).set_index('programme')
    table['fdr_unadjusted'] = _bh(table['p_unadjusted'].to_numpy())
    table['fdr_adjusted'] = _bh(table['p_adjusted'].to_numpy())
    front = ['beta_unadjusted', 'p_unadjusted', 'fdr_unadjusted',
             'beta_adjusted', 'p_adjusted', 'fdr_adjusted', 'pct_attenuation']
    table = table[front + [c for c in table.columns if c not in front]]

    return AdjustedEffects(table=table, group_levels=(reference, other),
                           covariates=cov_names, n_per_group=(n_ref, n_other))
