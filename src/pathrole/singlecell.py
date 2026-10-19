"""Which compartment expresses each programme.

The composition covariate in `pathrole.adjust` is only the right covariate if
the programmes really are compartment-restricted. This module measures that
from a single-cell reference, so the assumption behind the adjustment is
checked rather than asserted.

Effect size is the rank-biserial correlation rather than a difference in means
or a p value. With tens of thousands of cells every comparison is "significant",
so a p value carries no information; the rank-biserial correlation is bounded
in [-1, 1], is the Mann-Whitney U rescaled, and answers the question actually
being asked - how often a cell of one type scores above a cell of another.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathrole.roles import RoleLibrary
from pathrole.scoring import score_programmes


def rank_biserial(a, b) -> float:
    """Rank-biserial correlation: +1 if every a exceeds every b, -1 if the
    reverse, 0 if the two are interleaved."""
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError('both groups must be non-empty')
    u = stats.mannwhitneyu(a, b, alternative='two-sided').statistic
    return float(2 * u / (a.size * b.size) - 1)


def compartment_profile(cell_scores: pd.DataFrame,
                        labels: pd.Series,
                        *,
                        focus: str | None = None,
                        min_cells: int = 20) -> pd.DataFrame:
    """Mean programme score per compartment, and optionally a contrast.

    Parameters
    ----------
    cell_scores
        Cells in rows, programmes in columns.
    labels
        Compartment label per cell.
    focus
        If given, also return the rank-biserial correlation of this
        compartment against all other cells for every programme - the
        quantity that tells you whether a programme is restricted to it.
    min_cells
        Compartments with fewer cells are dropped.

    Returns
    -------
    Programmes in rows, compartments in columns, plus `n_cells` as a row and,
    when `focus` is set, a `rank_biserial_<focus>` column.
    """
    labels = pd.Series(labels).dropna().astype(str)
    idx = cell_scores.index.intersection(labels.index)
    if len(idx) == 0:
        raise ValueError('cell_scores and labels share no cells')
    sc, lab = cell_scores.loc[idx], labels.loc[idx]

    counts = lab.value_counts()
    keep = counts[counts >= min_cells].index
    if len(keep) == 0:
        raise ValueError(f'no compartment has at least {min_cells} cells')
    if focus is not None and focus not in keep:
        raise ValueError(f'focus compartment {focus!r} is absent or too small; '
                         f'available: {sorted(keep)}')
    sc, lab = sc[lab.isin(keep)], lab[lab.isin(keep)]

    out = sc.groupby(lab, observed=True).mean().T
    out.loc['n_cells'] = lab.value_counts().reindex(out.columns)

    if focus is not None:
        inside = lab == focus
        rb = {p: rank_biserial(sc.loc[inside, p], sc.loc[~inside, p])
              for p in sc.columns}
        out[f'rank_biserial_{focus}'] = pd.Series(rb)

    return out


def compartment_fractions(labels: pd.Series,
                          samples: pd.Series,
                          *,
                          compartments: list[str] | None = None) -> pd.DataFrame:
    """Per-sample compartment fractions from single-cell labels.

    Use this when the single-cell data come from the same samples as the bulk.
    When they do not - the usual case - estimate the fractions by deconvolving
    the bulk profiles against a single-cell reference with a dedicated tool and
    pass the result to :func:`pathrole.adjust.adjusted_effects` directly.
    """
    labels = pd.Series(labels).astype(str)
    samples = pd.Series(samples).astype(str)
    idx = labels.index.intersection(samples.index)
    if len(idx) == 0:
        raise ValueError('labels and samples share no cells')
    tab = pd.crosstab(samples.loc[idx], labels.loc[idx], normalize='index')
    if compartments is not None:
        missing = set(compartments) - set(tab.columns)
        if missing:
            raise ValueError(f'compartments absent from the data: {sorted(missing)}')
        tab = tab[compartments]
    return tab


def score_cells(expr: pd.DataFrame, library: RoleLibrary, **kw) -> pd.DataFrame:
    """Score programmes per cell. `expr` is genes x cells, as from scoring."""
    return score_programmes(expr, library, **kw)
