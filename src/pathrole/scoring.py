"""Per-sample programme scores.

A programme score is the mean, across the programme's genes, of each gene's
z-score over the cohort. This is deliberately the simplest defensible choice:
it is scale-free, it weights every member equally, and it has no tuning
parameters that could be adjusted after seeing the result. Because every gene
is centred across samples, a score of zero means "average for this cohort", and
differences between groups are what the adjustment in `pathrole.adjust` then
models.

The alternative - a rank-based enrichment score per sample - is more robust to
outliers but is not comparable across programmes of different size, which is
exactly the comparison the sensing/execution contrast requires.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pathrole.roles import RoleLibrary


def _zscore(expr: pd.DataFrame) -> pd.DataFrame:
    sd = expr.std(axis=1).replace(0, np.nan)
    return expr.sub(expr.mean(axis=1), axis=0).div(sd, axis=0)


def score_programmes(expr: pd.DataFrame,
                     library: RoleLibrary,
                     *,
                     log_transform: bool = True,
                     min_genes: int = 3,
                     on_missing: str = 'warn') -> pd.DataFrame:
    """Score every programme in `library` for every sample in `expr`.

    Parameters
    ----------
    expr
        Genes in rows, samples in columns. Values on a linear scale unless
        `log_transform=False`.
    library
        The role-annotated programmes to score.
    log_transform
        Apply log2(x + 1) before z-scoring. Set False if `expr` is already
        log-scale.
    min_genes
        A programme with fewer than this many genes present in `expr` is
        dropped rather than scored from a handful of genes.
    on_missing
        'warn' (default), 'ignore', or 'raise' - what to do when a programme
        falls below `min_genes`.

    Returns
    -------
    DataFrame, samples in rows and programme names in columns.
    """
    if on_missing not in ('warn', 'ignore', 'raise'):
        raise ValueError("on_missing must be 'warn', 'ignore' or 'raise'")
    if expr.empty:
        raise ValueError('expression matrix is empty')
    if expr.index.has_duplicates:
        raise ValueError('expression matrix has duplicate gene names; '
                         'collapse them before scoring')

    mat = expr.apply(pd.to_numeric, errors='coerce')
    if log_transform:
        if (mat.to_numpy(dtype=float) < 0).any():
            raise ValueError('negative values with log_transform=True; '
                             'pass log_transform=False for log-scale input')
        mat = np.log2(mat + 1)
    z = _zscore(mat)

    scores, dropped = {}, []
    for prog in library:
        have = prog.present(z.index)
        if len(have) < min_genes:
            dropped.append(f'{prog.name} ({len(have)}/{len(prog)} genes)')
            continue
        scores[prog.name] = z.loc[list(have)].mean(axis=0)

    if dropped:
        msg = ('programmes dropped for too few genes in the expression matrix: '
               + '; '.join(dropped))
        if on_missing == 'raise':
            raise ValueError(msg)
        if on_missing == 'warn':
            warnings.warn(msg, stacklevel=2)
    if not scores:
        raise ValueError('no programme had enough genes present to score')

    return pd.DataFrame(scores)
