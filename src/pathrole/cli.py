"""Command line entry point: `pathrole run` and `pathrole demo`."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from pathrole import __version__
from pathrole.adjust import adjusted_effects
from pathrole.roles import RoleLibrary, builtin_library
from pathrole.scoring import score_programmes
from pathrole.simulate import simulate_cohort


def _read_table(path: str, index_col=0) -> pd.DataFrame:
    sep = '\t' if path.endswith(('.tsv', '.txt')) else ','
    return pd.read_csv(path, sep=sep, index_col=index_col)


def _cmd_run(a: argparse.Namespace) -> int:
    expr = _read_table(a.expression)
    library = (builtin_library() if a.library is None
               else (RoleLibrary.from_json(a.library) if a.library.endswith('.json')
                     else RoleLibrary.from_tsv(a.library)))
    groups = _read_table(a.groups).iloc[:, 0]
    composition = _read_table(a.composition) if a.composition else None

    cov = composition
    if composition is not None and a.composition_columns:
        missing = set(a.composition_columns) - set(composition.columns)
        if missing:
            print(f'error: composition columns not found: {sorted(missing)}',
                  file=sys.stderr)
            return 2
        cov = composition[a.composition_columns].sum(axis=1).rename('composition')

    scores = score_programmes(expr, library, log_transform=not a.already_log)
    res = adjusted_effects(scores, groups, cov, reference=a.reference)

    scores.to_csv(a.out_prefix + '_scores.csv')
    res.table.to_csv(a.out_prefix + '_effects.csv')

    ref, other = res.group_levels
    print(f'{len(scores.columns)} programmes scored over '
          f'{len(scores)} samples ({ref} n={res.n_per_group[0]}, '
          f'{other} n={res.n_per_group[1]})')
    if res.covariates:
        print(f'adjusted for: {", ".join(res.covariates)}')
    cols = ['beta_unadjusted', 'fdr_unadjusted', 'beta_adjusted',
            'fdr_adjusted', 'pct_attenuation']
    with pd.option_context('display.width', 140, 'display.max_columns', 20):
        print(res.table[cols].round(4).to_string())
    lost = res.composition_driven()
    if len(lost):
        print(f'\ncomposition-driven (significant before adjustment, not after '
              f'or heavily attenuated): {", ".join(lost.index)}')
    print(f'\nwrote {a.out_prefix}_scores.csv and {a.out_prefix}_effects.csv')
    return 0


def _cmd_demo(a: argparse.Namespace) -> int:
    """Replicate simulations, because a single draw is not a false-positive rate."""
    fp_naive = fp_adj = tp_adj = n_null = n_real = 0
    for rep in range(a.replicates):
        sim = simulate_cohort(frac_delta=a.frac_delta, seed=a.seed + rep)
        scores = score_programmes(sim.expression, sim.library)
        naive = adjusted_effects(scores, sim.groups, None)
        adj = adjusted_effects(scores, sim.groups, sim.fractions['compartment_1'])
        nulls, reals = sim.null_programmes, sim.real_programmes
        fp_naive += int((naive.table.loc[nulls, 'fdr_unadjusted'] < 0.05).sum())
        fp_adj += int((adj.table.loc[nulls, 'fdr_adjusted'] < 0.05).sum())
        tp_adj += int((adj.table.loc[reals, 'fdr_adjusted'] < 0.05).sum())
        n_null += len(nulls)
        n_real += len(reals)

    print(f'{a.replicates} replicate cohorts, compartment-1 fraction differing '
          f'by {a.frac_delta:.2f} between groups')
    print(f'false positives on {n_null} null programmes '
          f'(true effect exactly zero):')
    print(f'  unadjusted   {fp_naive:5d}  ({100 * fp_naive / n_null:5.1f}%)')
    print(f'  adjusted     {fp_adj:5d}  ({100 * fp_adj / n_null:5.1f}%)')
    print(f'power on {n_real} real programmes, adjusted: '
          f'{100 * tp_adj / n_real:.1f}%')
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog='pathrole',
        description='Role-resolved, composition-adjusted pathway analysis.')
    p.add_argument('--version', action='version', version=f'pathrole {__version__}')
    sub = p.add_subparsers(dest='command', required=True)

    r = sub.add_parser('run', help='score programmes and estimate group effects')
    r.add_argument('--expression', required=True,
                   help='genes x samples CSV/TSV, genes in the first column')
    r.add_argument('--groups', required=True,
                   help='CSV/TSV whose first column after the index is the group label')
    r.add_argument('--composition', help='samples x covariates CSV/TSV')
    r.add_argument('--composition-columns', nargs='+',
                   help='columns of --composition to sum into one covariate')
    r.add_argument('--library', help='role library .tsv or .json; '
                                     'omit to use the built-in example library')
    r.add_argument('--reference', help='group level to treat as baseline')
    r.add_argument('--already-log', action='store_true',
                   help='expression is already log-scale')
    r.add_argument('--out-prefix', default='pathrole')
    r.set_defaults(func=_cmd_run)

    d = sub.add_parser('demo', help='run the simulation benchmark once')
    d.add_argument('--frac-delta', type=float, default=0.08,
                   help='compartment-fraction difference between groups')
    d.add_argument('--replicates', type=int, default=50)
    d.add_argument('--seed', type=int, default=0)
    d.set_defaults(func=_cmd_demo)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == '__main__':
    raise SystemExit(main())
