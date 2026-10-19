"""Simulate a bulk cohort with known state effects and known confounding.

The generative model is the one that makes composition confounding real: a bulk
sample is a mixture of compartments, so its expression of a compartment-
restricted gene depends on how much of that compartment is present, whether or
not the cells themselves differ between groups.

For a gene j belonging to a programme with compartment weights w (summing to 1
across compartments) and a true per-compartment state effect d_j:

    x_j(s) = sum_c f_c(s) * w_c * (base_j + d_j * group(s)) + e

where f_c(s) is the fraction of compartment c in sample s and e is Gaussian
noise. Setting d_j = 0 for some programmes while letting f differ by group is
what produces a null programme that a naive analysis will nonetheless call
significant - the false positive the adjustment is meant to prevent.

The simulator is part of the package rather than the paper's analysis code so
that the behaviour claimed for the method can be re-checked by anyone, on their
own parameter choices, with one function call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pathrole.roles import Programme, RoleLibrary


@dataclass
class SimulatedCohort:
    """Everything the simulator knows, including the answers."""

    expression: pd.DataFrame        # genes x samples, linear scale
    groups: pd.Series               # 'A' / 'B' per sample
    fractions: pd.DataFrame         # samples x compartments, true fractions
    library: RoleLibrary
    truth: pd.DataFrame             # programme x (true_effect, compartment)

    @property
    def null_programmes(self) -> list[str]:
        return self.truth.index[self.truth['true_effect'] == 0].tolist()

    @property
    def real_programmes(self) -> list[str]:
        return self.truth.index[self.truth['true_effect'] != 0].tolist()


def simulate_cohort(*,
                    n_samples: int = 200,
                    n_genes_per_programme: int = 10,
                    n_null_programmes: int = 10,
                    n_real_programmes: int = 10,
                    effect_size: float = 0.8,
                    frac_mean: float = 0.25,
                    frac_delta: float = 0.08,
                    frac_sd: float = 0.06,
                    restriction: float = 0.9,
                    base_level: float = 6.0,
                    noise_sd: float = 0.35,
                    group_imbalance: float = 0.5,
                    seed: int = 0) -> SimulatedCohort:
    """Generate one cohort.

    Parameters
    ----------
    n_null_programmes, n_real_programmes
        How many programmes have a true per-compartment effect of exactly zero,
        and how many have `effect_size`. Null programmes are the ones a naive
        analysis should not call.
    frac_delta
        Difference in mean compartment-1 fraction between the two groups - the
        strength of the confounding. Zero means no confounding.
    restriction
        Fraction of a restricted programme's signal that comes from its own
        compartment. 0.9 means strongly compartment-restricted; 0.5 means not
        restricted at all, and the adjustment should then have little to do.
    group_imbalance
        Proportion of samples in group B.
    noise_sd
        Gaussian noise on the log scale before exponentiation.

    Returns
    -------
    SimulatedCohort, whose `truth` records the answer for every programme.
    """
    if not 0 < group_imbalance < 1:
        raise ValueError('group_imbalance must be strictly between 0 and 1')
    if not 0 <= restriction <= 1:
        raise ValueError('restriction must be between 0 and 1')
    if n_null_programmes < 1 or n_real_programmes < 1:
        raise ValueError('need at least one null and one real programme')
    rng = np.random.default_rng(seed)

    samples = [f'S{i:04d}' for i in range(n_samples)]
    n_b = int(round(n_samples * group_imbalance))
    labels = np.array(['A'] * (n_samples - n_b) + ['B'] * n_b)
    rng.shuffle(labels)
    groups = pd.Series(labels, index=samples, name='group')
    is_b = (groups == 'B').to_numpy(dtype=float)

    # compartment 1 is the confounded one; its abundance differs by group
    f1 = np.clip(rng.normal(frac_mean + frac_delta * (is_b - 0.5),
                            frac_sd, n_samples), 0.01, 0.99)
    fractions = pd.DataFrame({'compartment_1': f1, 'compartment_2': 1 - f1},
                             index=samples)

    progs, rows, blocks = [], [], []
    specs = ([('null', 0.0) for _ in range(n_null_programmes)]
             + [('real', effect_size) for _ in range(n_real_programmes)])
    for k, (kind, eff) in enumerate(specs):
        # alternate which compartment the programme is restricted to, so the
        # benchmark is not a special case of one compartment
        comp = 'compartment_1' if k % 2 == 0 else 'compartment_2'
        w = np.where(fractions.columns == comp, restriction, 1 - restriction)
        genes = [f'{kind.upper()}{k:02d}_G{j:02d}'
                 for j in range(n_genes_per_programme)]
        prog = Programme(f'{kind} programme {k:02d}', 'simulated', tuple(genes))
        progs.append(prog)
        rows.append({'programme': prog.name, 'true_effect': eff,
                     'compartment': comp})

        base = base_level + rng.normal(0, 0.4, (len(genes), 1))
        # per-compartment expression: the state effect acts inside the cells
        state = base + eff * is_b[None, :]
        mix = (fractions.to_numpy() * w[None, :]).sum(axis=1)   # per-sample weight
        signal = state * mix[None, :]
        obs = signal + rng.normal(0, noise_sd, signal.shape)
        blocks.append(pd.DataFrame(np.exp2(obs) - 1, index=genes, columns=samples))

    expression = pd.concat(blocks).clip(lower=0)
    truth = pd.DataFrame(rows).set_index('programme')
    return SimulatedCohort(expression=expression, groups=groups,
                           fractions=fractions, library=RoleLibrary(progs),
                           truth=truth)
