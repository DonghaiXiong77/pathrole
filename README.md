# pathrole

Role-resolved, composition-adjusted pathway analysis for bulk transcriptomes.

[![tests](https://github.com/DonghaiXiong77/pathrole/actions/workflows/tests.yml/badge.svg)](https://github.com/DonghaiXiong77/pathrole/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Standard pathway enrichment pools every member of a pathway into one score.
For pathways whose members do different jobs — a receptor that *senses* a
signal, an effector that *executes* it, an inhibitor that *brakes* the effector
— that pooling hides the thing worth knowing. A tissue can transcribe the
sensing arm of a death pathway and none of its effectors, and conventional
enrichment will report the pathway as active.

A second problem compounds the first. Many pathways are expressed mainly by one
cell type, so a difference between two groups of **bulk** samples can reflect a
difference in *which cells are present* rather than in *what those cells are
doing*. The two claims are routinely conflated.

`pathrole` separates both:

| module | question it answers |
|---|---|
| `roles` | which members of this pathway sense, execute, or inhibit? |
| `scoring` | how active is each arm in each sample? |
| `singlecell` | which compartment actually expresses each arm? |
| `adjust` | does the group effect survive adjustment for composition? |
| `simulate` | does the adjustment work, on data where the answer is known? |

## Install

```bash
pip install git+https://github.com/DonghaiXiong77/pathrole.git        # from GitHub
pip install -e ".[test]"                   # from a clone, with pytest
```

Requires Python ≥ 3.10, numpy, pandas, scipy. Nothing else.

## Sixty-second tour

```python
import pathrole as pr

lib    = pr.builtin_library()              # or RoleLibrary.from_tsv("mine.tsv")
scores = pr.score_programmes(expr, lib)    # genes x samples  ->  samples x programmes
res    = pr.adjusted_effects(scores, groups, myeloid_fraction)

res.table[["beta_unadjusted", "beta_adjusted", "pct_attenuation", "fdr_adjusted"]]
res.composition_driven()                   # the ones that do not survive
```

From the command line:

```bash
pathrole run --expression tpm.csv --groups driver.csv \
             --composition deconv.csv --composition-columns Macrophages_M1 Macrophages_M2 \
             --out-prefix result
```

## Defining a role library

Three columns, one row per gene. Declare roles **before** looking at fold
changes — the sensing/execution contrast is only interpretable if membership
was not chosen with the answer in view.

```tsv
modality	role	gene
Pyroptosis	sensing	NLRP3
Pyroptosis	sensing	CASP1
Pyroptosis	execution	GSDMD
Pyroptosis	execution	GSDME
```

```python
lib = pr.RoleLibrary.from_tsv("library.tsv")
lib.paired_arms()          # modalities declaring both a sensing and execution arm
lib.coverage(expr.index)   # {programme: (genes present, genes declared)}
```

`sensing`, `execution` and `brake` are conventional, but the vocabulary is free
text — use whatever divides your pathway meaningfully.

## Checking the assumption behind the adjustment

Adjusting for a compartment's abundance is only the right move if the
programmes are actually restricted to that compartment. Measure it rather than
assume it:

```python
cell_scores = pr.score_programmes(sc_expr, lib)        # genes x cells
profile = pr.compartment_profile(cell_scores, cell_types, focus="Myeloid")
profile["rank_biserial_Myeloid"]      # +1 restricted to it, 0 not restricted
```

Rank-biserial correlation rather than a p value: with 10⁵ cells everything is
"significant", and the bounded effect size is the quantity that carries
information.

## Does the adjustment work?

`pathrole.simulate` generates cohorts where the truth is known: bulk samples
are mixtures of two compartments, some programmes have a real per-compartment
state effect and some have **exactly zero**, and the compartment fraction may
differ between groups.

```bash
pathrole demo --frac-delta 0.16 --replicates 40
```

```
40 replicate cohorts, compartment-1 fraction differing by 0.16 between groups
false positives on 400 null programmes (true effect exactly zero):
  unadjusted     400  (100.0%)
  adjusted         9  (  2.2%)
power on 400 real programmes, adjusted: 100.0%
```

Across 100 replicate cohorts of 200 samples, a compartment-fraction difference
of just **0.04** between groups — four percentage points — takes the unadjusted
false positive rate from 4.3% to **98.9%**, while the adjusted rate stays at
2.8%. Unadjusted *power* falls too (to 63% at a difference of 0.12), because
confounding masks real effects as readily as it manufactures false ones.

The control matters as much as the result: for programmes that are **not**
compartment-restricted (`--restriction 0.5`) the unadjusted rate stays at 3.2%
and adjustment changes nothing. The method addresses a specific failure mode
and is inert outside it.

## What this is not

- Not a deconvolution method. Supply composition estimates from whatever tool
  you trust; `pathrole` consumes them.
- Not a replacement for GSEA. Use enrichment to find candidate pathways, then
  `pathrole` to ask which arm moved and whether composition explains it.
- Not a causal method. Adjustment removes a measured confounder; it cannot
  speak to unmeasured ones, and the usual cautions about conditioning on a
  variable that may be a mediator apply.

## Citation

If `pathrole` is useful, please cite the paper describing it:

> Yang J, Xiong D, Huang M, Zheng J, Wang Y, Wang X, Yang C. Role-resolved,
> composition-adjusted pathway analysis separates tumour-intrinsic from
> infiltration-driven signals in papillary thyroid cancer. *Genomics* (under
> review).

The software release itself is archived at
[10.5281/zenodo.23485319](https://doi.org/10.5281/zenodo.23485319).

## License

MIT — see [LICENSE](LICENSE).
