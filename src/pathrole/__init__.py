"""pathrole - role-resolved, composition-adjusted pathway analysis.

Standard pathway enrichment pools every member of a pathway into one score. For
pathways whose members do different jobs - a receptor that senses a signal, an
effector that executes it, an inhibitor that blocks the effector - that pooling
hides the thing worth knowing: a tissue can transcribe the sensing arm of a
pathway and none of its effectors, and conventional enrichment will report the
pathway as active.

A second problem compounds the first in bulk tissue. Many pathways are expressed
mainly by one cell type, so a difference between two groups of bulk samples can
reflect a difference in cellular composition rather than in the cells' state.

pathrole addresses both:

    roles      declare which members of a pathway sense, execute or inhibit
    scoring    score those role-resolved programmes per sample
    singlecell measure which compartment expresses each programme
    adjust     re-estimate each group effect with composition as a covariate,
               and report how much of the effect that removes

The package is deliberately small and makes no assumptions about organism,
assay or disease: it needs an expression matrix, a group label per sample, a
composition estimate per sample, and a role-annotated gene set library.
"""

from pathrole.adjust import AdjustedEffects, adjusted_effects
from pathrole.roles import RoleLibrary, Programme, builtin_library
from pathrole.scoring import score_programmes
from pathrole.simulate import simulate_cohort
from pathrole.singlecell import compartment_profile

__version__ = '1.0.0'

__all__ = [
    'RoleLibrary', 'Programme', 'builtin_library',
    'score_programmes', 'compartment_profile',
    'adjusted_effects', 'AdjustedEffects',
    'simulate_cohort',
]
