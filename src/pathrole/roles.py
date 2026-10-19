"""Role-annotated gene set libraries.

A Programme is a set of genes that share a pathway *and* a functional role
within it. Declaring the role is the whole point: `Programme('Pyroptosis',
'execution', ['GSDMD', ...])` asserts that these genes are the ones that commit
a cell to pyroptosis, so an analysis can ask about them separately from the
sensors upstream.

Roles are free text so the vocabulary can suit the pathway, but three are
conventional and the built-in library uses only these:

    sensing    receptors, sensors, adaptors and initiator enzymes
    execution  the effectors that commit the cell to the outcome
    brake      endogenous inhibitors of those effectors

Declaring roles before looking at the data is a design requirement, not a
suggestion: the comparison between arms is only interpretable if membership was
not chosen with the fold changes in view.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

CONVENTIONAL_ROLES = ('sensing', 'execution', 'brake')


@dataclass(frozen=True)
class Programme:
    """One pathway arm: a modality, a role within it, and its genes."""

    modality: str
    role: str
    genes: tuple[str, ...]

    def __post_init__(self):
        if not self.modality or not self.modality.strip():
            raise ValueError('modality must be a non-empty string')
        if not self.role or not self.role.strip():
            raise ValueError('role must be a non-empty string')
        genes = tuple(dict.fromkeys(g.strip() for g in self.genes if g and g.strip()))
        if len(genes) < 2:
            raise ValueError(
                f'{self.modality}/{self.role}: a programme needs at least 2 genes, '
                f'got {len(genes)}')
        object.__setattr__(self, 'genes', genes)

    @property
    def name(self) -> str:
        return f'{self.modality} {self.role}'

    def present(self, universe: Iterable[str]) -> tuple[str, ...]:
        """The programme's genes that occur in `universe`, order preserved."""
        have = set(universe)
        return tuple(g for g in self.genes if g in have)

    def __len__(self) -> int:
        return len(self.genes)


class RoleLibrary:
    """An ordered collection of Programmes with unique names."""

    def __init__(self, programmes: Sequence[Programme]):
        if not programmes:
            raise ValueError('a library needs at least one programme')
        names = [p.name for p in programmes]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f'duplicate programme names: {dupes}')
        self._programmes = tuple(programmes)

    # ------------------------------------------------------------- access
    def __iter__(self) -> Iterator[Programme]:
        return iter(self._programmes)

    def __len__(self) -> int:
        return len(self._programmes)

    def __getitem__(self, name: str) -> Programme:
        for p in self._programmes:
            if p.name == name:
                return p
        raise KeyError(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._programmes)

    def by_role(self, role: str) -> 'RoleLibrary':
        sel = [p for p in self._programmes if p.role == role]
        if not sel:
            raise KeyError(f'no programme has role {role!r}; '
                           f'roles present: {sorted({p.role for p in self})}')
        return RoleLibrary(sel)

    def by_modality(self, modality: str) -> 'RoleLibrary':
        sel = [p for p in self._programmes if p.modality == modality]
        if not sel:
            raise KeyError(f'no programme has modality {modality!r}')
        return RoleLibrary(sel)

    def paired_arms(self) -> list[tuple[str, Programme, Programme]]:
        """Modalities that declare both a sensing and an execution arm.

        These are the comparisons the package exists to make, so they are
        enumerated explicitly rather than left to the caller to find.
        """
        out = []
        for mod in dict.fromkeys(p.modality for p in self._programmes):
            arms = {p.role: p for p in self._programmes if p.modality == mod}
            if 'sensing' in arms and 'execution' in arms:
                out.append((mod, arms['sensing'], arms['execution']))
        return out

    # --------------------------------------------------------------- i/o
    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> 'RoleLibrary':
        grouped: dict[tuple[str, str], list[str]] = {}
        for i, r in enumerate(rows, start=1):
            try:
                key = (r['modality'].strip(), r['role'].strip())
                gene = r['gene'].strip()
            except KeyError as e:
                raise ValueError(
                    f'row {i}: missing column {e}; '
                    'expected columns modality, role, gene') from None
            if gene:
                grouped.setdefault(key, []).append(gene)
        if not grouped:
            raise ValueError('no programmes found in input')
        return cls([Programme(m, r, tuple(g)) for (m, r), g in grouped.items()])

    @classmethod
    def from_tsv(cls, path: str | Path) -> 'RoleLibrary':
        """Read a three-column TSV with a header: modality, role, gene."""
        with open(path, newline='', encoding='utf-8') as fh:
            return cls.from_rows(csv.DictReader(fh, delimiter='\t'))

    @classmethod
    def from_json(cls, path: str | Path) -> 'RoleLibrary':
        """Read {"Modality": {"role": ["GENE", ...], ...}, ...}."""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        progs = [Programme(mod, role, tuple(genes))
                 for mod, roles in data.items()
                 for role, genes in roles.items()]
        return cls(progs)

    def to_tsv(self, path: str | Path) -> None:
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh, delimiter='\t')
            w.writerow(['modality', 'role', 'gene'])
            for p in self._programmes:
                for g in p.genes:
                    w.writerow([p.modality, p.role, g])

    def coverage(self, universe: Iterable[str]) -> dict[str, tuple[int, int]]:
        """{programme name: (genes present, genes declared)}."""
        uni = set(universe)
        return {p.name: (sum(g in uni for g in p.genes), len(p.genes))
                for p in self._programmes}

    def __repr__(self) -> str:
        return (f'RoleLibrary({len(self)} programmes, '
                f'{len({p.modality for p in self})} modalities)')


# --------------------------------------------------------------- built-in
_RCD = {
    'Pyroptosis': {
        'sensing': ['NLRP3', 'NLRC4', 'AIM2', 'NLRP1', 'PYCARD',
                    'CASP1', 'CASP4', 'CASP5', 'IL1B', 'IL18'],
        'execution': ['GSDMD', 'GSDME', 'GSDMA', 'GSDMB', 'GSDMC'],
    },
    'Extrinsic apoptosis': {
        'sensing': ['FAS', 'FASLG', 'TNFRSF1A', 'TNFRSF1B', 'TNF',
                    'TNFRSF10A', 'TNFRSF10B', 'TNFSF10', 'FADD', 'CASP8'],
    },
    'Intrinsic apoptosis': {
        'execution': ['BAX', 'BAK1', 'APAF1', 'CASP9', 'CASP3', 'CASP7', 'CYCS'],
        'brake': ['CFLAR', 'BCL2', 'BCL2L1', 'MCL1', 'BIRC2', 'BIRC3', 'XIAP'],
    },
    'Necroptosis': {
        'sensing': ['ZBP1', 'TICAM1', 'TNFAIP3', 'CYLD'],
        'execution': ['RIPK1', 'RIPK3', 'MLKL'],
    },
    'Redox': {
        'oxidative phosphorylation': ['NDUFA4', 'NDUFB2', 'NDUFC2', 'NDUFA5',
                                      'NDUFV2', 'COX8A', 'COX7B', 'COX6A1',
                                      'MDH2', 'CYCS', 'ATP5F1A', 'ATP5F1B'],
        'antioxidant defence': ['NQO1', 'SLC7A11', 'GPX4', 'PRDX1', 'PRDX5',
                                'FTL', 'FTH1', 'GCLC', 'GSS', 'TXN'],
    },
}


def builtin_library() -> RoleLibrary:
    """The regulated-cell-death and redox library used in the worked example.

    Role assignments follow the Nomenclature Committee on Cell Death 2018
    definitions. It is offered as a starting point and a format example, not as
    a canonical annotation - override it for your own pathways.
    """
    return RoleLibrary([Programme(mod, role, tuple(genes))
                        for mod, roles in _RCD.items()
                        for role, genes in roles.items()])
