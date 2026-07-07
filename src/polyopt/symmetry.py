"""Box-safe symmetry-breaking cuts for tensor decomposition losses.

The loss is invariant under (per rank-1 term) double sign flips
(u_r, v_r) -> (-u_r, -v_r), (u_r, w_r) -> (-u_r, -w_r), and under
permutations of the R rank-1 terms. Some global minimizer therefore
satisfies the cuts below, so adding them preserves the global minimum
while shrinking the relaxation's feasible set. Scaling gauges
(alpha*u, v/alpha) are NOT box-safe and must not be used.

Cuts are expressed as (coeffs: dict[monomial -> float], const) meaning
sum coeffs * M[monomial] + const >= 0, and converted to sparse rows over
a MomentIndex by `cuts_to_rows`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from polyopt.matmul_poly import MatmulVariables
from polyopt.relaxation import MomentIndex
from polyopt.sparse_poly import Monomial

Cut = tuple[dict[Monomial, float], float]


def symmetry_cuts(var: MatmulVariables) -> list[Cut]:
    cuts: list[Cut] = []
    # sign gauge: flip (u_r, v_r) and (u_r, w_r) to make the first entry
    # of u_r and of v_r nonnegative
    for r in range(var.rank):
        cuts.append(({(var.u(r, 0),): 1.0}, 0.0))
        cuts.append(({(var.v(r, 0),): 1.0}, 0.0))
    # term order: sort rank-1 terms by first u entry (valid jointly with
    # the sign fix; permutations commute with per-term sign flips)
    for r in range(var.rank - 1):
        cuts.append(
            ({(var.u(r + 1, 0),): 1.0, (var.u(r, 0),): -1.0}, 0.0)
        )
    return cuts


def cuts_to_rows(cuts: list[Cut], moments: MomentIndex
                 ) -> tuple[sp.csr_matrix, np.ndarray]:
    rows, cols, vals, consts = [], [], [], []
    kept = 0
    for coeffs, const in cuts:
        if not all(m in moments for m in coeffs):
            continue  # cut references unlifted moments; skip (still valid)
        for m, c in coeffs.items():
            rows.append(kept), cols.append(moments[m]), vals.append(c)
        consts.append(const)
        kept += 1
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(kept, len(moments)))
    return mat, np.asarray(consts)
