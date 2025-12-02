"""
KNOT UTILITIES (Modern Version)
===============================
( knut_modern.py )
original by Konstantinos Meichanetzidis (phykme@leeds.ac.uk)
modernized: 2024

Auxiliary routines for reading and converting between
representations of knots, evaluation of basic knot
properties, encoding the Potts partition function in
CSP form, and evaluating the Jones polynomial.

DEPENDENCIES: numpy, tensorcsp_modern.py, cmath, time
"""

from __future__ import annotations
from typing import Callable
import cmath
from time import perf_counter

import numpy as np

from tensorcsp_modern import (
    cnf_nvar, cnf_tngraph, clause_tensor, attr_contract,
    contract_greedy, contract_dendrogram, recursive_bipartition
)


def writhe(x: np.ndarray, return_signs: bool = False) -> int | tuple[int, np.ndarray]:
    """
    Return writhe of planar code x for len(x) > 1.

    The writhe is the sum of crossing signs in the knot diagram.

    Args:
        x: Planar diagram code (array of crossings)
        return_signs: If True, also return individual crossing signs

    Returns:
        Writhe value, or tuple of (writhe, signs array) if return_signs=True
    """
    x = np.asarray(x)
    # Since planar code contains only "under" crossings,
    # this yields handedness sign for each one:
    sg = np.array([c[1] - c[3] for c in x])
    # Make sure to treat the last arc properly:
    sg[np.abs(sg) > 1] = -np.sign(sg[np.abs(sg) > 1])

    if return_signs:
        return int(np.sum(sg)), sg
    return int(np.sum(sg))


def taitnumber(c: np.ndarray) -> int:
    """
    Return Tait number from signed edge list of Tait graph.

    The Tait number is the sum of signs of all edges.
    """
    c = np.asarray(c)
    return int(np.sum(np.sign(c[:, 0])))


def pd2tait(X: np.ndarray | list) -> np.ndarray:
    """
    Extract Tait graph from Planar Diagram presentation.

    Converts a planar diagram (list of 4-tuples representing crossings)
    to a Tait graph represented as a signed edge list.

    Args:
        X: Planar diagram code (each row is a crossing with 4 arc labels)

    Returns:
        Signed edge list where each row is [region1, region2] with signs
    """
    x = np.asarray(X)
    na = x.max()  # Number of arcs

    # Gather region 1
    p0 = np.argwhere(x == 1)[0]  # Position in x where arc 1 appears first
    p = p0.copy()

    regions = [[]]

    # Left turn is the number to the left (up to cyclic permutation in the 4-tuple)
    pp = np.argwhere(x == x[p[0], (p[1] - 1) % 4])
    if pp[0][0] != p[0]:
        p = pp[0]
        regions[0].append(x[p[0], p[1]])
    else:
        p = pp[1]
        regions[0].append(x[p[0], p[1]])

    while x[p[0], p[1]] != 1:
        pp = np.argwhere(x == x[p[0], (p[1] - 1) % 4])
        if pp[0][0] != p[0]:
            p = pp[0]
            regions[0].append(x[p[0], p[1]])
        else:
            p = pp[1]
            regions[0].append(x[p[0], p[1]])

    # Gather rest of black regions. On odd (even) arc turn left (right).
    # Skip arcs which belong to regions already.
    for ii in range(2, na + 1):
        p1 = np.argwhere(x == ii)  # Position in x where arc ii appears
        if p1[0][0] != p0[0]:
            p0 = p1[0]
        else:
            p0 = p1[1]
        p = p0.copy()

        # Only continue if the arc is unaccounted for:
        if sum(ii in r for r in regions) == 0:
            # Left turns:
            if ii % 2 == 1:
                regions.append([])
                pp = np.argwhere(x == x[p[0], (p[1] - 1) % 4])
                if pp[0][0] != p[0]:
                    p = pp[0]
                    regions[-1].append(x[p[0], p[1]])
                else:
                    p = pp[1]
                    regions[-1].append(x[p[0], p[1]])

                while x[p[0], p[1]] != ii:
                    pp = np.argwhere(x == x[p[0], (p[1] - 1) % 4])
                    if pp[0][0] != p[0]:
                        p = pp[0]
                        regions[-1].append(x[p[0], p[1]])
                    else:
                        p = pp[1]
                        regions[-1].append(x[p[0], p[1]])

            # Right turns:
            if ii % 2 == 0:
                regions.append([])
                pp = np.argwhere(x == x[p[0], (p[1] + 1) % 4])
                if pp[0][0] != p[0]:
                    p = pp[0]
                    regions[-1].append(x[p[0], p[1]])
                else:
                    p = pp[1]
                    regions[-1].append(x[p[0], p[1]])

                while x[p[0], p[1]] != ii:
                    pp = np.argwhere(x == x[p[0], (p[1] + 1) % 4])
                    if pp[0][0] != p[0]:
                        p = pp[0]
                        regions[-1].append(x[p[0], p[1]])
                    else:
                        p = pp[1]
                        regions[-1].append(x[p[0], p[1]])

    # Now that we have all the regions in terms of the arcs that
    # enclose the region we go through all crossings (4 tuples)
    # and obtain the signed edge list.
    el = [[] for _ in range(len(x))]

    for iic, ii in enumerate(x):
        if not (na in ii and 1 in ii):
            eps = ((-1) ** min(ii)) * ((-1) ** (np.argmin(ii) + 1))
        else:
            eps = ((-1) ** max(ii)) * ((-1) ** (np.argmax(ii) + 1))

        pair1 = [ii[0], ii[1]]
        pair2 = [ii[2], ii[3]]

        for rc, r in enumerate(regions):
            if pair1[0] in r and pair1[1] in r:
                el[iic].append((rc + 1) * eps)
            if pair2[0] in r and pair2[1] in r:
                el[iic].append((rc + 1) * eps)

        pair1 = [ii[0], ii[3]]
        pair2 = [ii[1], ii[2]]

        for rc, r in enumerate(regions):
            if pair1[0] in r and pair1[1] in r:
                el[iic].append((rc + 1) * eps)
            if pair2[0] in r and pair2[1] in r:
                el[iic].append((rc + 1) * eps)

    return np.array(el)


def tpotts(q: float) -> complex:
    """
    Relation between Jones variable t (complex for 0 < q < 4)
    and number of spin states q in the Potts model.

    Args:
        q: Number of Potts states

    Returns:
        Jones polynomial variable t
    """
    t = 0.5 * (q + np.sqrt(q) * cmath.sqrt(q - 4) - 2)
    return t


def boltz_tensor(ek: complex, q: int) -> np.ndarray:
    """
    Boltzmann factor matrix for interaction bond of q-state
    Potts model for the Jones polynomial.

    Diagonal terms are set to ek (in general complex).
    Off-diagonal terms are set to 1.

    Some diagonal entries are later turned to ek**(-1)
    according to the Tait sign of the corresponding edge in
    the Tait graph. This is done with boltz_entry() when the
    tensor network is built.

    Args:
        ek: Boltzmann weight for same-spin pairs
        q: Number of Potts states

    Returns:
        q x q Boltzmann tensor
    """
    b = np.ones((q, q), dtype=complex)
    b[np.diag_indices(q)] = ek * np.ones(q)
    return b


def boltz_entry(b: np.ndarray, q: int, i: int, m: int) -> complex:
    """
    Enforce Tait sign convention in Boltzmann tensor.

    Args:
        b: Boltzmann tensor
        q: Number of states
        i: Linear index
        m: Tait sign mask

    Returns:
        Modified Boltzmann weight
    """
    idx = np.unravel_index(i, b.shape)
    return b[idx] ** ((-1) ** (m + 1))


def boltz_tngraph(c: np.ndarray, ek: complex, q: int, dtype: type = complex):
    """
    Constructs graph object endowed with variable and interaction
    tensors corresponding to q-state Potts model on graph dictated
    by signed edge list c.

    A positive sign in an entry of c means a positive Tait sign,
    and in turn an ek bond. Negative sign is an ek**(-1) bond.

    Args:
        c: Signed edge list (Tait graph)
        ek: Potts interaction strength
        q: Number of Potts states
        dtype: Data type for tensors

    Returns:
        NetworkX graph with Potts tensor attributes
    """
    b = boltz_tensor(ek, q)
    gt = lambda i, m: boltz_entry(b, q, i, m)
    g = cnf_tngraph(c, q, gate=gt, dtype=dtype)
    return g


def DeltaH_greedy(c: np.ndarray) -> int:
    """
    Returns maximal degree encountered during greedy
    contraction of the Tait graph encoded in edgelist c.

    Args:
        c: Signed edge list

    Returns:
        Maximum degree during contraction
    """
    nc = len(c)  # number of crossings
    q = 1

    if nc > 0:
        nv = cnf_nvar(c)
        ekpotts = -tpotts(q)  # Potts interaction for Jones
        g = boltz_tngraph(c, ekpotts, q)
        b, gn = contract_greedy(g, combine_attrs=None)
        maxdeg = max(b)
    else:
        maxdeg = 1

    return int(maxdeg)


def DeltaH_METIS(c: np.ndarray) -> int:
    """
    Returns maximal degree encountered during METIS
    contraction of the Tait graph encoded in edgelist c.

    Args:
        c: Signed edge list

    Returns:
        Maximum degree during contraction
    """
    nc = len(c)  # number of crossings
    q = 1

    if nc > 0:
        nv = cnf_nvar(c)
        ekpotts = -tpotts(q)  # Potts interaction for Jones
        g = boltz_tngraph(c, ekpotts, q)
        m = recursive_bipartition(g)
        md, sg = contract_dendrogram(g, m, combine_attrs=None)
        maxdeg = max(md)
    else:
        maxdeg = 1

    return int(maxdeg)


def Jones_greedy(
    c: np.ndarray,
    tau: int,
    w: int,
    q: float
) -> tuple[complex, float]:
    """
    Returns Jones polynomial evaluated at t(q) via greedy
    contraction of the tensor network of the knot encoded
    in edgelist c.

    Args:
        c: Signed edge list (Tait graph)
        tau: Tait number
        w: Writhe of the knot
        q: Potts parameter (determines Jones variable t)

    Returns:
        Tuple of (Jones polynomial value, runtime in seconds)
    """
    nc = len(c)  # number of crossings

    if nc > 0:
        nv = cnf_nvar(c)
        ekpotts = -tpotts(q)
        g = boltz_tngraph(c, ekpotts, q)

        t1 = perf_counter()
        b, gn = contract_greedy(g, combine_attrs={'attr': attr_contract})
        t2 = perf_counter()
        runtime = t2 - t1

        # Get final tensor value
        final_node = list(gn.nodes())[0]
        Z = gn.nodes[final_node]["attr"][1]
    else:
        nv = 1
        Z = q
        runtime = 0.0

    # Multiply Z with appropriate prefactors to get Jones polynomial
    t = tpotts(q)
    jpoly = (Z *
             (-t ** 0.5 - t ** -0.5) ** (-nv - 1) *
             (-t ** (3.0 / 4)) ** w *
             t ** (0.25 * tau))

    return jpoly, runtime


def Jones_METIS(
    c: np.ndarray,
    tau: int,
    w: int,
    q: float
) -> tuple[complex, float]:
    """
    Returns Jones polynomial evaluated at t(q) via METIS
    contraction of the tensor network of the knot encoded
    in edgelist c.

    Args:
        c: Signed edge list (Tait graph)
        tau: Tait number
        w: Writhe of the knot
        q: Potts parameter (determines Jones variable t)

    Returns:
        Tuple of (Jones polynomial value, runtime in seconds)
    """
    nc = len(c)  # number of crossings

    if nc > 0:
        nv = cnf_nvar(c)
        ekpotts = -tpotts(q)
        g = boltz_tngraph(c, ekpotts, q)

        t1 = perf_counter()
        m = recursive_bipartition(g)  # Uses METIS
        md, sg = contract_dendrogram(g, m, combine_attrs={'attr': attr_contract})
        t2 = perf_counter()
        runtime = t2 - t1

        # Get final tensor value
        final_node = list(sg.nodes())[0]
        Z = sg.nodes[final_node]["attr"][1]
    else:
        nv = 1
        Z = q
        runtime = 0.0

    # Multiply Z with appropriate prefactors to get Jones polynomial
    t = tpotts(q)
    jpoly = (Z *
             (-t ** 0.5 - t ** -0.5) ** (-nv - 1) *
             (-t ** (3.0 / 4)) ** w *
             t ** (0.25 * tau))

    return jpoly, runtime
