"""
FUNCTIONS FOR EXPRESSING CSP FORMULAS AS TENSOR NETWORKS (Modern Version)
=========================================================================
( tensorcsp_modern.py )
original: 20170726 by Stefanos Kourtis (kourtis@bu.edu)
modernized: 2024

Routines for encoding constraint satisfaction problems as
tensor networks and basic tools for contraction. Contains
utilities for CNF formulas, as well as basic reading and
writing of CNF instances in DIMACS format.

DEPENDENCIES: numpy, networkx, opt_einsum (optional), grut_modern.py
"""

from __future__ import annotations
from typing import Callable, Any
from pathlib import Path

import numpy as np
import networkx as nx

# Try to import opt_einsum for optimized tensor contractions
try:
    import opt_einsum as oe
    _OPT_EINSUM_AVAILABLE = True
except ImportError:
    _OPT_EINSUM_AVAILABLE = False

# Import modern graph utilities
from grut_modern import *


# Boolean gate functions; i holds input bits, m is negation mask
def oror(i: int, m: int = 0) -> int:
    """OR gate: returns 1 if any bit is set after XOR with mask."""
    val = i ^ m
    return int(bin(val).count('1') > 0)


def xorxor(i: int, m: int = 0) -> int:
    """XOR gate: returns 1 if odd number of bits set after XOR with mask."""
    val = i ^ m
    return int(bin(val).count('1') % 2 > 0)


def var_tensor(l: int = 3, q: int = 2, dtype: type = int) -> np.ndarray:
    """
    Return variable tensor of rank l with entries over
    domain of dimension q. Also called COPY tensor.

    The COPY tensor has 1s on the "diagonal" (where all indices are equal)
    and 0s elsewhere.
    """
    t = np.zeros([q] * l, dtype=dtype)
    t[np.diag_indices(q, l)] = 1
    return t


def clause_tensor(
    l: int,
    q: int = 2,
    g: Callable[[int, int], int] = oror,
    m: int = 0,
    dtype: type = int
) -> np.ndarray:
    """
    Return tensorization of the truth table of gate g.

    Generally, g is a l-ary relation in the constraint
    language of a (weighted) CSP and this function
    returns a tensor representation of the relation g.

    Args:
        l: Number of inputs (tensor rank)
        q: Domain dimension (default 2 for boolean)
        g: Gate function taking (input_bits, negation_mask)
        m: Negation mask
        dtype: Data type for the tensor
    """
    d = [q] * l
    t = np.zeros(d, dtype=dtype)
    for i in range(q ** l):
        idx = np.unravel_index(i, d)
        t[idx] = g(i, m)
    return t


def cnf_read(
    fs: str | list[str],
    sort_clauses: bool = True,
    read_xors: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Read CNF formula(s) from file(s) in DIMACS format.

    Args:
        fs: Filename or list of filenames
        sort_clauses: Sort variables in each clause by absolute value
        read_xors: Read extended DIMACS format (cryptominisat2 XOR clauses)

    Returns:
        If read_xors is False: array of clauses
        If read_xors is True: tuple of (clauses, xor_flags)
    """
    if isinstance(fs, (str, Path)):
        fs = [fs]

    xs = []
    cs = []

    for f in fs:
        c = []
        x = []
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line or line[0] in ('c', 'p'):
                    continue
                if line[0] == 'x':
                    line = line[1:]
                    x.append(1)
                else:
                    x.append(0)
                vs = np.array([int(i) for i in line.split()[:-1]])
                if sort_clauses:
                    vs = vs[np.argsort(np.abs(vs))]
                c.append(vs)

        # Convert to array if all clauses have same length
        if c and all(len(r) == len(c[0]) for r in c):
            c = np.array(c)
        cs.append(c)
        xs.append(x)

    if len(cs) == 1:
        cs = cs[0]
    if read_xors:
        return cs, np.array(xs[0] if len(xs) == 1 else xs)
    return cs


def cnf_write(
    c: np.ndarray | list,
    filename: str | Path,
    xs: list | None = None
) -> None:
    """
    Write CNF formula c to file in DIMACS format.

    Args:
        c: Array/list of clauses
        filename: Output file path
        xs: Optional XOR flags for extended DIMACS format
    """
    nv = cnf_nvar(c)
    nc = len(c)
    xs = [' '] * nc if xs is None else [' ' if x == 0 else 'x' for x in xs]

    with open(filename, 'w') as f:
        f.write(f'p cnf {nv} {nc}\n')
        for i in range(nc):
            clause_str = ' '.join(str(j) for j in c[i])
            f.write(f'{xs[i]}{clause_str} 0\n')


def cnf_negmask(c: np.ndarray | list) -> list[int]:
    """
    Compute negation mask of CNF formula c.

    Returns a list of integers where each integer's binary representation
    indicates which variables in the clause are negated.
    """
    masks = []
    for clause in c:
        # Create binary string: 1 if variable is negated, 0 otherwise
        bits = ''.join('1' if v < 0 else '0' for v in clause)
        masks.append(int(bits, 2))
    return masks


def cnf_nvar(c: np.ndarray | list) -> int:
    """Return number of variables in CNF formula c."""
    max_var = 0
    for clause in c:
        clause_max = max(abs(v) for v in clause)
        max_var = max(max_var, clause_max)
    return max_var


def cnf_graph(c: np.ndarray | list) -> nx.Graph:
    """
    Returns bipartite graph corresponding to CNF formula c.

    Components represent variables (nodes 0..nv-1) and clauses
    (nodes nv..nv+nc-1). Edges connect variables to the clauses
    they participate in.
    """
    nc = len(c)
    nv = cnf_nvar(c)

    g = nx.Graph()

    # Add variable nodes (0 to nv-1) and clause nodes (nv to nv+nc-1)
    g.add_nodes_from(range(nv + nc))

    # Mark node types for bipartite structure
    for i in range(nv):
        g.nodes[i]['bipartite'] = 0  # Variable nodes
    for i in range(nv, nv + nc):
        g.nodes[i]['bipartite'] = 1  # Clause nodes

    # Add edges between variables and their clauses
    for clause_idx, clause in enumerate(c):
        clause_node = nv + clause_idx
        for var in clause:
            var_node = abs(var) - 1  # Variables are 1-indexed in DIMACS
            g.add_edge(var_node, clause_node)

    return g


def cnf_tn(
    c: np.ndarray | list,
    q: int = 2,
    gate: Callable = oror,
    dtype: type = int
) -> list[np.ndarray]:
    """
    Returns tensor network for CNF formula c.

    The tensor network is defined on the graph returned by cnf_graph().

    Args:
        c: CNF formula (list of clauses)
        q: Domain dimension
        gate: Gate function for clause tensors
        dtype: Data type for tensors

    Returns:
        List of tensors (variable tensors first, then clause tensors)
    """
    nv = cnf_nvar(c)
    nm = cnf_negmask(c)
    g = cnf_graph(c)

    # Get node degrees for tensor ranks
    degrees = dict(g.degree())

    # Build tensor network; variable tensors first
    tn = []
    for i in range(nv):
        rank = degrees[i]
        tn.append(var_tensor(rank, q, dtype=dtype))

    # Then clause tensors
    for clause_idx, clause in enumerate(c):
        rank = len(clause)
        ct = clause_tensor(rank, q, gate, m=nm[clause_idx], dtype=dtype)
        tn.append(ct)

    return tn


def cnf_tngraph(
    c: np.ndarray | list,
    q: int = 2,
    gate: Callable = oror,
    dtype: type = int
) -> nx.Graph:
    """
    Graph object including tensor representation of CNF formula c.

    Each vertex stores (a) a list of unique edge indices incident to it,
    and (b) the truth tensor corresponding to variable/clause
    as a list attribute named 'attr'.

    Args:
        c: CNF formula
        q: Domain dimension
        gate: Gate function for clause tensors
        dtype: Data type for tensors

    Returns:
        NetworkX graph with tensor attributes
    """
    g = cnf_graph(c)
    tn = cnf_tn(c, q, gate=gate, dtype=dtype)

    # Use initial edge indices for unique bond indexing
    # throughout contraction sequence
    il = get_incidence_list(g)

    for i in g.nodes():
        g.nodes[i]["attr"] = [il[i], tn[i]]

    return g


def attr_contract(d: list) -> list:
    """
    Contraction function for vertex attributes, to be used with
    graph edge contraction.

    The input is a list containing the attributes of vertices being merged,
    each of which is a list of [incidence_list, tensor].

    This function finds common indices in the incidence lists,
    contracts corresponding tensor dimensions, then concatenates
    incidence lists into a new one.

    This function is designed to be passed as an argument to
    contract_edge() via combine_attrs={'attr': attr_contract}.
    """
    if len(d) == 0:
        return [[], np.array([])]
    if len(d) == 1:
        return d[0]

    i1, i2 = d[0][0].copy(), d[1][0].copy()
    t1, t2 = d[0][1], d[1][1]

    # Find common edge indices (bonds to contract)
    ce = np.intersect1d(i1, i2)

    # Get tensor dimensions corresponding to common edges
    d1 = [i1.index(i) for i in ce]
    d2 = [i2.index(i) for i in ce]

    # Remove contracted indices from incidence lists
    for i in sorted(d1, reverse=True):
        i1.pop(i)
    for i in sorted(d2, reverse=True):
        i2.pop(i)

    # Contract tensors
    if _OPT_EINSUM_AVAILABLE and len(d1) > 0:
        # Use opt_einsum for potentially better contraction
        t = oe.contract(t1, range(t1.ndim), t2,
                        [i + t1.ndim if i not in d2 else d1[d2.index(i)]
                         for i in range(t2.ndim)],
                        [i for i in range(t1.ndim) if i not in d1] +
                        [i + t1.ndim for i in range(t2.ndim) if i not in d2])
    else:
        t = np.tensordot(t1, t2, [d1, d2])

    return [i1 + i2, t]  # CAUTION! inclist *NOT* sorted


def contract_tensor_network(
    g: nx.Graph,
    contraction_order: list[tuple[int, int]] | None = None,
    method: str = "greedy"
) -> np.ndarray:
    """
    Contract the entire tensor network to a scalar.

    This is a convenience function that handles the full contraction.

    Args:
        g: Graph with tensor attributes
        contraction_order: Optional list of edges to contract in order
        method: Contraction method ("greedy" or "metis")

    Returns:
        Final contracted tensor (usually a scalar)
    """
    combine = {'attr': attr_contract}

    if method == "greedy":
        _, result = contract_greedy(g, combine_attrs=combine)
    elif method == "metis":
        merges = recursive_bipartition(g)
        _, result = contract_dendrogram(g, merges, combine_attrs=combine)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract final tensor
    if result.number_of_nodes() > 0:
        final_node = list(result.nodes())[0]
        return result.nodes[final_node]["attr"][1]
    return np.array(0)


# Convenience function for generating random k-regular graphs
def random_regular_graph(n: int, k: int) -> nx.Graph:
    """Generate a random k-regular graph with n vertices."""
    return nx.random_regular_graph(k, n)


def graph_to_2sat(g: nx.Graph) -> np.ndarray:
    """
    Convert a graph to a 2-SAT CNF formula (vertex cover encoding).

    Each edge (u, v) becomes a clause (u+1 OR v+1), meaning at least
    one endpoint must be in the vertex cover.

    Args:
        g: NetworkX graph

    Returns:
        CNF formula as numpy array
    """
    edges = list(g.edges())
    clauses = np.array([[u + 1, v + 1] for u, v in edges])
    return clauses
