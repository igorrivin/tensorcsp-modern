"""
GRAPH UTILITIES (Modern Version)
================================
( grut_modern.py )
original: 20170726 by Stefanos Kourtis (kourtis@bu.edu)
modernized: 2024

Auxiliary routines for evaluation of graph properties and
graph operations based on NetworkX (replacing igraph).
Partitions are obtained using pymetis/metis, which needs
to be installed for the corresponding functions to work.

DEPENDENCIES: numpy, networkx, pymetis (optional)
"""

from __future__ import annotations
from typing import Callable, Any, Literal
from copy import deepcopy

import numpy as np
from numpy.linalg import eigh
import networkx as nx

# Optional METIS import - try pymetis first, then metis
_metis_available = False
try:
    import pymetis
    _metis_available = True
    _metis_backend = "pymetis"
except ImportError:
    try:
        from metis import part_graph
        _metis_available = True
        _metis_backend = "metis"
    except ImportError:
        pass


def adjmat(g: nx.Graph) -> np.ndarray:
    """Adjacency matrix of graph object g as ndarray."""
    return nx.to_numpy_array(g, dtype=int)


def adjlist2adjmat(a: list[list[int]]) -> np.ndarray:
    """Convert adjacency list to adjacency matrix."""
    flat = sum(a, [])
    n = max(flat) + 1 if flat else 0
    m = np.zeros((n, n), dtype=int)
    for i, neighbors in enumerate(a):
        for j in neighbors:
            m[i, j] += 1
    return m


def get_adjacency_list(g: nx.Graph) -> list[list[int]]:
    """Get adjacency list from NetworkX graph."""
    nodes = sorted(g.nodes())
    return [sorted(g.neighbors(n)) for n in nodes]


def get_incidence_list(g: nx.Graph) -> list[list[int]]:
    """
    Get incidence list (list of edge indices incident to each vertex).
    Returns a list where element i contains the indices of edges incident to node i.
    """
    nodes = sorted(g.nodes())
    edges = list(g.edges())
    edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}

    inclist = []
    for node in nodes:
        incident_edges = []
        for neighbor in g.neighbors(node):
            edge_key = tuple(sorted((node, neighbor)))
            if edge_key in edge_to_idx:
                incident_edges.append(edge_to_idx[edge_key])
        inclist.append(incident_edges)
    return inclist


def get_cluster_vids(membership: np.ndarray | list) -> list[list[int]]:
    """
    Return vertex indices grouped according to
    graph clustering as encoded in membership vector.
    """
    membership = np.asarray(membership)
    nc = membership.max() + 1
    return [list(np.where(membership == i)[0]) for i in range(nc)]


def get_cluster_eids(membership: np.ndarray | list, g: nx.Graph) -> list[list[int]]:
    """
    Return edge indices grouped according to graph
    clustering as encoded in membership vector.
    """
    vi = get_cluster_vids(membership)
    edges = list(g.edges())
    ei = []
    for cluster_nodes in vi:
        cluster_set = set(cluster_nodes)
        cluster_edges = [
            i for i, (u, v) in enumerate(edges)
            if u in cluster_set and v in cluster_set
        ]
        ei.append(cluster_edges)
    return ei


def get_bipartition_eids(membership: np.ndarray | list, g: nx.Graph) -> list[int]:
    """
    Given a bipartition encoded in a membership vector,
    return the indices of the edges that form the edge
    separator between components.
    """
    m = np.asarray(membership, dtype=int)
    s1 = set(np.where(m == 0)[0])
    s2 = set(np.where(m > 0)[0])
    edges = list(g.edges())
    return [
        i for i, (u, v) in enumerate(edges)
        if (u in s1 and v in s2) or (u in s2 and v in s1)
    ]


def bipartition_width(membership: np.ndarray | list, g: nx.Graph) -> int:
    """Return bipartition width from membership vector."""
    return len(get_bipartition_eids(membership, g))


def _metis_partition(adjacency_list: list[list[int]], nparts: int = 2,
                     ncuts: int = 2) -> np.ndarray:
    """Internal function to call METIS partitioning."""
    if not _metis_available:
        raise ImportError("Neither pymetis nor metis is available")

    if _metis_backend == "pymetis":
        # pymetis uses different API
        _, membership = pymetis.part_graph(nparts, adjacency=adjacency_list)
        return np.array(membership)
    else:
        # metis library
        _, membership = part_graph(adjacency_list, ncuts=ncuts,
                                   recursive=False, contig=True, minconn=True)
        return np.array(membership)


def metis_bipartition(g: nx.Graph, n: int = 2) -> np.ndarray:
    """Perform METIS bipartition. Do n cuts and choose the best."""
    al = get_adjacency_list(g)
    return _metis_partition(al, nparts=2, ncuts=n)


def metis_kway(g: nx.Graph, k: int) -> np.ndarray:
    """Perform METIS k-way partition."""
    al = get_adjacency_list(g)
    return _metis_partition(al, nparts=k)


def fiedler_bipartition(g: nx.Graph) -> np.ndarray:
    """
    Perform bipartition based on the Fiedler vector.
    Note: Fiedler vector does not guarantee contiguous partitions.
    This function is here for testing purposes only.
    """
    laplacian = nx.laplacian_matrix(g).toarray()
    _, v = eigh(laplacian)
    return (v[:, 1] <= 0).astype(int)


def recursive_bipartition(
    g: nx.Graph,
    fbipart: Callable[[nx.Graph], np.ndarray] = None
) -> np.ndarray:
    """
    Build separator hierarchy using recursive bipartition.
    Returns a dendrogram merge sequence.
    """
    if fbipart is None:
        fbipart = metis_bipartition

    nv = g.number_of_nodes()
    cg = deepcopy(g)

    # Ensure nodes are labeled 0..nv-1
    if set(cg.nodes()) != set(range(nv)):
        mapping = {old: new for new, old in enumerate(sorted(cg.nodes()))}
        cg = nx.relabel_nodes(cg, mapping)

    # Store original node names
    for node in cg.nodes():
        cg.nodes[node]["name"] = node

    sg = [cg]
    tr = []
    im = 2 * nv

    for _ in range(nv):
        st = []
        for s in sg:
            if s.number_of_nodes() == 1:
                continue

            fb = fbipart(s)
            nodes = sorted(s.nodes())
            s1 = [nodes[i] for i in np.where(fb == 0)[0]]
            s2 = [nodes[i] for i in np.where(fb > 0)[0]]

            # METIS often refuses to partition very small graphs
            # so "peel off" least connected vertex instead
            while len(s1) * len(s2) == 0:
                degrees = dict(s.degree())
                min_degree_node = min(nodes, key=lambda n: degrees[n])
                fb = np.zeros(s.number_of_nodes(), dtype=int)
                fb[nodes.index(min_degree_node)] = 1
                s1 = [nodes[i] for i in np.where(fb == 0)[0]]
                s2 = [nodes[i] for i in np.where(fb > 0)[0]]

            g1 = s.subgraph(s1).copy()
            g2 = s.subgraph(s2).copy()
            st.append(g1)
            st.append(g2)

            if len(s1) == 1:
                i1 = g1.nodes[s1[0]]["name"]
            else:
                im -= 1
                i1 = im

            if len(s2) == 1:
                i2 = g2.nodes[s2[0]]["name"]
            else:
                im -= 1
                i2 = im

            tr.append(np.array([i1, i2]))

        sg = st
        if len(st) == 0:
            break

    tr = np.array(tr[::-1])
    if len(tr) > 0 and np.any(tr >= nv):
        th = tr[tr >= nv].min()
        df = th - nv
        tr[tr >= nv] = tr[tr >= nv] - df

    return tr


def graph_max_degree(g: nx.Graph) -> int:
    """Return maximum degree of graph."""
    if g.number_of_nodes() == 0:
        return 0
    return max(dict(g.degree()).values())


def find_cheapest_edge(
    g: nx.Graph,
    subset: list[int] | None = None
) -> tuple[int, int]:
    """
    Return edge whose contraction leads to the graph minor
    of g with the lowest maximum degree.
    Optionally restrict the search to a subset of edge indices.
    Returns the edge as (u, v) tuple.
    """
    edges = list(g.edges())
    if not edges:
        return None

    degrees = dict(g.degree())

    if subset is None:
        subset = range(len(edges))

    # Count multiple edges (for multigraphs)
    if isinstance(g, nx.MultiGraph):
        edge_counts = {(min(u, v), max(u, v)): g.number_of_edges(u, v)
                       for u, v in edges}
    else:
        edge_counts = {(min(u, v), max(u, v)): 1 for u, v in edges}

    best_edge_idx = None
    best_cost = float('inf')

    for idx in subset:
        u, v = edges[idx]
        edge_key = (min(u, v), max(u, v))
        mult = edge_counts.get(edge_key, 1)
        # Cost is product of (degree - multiplicity) for contracted vertices
        cost = (degrees[u] - mult) * (degrees[v] - mult)
        if cost < best_cost:
            best_cost = cost
            best_edge_idx = idx

    return edges[best_edge_idx] if best_edge_idx is not None else None


def contract_edge(
    g: nx.Graph,
    edge: tuple[int, int],
    combine_attrs: dict[str, Callable] | None = None,
    overwrite: Literal["higher", "lower", "none"] | int = "higher"
) -> None:
    """
    Perform edge contraction with user-specified function
    to combine vertex attributes. Modifies graph in place.

    Args:
        g: NetworkX graph to modify
        edge: Edge (u, v) to contract
        combine_attrs: Dict mapping attribute names to combining functions
        overwrite: Which vertex to keep ("higher", "lower", "none", or specific node)
    """
    u, v = edge

    # Remove all edges between u and v
    if isinstance(g, nx.MultiGraph):
        keys = list(g[u][v].keys())
        for key in keys:
            g.remove_edge(u, v, key)
    else:
        g.remove_edge(u, v)

    # Determine which node survives
    if overwrite == "higher":
        keep, remove = v, u
    elif overwrite == "lower":
        keep, remove = u, v
    elif overwrite == "none":
        # Create new node
        new_node = max(g.nodes()) + 1
        g.add_node(new_node)
        keep, remove = new_node, None
        # Need to rewire both u and v to new_node
        for neighbor in list(g.neighbors(u)):
            if neighbor != v:
                g.add_edge(new_node, neighbor)
        for neighbor in list(g.neighbors(v)):
            if neighbor != u and neighbor != new_node:
                g.add_edge(new_node, neighbor)
        # Combine attributes from both
        if combine_attrs:
            for attr, func in combine_attrs.items():
                attrs = []
                if attr in g.nodes[u]:
                    attrs.append(g.nodes[u][attr])
                if attr in g.nodes[v]:
                    attrs.append(g.nodes[v][attr])
                if attrs:
                    g.nodes[new_node][attr] = func(attrs)
        g.remove_node(u)
        g.remove_node(v)
        return
    elif isinstance(overwrite, int):
        if overwrite == u:
            keep, remove = u, v
        elif overwrite == v:
            keep, remove = v, u
        else:
            # Contract into specified node (must exist)
            keep, remove = overwrite, None
            for neighbor in list(g.neighbors(u)):
                if neighbor != v and neighbor != keep:
                    g.add_edge(keep, neighbor)
            for neighbor in list(g.neighbors(v)):
                if neighbor != u and neighbor != keep and not g.has_edge(keep, neighbor):
                    g.add_edge(keep, neighbor)
            if combine_attrs:
                for attr, func in combine_attrs.items():
                    attrs = []
                    if attr in g.nodes.get(keep, {}):
                        attrs.append(g.nodes[keep][attr])
                    if attr in g.nodes[u]:
                        attrs.append(g.nodes[u][attr])
                    if attr in g.nodes[v]:
                        attrs.append(g.nodes[v][attr])
                    if attrs:
                        g.nodes[keep][attr] = func(attrs)
            g.remove_node(u)
            g.remove_node(v)
            return
    else:
        keep, remove = v, u

    # Rewire edges from removed node to kept node
    for neighbor in list(g.neighbors(remove)):
        if neighbor != keep:
            g.add_edge(keep, neighbor)

    # Combine attributes
    if combine_attrs:
        for attr, func in combine_attrs.items():
            attrs = []
            if attr in g.nodes[keep]:
                attrs.append(g.nodes[keep][attr])
            if attr in g.nodes[remove]:
                attrs.append(g.nodes[remove][attr])
            if attrs:
                g.nodes[keep][attr] = func(attrs)

    g.remove_node(remove)


def contract_greedy(
    g: nx.Graph,
    n: int = 0,
    fsize: Callable[[nx.Graph], int] = graph_max_degree,
    combine_attrs: dict[str, Callable] | None = None
) -> tuple[np.ndarray, nx.Graph]:
    """
    Greedy contraction algorithm.

    Args:
        g: Graph to contract
        n: Maximum number of contractions (0 = unlimited)
        fsize: Function to compute graph size metric
        combine_attrs: Attribute combining functions

    Returns:
        Tuple of (size history array, contracted graph)
    """
    cg = deepcopy(g)
    bs = [fsize(cg)]

    if n == 0:
        n = 100000

    while cg.number_of_edges() > 0 and n > 0:
        edge = find_cheapest_edge(cg)
        if edge is None:
            break
        contract_edge(cg, edge, combine_attrs=combine_attrs)
        bs.append(fsize(cg))
        n -= 1

    bs = np.array(bs)

    # Remove isolated nodes (degree 0) except one
    isolated = [n for n in cg.nodes() if cg.degree(n) == 0]
    if len(isolated) == cg.number_of_nodes() and len(isolated) > 0:
        isolated = isolated[:-1]
    cg.remove_nodes_from(isolated)

    return bs, cg


def contract_dendrogram(
    g: nx.Graph,
    merges: np.ndarray,
    fsize: Callable[[nx.Graph], int] = graph_max_degree,
    combine_attrs: dict[str, Callable] | None = None,
    stop: int = -1
) -> tuple[np.ndarray, nx.Graph]:
    """
    Contract graph according to merge sequence.

    Args:
        g: Graph to contract
        merges: Array of [node1, node2] pairs to merge
        fsize: Function to compute graph size metric
        combine_attrs: Attribute combining functions
        stop: Stop after this many merges (-1 = all)

    Returns:
        Tuple of (size history array, contracted graph)
    """
    merges = np.asarray(merges, dtype=int)
    bs = [fsize(g)]
    cg = deepcopy(g)

    # Track node mapping as we contract
    node_map = {i: i for i in range(g.number_of_nodes())}
    next_node = g.number_of_nodes()

    for i, (m1, m2) in enumerate(merges):
        # Find current nodes corresponding to original merge targets
        curr1 = node_map.get(m1, m1)
        curr2 = node_map.get(m2, m2)

        # Skip if nodes don't exist or are the same
        if curr1 not in cg.nodes() or curr2 not in cg.nodes():
            continue
        if curr1 == curr2:
            continue

        # Check if edge exists, if not find path and contract along it
        if not cg.has_edge(curr1, curr2):
            # For dendrogram contraction, we may need to handle non-adjacent merges
            # by contracting along a path
            try:
                path = nx.shortest_path(cg, curr1, curr2)
                for j in range(len(path) - 1):
                    if cg.has_edge(path[j], path[j+1]):
                        contract_edge(cg, (path[j], path[j+1]),
                                      combine_attrs=combine_attrs,
                                      overwrite="none")
                        bs.append(fsize(cg))
            except nx.NetworkXNoPath:
                continue
        else:
            contract_edge(cg, (curr1, curr2), combine_attrs=combine_attrs,
                         overwrite="none")
            bs.append(fsize(cg))

        # Update node mapping
        new_node = max(cg.nodes()) if cg.number_of_nodes() > 0 else next_node
        node_map[m1] = new_node
        node_map[m2] = new_node
        next_node = new_node + 1

        if i == stop - 1:
            break

    bs = np.array(bs)

    # Remove isolated nodes except one
    isolated = [n for n in cg.nodes() if cg.degree(n) == 0]
    if len(isolated) == cg.number_of_nodes() and len(isolated) > 0:
        isolated = isolated[:-1]
    cg.remove_nodes_from(isolated)

    return bs, cg


def girvan_newman_dendrogram(g: nx.Graph) -> tuple[np.ndarray, list]:
    """
    Compute Girvan-Newman community structure and return merge sequence.
    This replaces igraph's community_edge_betweenness().

    Returns:
        Tuple of (merge sequence array, list of community partitions)
    """
    from networkx.algorithms.community import girvan_newman

    cg = deepcopy(g)
    nv = g.number_of_nodes()

    # Get all levels of the dendrogram
    communities_generator = girvan_newman(cg)

    # Build merge sequence by tracking community splits in reverse
    communities_list = []
    for communities in communities_generator:
        communities_list.append([set(c) for c in communities])

    # Reverse to get merges instead of splits
    merges = []
    node_to_cluster = {n: n for n in g.nodes()}
    next_cluster = nv

    # Process from finest to coarsest
    for i in range(len(communities_list) - 1, 0, -1):
        current = communities_list[i]
        parent = communities_list[i - 1]

        # Find which communities merged
        for pc in parent:
            children = [c for c in current if c.issubset(pc)]
            if len(children) == 2:
                # These two communities merged
                c1, c2 = children
                # Find representative nodes
                rep1 = min(c1)
                rep2 = min(c2)
                merges.append([rep1, rep2])

    return np.array(merges) if merges else np.array([]).reshape(0, 2), communities_list
