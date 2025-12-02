"""
EXAMPLE: SOLVING 3-REGULAR #2SAT INSTANCES (Modern Version)
============================================================
( example_2sat3_modern.py )
original: 20180425 by Stefanos Kourtis (kourtis@bu.edu)
modernized: 2024

Example script demonstrating the solution of instances of
positive 3-regular #2SAT (i.e. #CUBIC-VERTEX-COVER) using
graph partitioning and tensor network contraction. Random
instances are generated as random 3-regular graphs, where
vertices correspond to variables and edges to clauses.

DEPENDENCIES: tensorcsp_modern.py
"""

from time import perf_counter
import numpy as np
import networkx as nx

from tensorcsp_modern import (
    cnf_write, cnf_tngraph, attr_contract,
    contract_greedy, contract_dendrogram,
    recursive_bipartition, metis_bipartition,
    graph_to_2sat
)


def main():
    # Generate random 3-regular graph with nv vertices. Equivalently
    # these can be thought of as 2SAT instances with nv variables and
    # each edge represents a clause. Write the corresponding CNF to
    # a DIMACS file for purposes of comparison with model counters.
    #
    # NB: Counts for nv>100 start to overflow default numpy int. Pass
    #     dtype=float to cnf_tngraph for floating-point precision.
    nv = 80

    # Generate random 3-regular graph using NetworkX
    cg = nx.random_regular_graph(3, nv)

    # Convert to 2SAT CNF formula (edge list + 1 for 1-based indexing)
    cf = graph_to_2sat(cg)

    # Write to file in DIMACS format
    cnf_write(cf, "tmp.cnf")

    # Build tensor graph
    tg = cnf_tngraph(cf, dtype=int)

    # First solve using a greedy contraction algorithm:
    print("=" * 60)
    print("Greedy Contraction")
    print("=" * 60)

    start = perf_counter()
    md, sg = contract_greedy(tg, combine_attrs={'attr': attr_contract})
    end = perf_counter()

    # Get solution from the remaining node
    final_node = list(sg.nodes())[0]
    sol = sg.nodes[final_node]["attr"][1]

    print(f"Solved in {end - start:.4f} seconds")
    print(f"  #Solutions: {sol}")
    print(f"  Max degree: {md.max()}")
    print()

    # Then solve using METIS graph partitioning:
    print("=" * 60)
    print("METIS Partitioning")
    print("=" * 60)

    # Rebuild tensor graph (previous one was modified)
    tg = cnf_tngraph(cf, dtype=int)

    try:
        start = perf_counter()
        m = recursive_bipartition(tg, metis_bipartition)
        md, sg = contract_dendrogram(tg, m, combine_attrs={'attr': attr_contract})
        end = perf_counter()

        final_node = list(sg.nodes())[0]
        sol = sg.nodes[final_node]["attr"][1]

        print(f"Solved in {end - start:.4f} seconds")
        print(f"  #Solutions: {sol}")
        print(f"  Max degree: {md.max()}")
    except ImportError as e:
        print(f"METIS not available: {e}")
        print("Install pymetis: pip install pymetis")
    print()

    # Note: The dendrogram-based contraction (Fiedler, Girvan-Newman)
    # can hit numpy's 32-dimension limit for large graphs due to
    # suboptimal contraction ordering. For large instances, greedy
    # or METIS contraction is recommended.


if __name__ == "__main__":
    main()
