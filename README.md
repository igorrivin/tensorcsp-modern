# tensorCSP

A Python library for solving **Constraint Satisfaction Problems (CSPs)** using **tensor network contractions**. This includes applications to:

- **#SAT** (counting satisfying assignments)
- **Vertex cover counting**
- **Jones polynomial** computation for knots

The library uses graph partitioning algorithms to find efficient tensor contraction orderings, enabling the solution of problems that would be intractable with naive approaches.

## Theory

The methods are described in:

> S. Kourtis, C. Chamon, E. R. Mucciolo, and A. E. Ruckenstein,
> *Fast counting with tensor networks*,
> [arXiv:1805.00475](https://arxiv.org/abs/1805.00475) | [SciPost Phys. 7, 060 (2019)](https://scipost.org/10.21468/SciPostPhys.7.5.060)

### How It Works

1. **Encode** the CSP as a tensor network (variables → COPY tensors, constraints → gate tensors)
2. **Build** a graph where nodes are tensors and edges are contracted indices
3. **Find** an efficient contraction ordering using graph partitioning (METIS) or greedy algorithms
4. **Contract** the network to obtain the solution count

## Installation

```bash
# Clone the repository
git clone https://github.com/OWNER/tensorcsp-modern.git
cd tensorcsp-modern

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .

# With optional optimizations
pip install -e ".[all]"
```

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `numpy` | Yes | Tensor operations |
| `networkx` | Yes | Graph algorithms |
| `opt_einsum` | No | Optimized tensor contractions |
| `pymetis` | No | METIS graph partitioning |

## Quick Start

### Counting Vertex Covers (2-SAT)

```python
import networkx as nx
from tensorcsp_modern import (
    graph_to_2sat, cnf_tngraph, attr_contract, contract_greedy
)

# Create a random 3-regular graph with 20 vertices
G = nx.random_regular_graph(3, 20)

# Convert to 2-SAT CNF formula
cnf = graph_to_2sat(G)

# Build tensor network
tn_graph = cnf_tngraph(cnf, dtype=int)

# Contract using greedy algorithm
max_degrees, result = contract_greedy(
    tn_graph,
    combine_attrs={'attr': attr_contract}
)

# Get the count
final_node = list(result.nodes())[0]
count = result.nodes[final_node]["attr"][1]
print(f"Number of vertex covers: {count}")
```

### Computing Jones Polynomial

```python
import numpy as np
from knut_modern import pd2tait, taitnumber, writhe, Jones_greedy, tpotts

# Trefoil knot in planar diagram notation
trefoil = np.array([
    [1, 4, 2, 5],
    [3, 6, 4, 1],
    [5, 2, 6, 3]
])

# Convert to Tait graph
tait = pd2tait(trefoil)
tau = taitnumber(tait)
w = writhe(trefoil)

# Evaluate Jones polynomial at q=5
q = 5
V, runtime = Jones_greedy(tait, tau, w, q)

print(f"V(t(q={q})) = {V}")
print(f"Computed in {runtime:.4f} seconds")
```

> **Note:** The current implementation computes **point evaluations** of the Jones polynomial at specific values of t(q). To recover the full polynomial, evaluate at `deg+1` points and interpolate.

## File Structure

### Modern Version (NetworkX-based)

| File | Description |
|------|-------------|
| `tensorcsp_modern.py` | Core CSP encoding and tensor network construction |
| `grut_modern.py` | Graph utilities, partitioning, contraction algorithms |
| `knut_modern.py` | Knot utilities for Jones polynomial |
| `example_2sat3_modern.py` | Example: counting solutions to 3-regular 2-SAT |
| `example_jones_modern.py` | Example: Jones polynomial computation |

### Legacy Version (igraph-based)

The original implementation using igraph is preserved for reference:

| File | Description |
|------|-------------|
| `tensorcsp.py` | Original CSP encoding (requires igraph) |
| `grut.py` | Original graph utilities (requires igraph) |
| `knut.py` | Original knot utilities |

## Contraction Strategies

The library supports multiple contraction ordering strategies:

### 1. Greedy Contraction
```python
from tensorcsp_modern import contract_greedy

max_deg, result = contract_greedy(graph, combine_attrs={'attr': attr_contract})
```
Iteratively contracts the edge that minimizes the maximum degree. Fast and robust.

### 2. METIS Partitioning
```python
from tensorcsp_modern import recursive_bipartition, contract_dendrogram, metis_bipartition

merges = recursive_bipartition(graph, metis_bipartition)
max_deg, result = contract_dendrogram(graph, merges, combine_attrs={'attr': attr_contract})
```
Uses METIS graph partitioning for near-optimal orderings. Requires `pymetis`.

### 3. Fiedler Vector
```python
from grut_modern import fiedler_bipartition

merges = recursive_bipartition(graph, fiedler_bipartition)
```
Spectral partitioning using the Fiedler vector. No additional dependencies.

## API Reference

### Core Functions

#### `cnf_read(filename, sort_clauses=True)`
Read CNF formula from DIMACS format file.

#### `cnf_tngraph(cnf, q=2, gate=oror, dtype=int)`
Convert CNF formula to tensor network graph.
- `cnf`: List of clauses (each clause is a list of signed variable indices)
- `q`: Domain size (2 for boolean)
- `gate`: Gate function (`oror` for OR, `xorxor` for XOR)
- Returns: NetworkX graph with tensor attributes

#### `contract_greedy(graph, combine_attrs=None)`
Contract graph using greedy algorithm.
- Returns: `(max_degrees, contracted_graph)`

#### `attr_contract(attrs)`
Tensor contraction function for vertex attributes.

### Knot Functions

#### `pd2tait(planar_diagram)`
Convert planar diagram to Tait graph (signed edge list).

#### `Jones_greedy(tait_graph, tau, writhe, q)`
Compute Jones polynomial at t(q).
- Returns: `(value, runtime)`

## Performance Tips

1. **Use METIS for large instances**: The greedy algorithm is O(n²), while METIS provides better scaling.

2. **Use floating-point for large counts**: Integer overflow occurs around n>100 variables.
   ```python
   tn_graph = cnf_tngraph(cnf, dtype=float)
   ```

3. **Install opt_einsum**: Provides optimized tensor contraction paths.
   ```bash
   pip install opt_einsum
   ```

## Examples

Run the included examples:

```bash
# Count vertex covers on random 3-regular graphs
python example_2sat3_modern.py

# Compute Jones polynomials for various knots
python example_jones_modern.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kourtis2019fast,
  title={Fast counting with tensor networks},
  author={Kourtis, Stefanos and Chamon, Claudio and Mucciolo, Eduardo R and Ruckenstein, Andrei E},
  journal={SciPost Physics},
  volume={7},
  number={5},
  pages={060},
  year={2019},
  doi={10.21468/SciPostPhys.7.5.060}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Authors

- **Stefanos Kourtis** - Boston University
- **Konstantinos Meichanetzidis** - University of Leeds
- Modernization by Claude (Anthropic)
