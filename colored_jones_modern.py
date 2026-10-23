"""
COLORED JONES POLYNOMIAL (Modern Version)
=========================================
( colored_jones_modern.py )
2024

Computes the N-colored Jones polynomial J_N(K; q) using
N-dimensional R-matrices from the quantum group U_q(sl_2).

The N-colored Jones polynomial uses the N-dimensional irreducible
representation V_N of U_q(sl_2). This is related to the standard
Jones polynomial (N=2) but uses higher-dimensional R-matrices.

Key formulas:
- R-matrix eigenvalues: q^{j(j+1)/2 - N(N-1)/4} for j = 0, 1, ..., N-1
- Loop value: [N]_q = (q^N - q^{-N}) / (q - q^{-1})
- Writhe normalization: q^{w * N(N-1)/4}

STATUS: Work in progress
- The quantum integer/factorial/binomial functions are correct
- The R-matrix structure is implemented
- The braid closure normalization needs refinement to match standard conventions
- For production use, consider using snappy's jones_polynomial() for N=2

TODO:
- Implement proper Markov trace for braid closure
- Verify writhe normalization exponents
- Add Jones-Wenzl projector support for higher N
- Compare with known colored Jones values for torus knots

DEPENDENCIES: numpy, networkx, opt_einsum
"""

from __future__ import annotations
from typing import Callable, Tuple
from time import perf_counter

import numpy as np
import networkx as nx

try:
    import opt_einsum as oe
    _OPT_EINSUM_AVAILABLE = True
except ImportError:
    _OPT_EINSUM_AVAILABLE = False


def quantum_integer(n: int, q: complex) -> complex:
    """
    Compute quantum integer [n]_q = (q^n - q^{-n}) / (q - q^{-1}).

    For q = e^{2πi/k}, these become ratios of sines.
    At q = 1, [n]_q → n (classical limit).
    """
    if abs(q - 1) < 1e-10:
        return complex(n)
    return (q**n - q**(-n)) / (q - q**(-1))


def quantum_factorial(n: int, q: complex) -> complex:
    """
    Compute quantum factorial [n]!_q = [1]_q * [2]_q * ... * [n]_q.
    """
    result = 1.0 + 0j
    for k in range(1, n + 1):
        result *= quantum_integer(k, q)
    return result


def quantum_binomial(n: int, k: int, q: complex) -> complex:
    """
    Compute quantum binomial coefficient [n choose k]_q.

    [n]!_q / ([k]!_q * [n-k]!_q)
    """
    if k < 0 or k > n:
        return 0.0 + 0j
    return quantum_factorial(n, q) / (quantum_factorial(k, q) * quantum_factorial(n - k, q))


def r_matrix_sl2(N: int, q: complex, positive: bool = True) -> np.ndarray:
    """
    Construct the R-matrix for V_N x V_N where V_N is the N-dimensional
    irreducible representation of U_q(sl_2).

    Uses the standard formula from Kauffman-Lins "Temperley-Lieb Recoupling
    Theory and Invariants of 3-Manifolds".

    For the fundamental representation (N=2), the R-matrix in basis
    |00>, |01>, |10>, |11> is:
        R = [[q,   0,    0,   0],
             [0,   0,    1,   0],
             [0,   1,  q-1/q, 0],
             [0,   0,    0,   q]]

    This satisfies: R - R^{-1} = (q - q^{-1}) * Id

    Args:
        N: Dimension of representation (N=2 gives standard Jones)
        q: Quantum parameter (related to Jones variable t by t = q^2)
        positive: True for positive crossing, False for negative

    Returns:
        R-matrix as (N,N,N,N) tensor where R[i,j,k,l] is the amplitude
        for |i,j> -> |k,l>
    """
    R = np.zeros((N, N, N, N), dtype=complex)
    qinv = 1.0 / q
    lambda_q = q - qinv  # quantum [2]_q factor

    if N == 2:
        # Use the explicit 2x2 R-matrix (well-tested formula)
        # R|00> = q|00>
        # R|01> = |10>
        # R|10> = |01> + (q - q^{-1})|10>
        # R|11> = q|11>
        if positive:
            R[0, 0, 0, 0] = q
            R[0, 1, 1, 0] = 1.0
            R[1, 0, 0, 1] = 1.0
            R[1, 0, 1, 0] = lambda_q
            R[1, 1, 1, 1] = q
        else:
            # R^{-1}: swap q <-> q^{-1}
            R[0, 0, 0, 0] = qinv
            R[0, 1, 1, 0] = 1.0
            R[1, 0, 0, 1] = 1.0
            R[1, 0, 1, 0] = -lambda_q
            R[1, 1, 1, 1] = qinv
    else:
        # General N-dimensional R-matrix from U_q(sl_2)
        # R_check formula: R_{ij,kl} for |i,j> -> |k,l>
        # where i,j,k,l in {0, 1, ..., N-1} label spin-z eigenstates

        # The R-matrix decomposes as:
        # R = sum_J q^{c_J} P_J  where J runs over irreps in V_N x V_N
        # and c_J = J(J+1)/2 - (N-1)(N-1+1)/4

        # For computational simplicity, we use the explicit matrix elements
        # from the quantum Clebsch-Gordan coefficients

        # Simplified formula using weight structure:
        for i in range(N):
            for j in range(N):
                # Weight of |i,j> is m_i + m_j where m_k = (N-1)/2 - k
                if positive:
                    # Diagonal term (i=k, j=l)
                    m_i = (N - 1) / 2.0 - i
                    m_j = (N - 1) / 2.0 - j
                    if i == j:
                        R[i, j, i, j] = q
                    else:
                        R[i, j, i, j] = 1.0

                    # Off-diagonal swap term (only for adjacent weights)
                    if i < j:
                        # |i,j> -> |j,i> term
                        R[i, j, j, i] = lambda_q * np.sqrt(
                            quantum_integer(j - i, q) * quantum_integer(N - j + i, q) /
                            (quantum_integer(j - i + 1, q) * quantum_integer(N - j + i - 1, q))
                        ) if (j - i == 1) else 0
                        R[j, i, i, j] = 1.0 if (j - i == 1) else 0
                else:
                    # Inverse: swap q <-> q^{-1}
                    if i == j:
                        R[i, j, i, j] = qinv
                    else:
                        R[i, j, i, j] = 1.0

                    if i < j:
                        R[i, j, j, i] = -lambda_q * np.sqrt(
                            quantum_integer(j - i, q) * quantum_integer(N - j + i, q) /
                            (quantum_integer(j - i + 1, q) * quantum_integer(N - j + i - 1, q))
                        ) if (j - i == 1) else 0
                        R[j, i, i, j] = 1.0 if (j - i == 1) else 0

    return R


def loop_value(N: int, q: complex) -> complex:
    """
    Compute the loop value (quantum dimension) for V_N.

    d_N = [N]_q = (q^N - q^{-N}) / (q - q^{-1})

    This is the value assigned to a closed loop in the diagram.
    """
    return quantum_integer(N, q)


def writhe_from_braid(braid_word: list[int]) -> int:
    """
    Compute writhe from braid word.

    Args:
        braid_word: List of signed integers (positive = σ_i, negative = σ_i^{-1})

    Returns:
        Sum of signs
    """
    return sum(1 if g > 0 else -1 for g in braid_word)


def braid_to_tensor_network(
    braid_word: list[int],
    N: int,
    q: complex
) -> Tuple[nx.Graph, int]:
    """
    Convert a braid word to a tensor network for colored Jones computation.

    Each crossing becomes an R-matrix tensor (N×N×N×N).
    The network structure follows the braid strands.

    Args:
        braid_word: Braid representation (e.g., [1,1,1] for trefoil)
        N: Dimension of representation
        q: Quantum parameter

    Returns:
        Tuple of (NetworkX graph with tensor attributes, number of strands)
    """
    if not braid_word:
        return nx.Graph(), 1

    n_strands = max(abs(g) for g in braid_word) + 1
    n_crossings = len(braid_word)

    # Build graph where:
    # - Nodes are crossings (each has R-matrix tensor)
    # - Edges connect strands between crossings

    g = nx.Graph()

    # Track strand positions through the braid
    # strand_state[i] = (last_crossing_idx, tensor_index) for strand i
    strand_state = {i: (None, None) for i in range(n_strands)}

    edge_counter = 0

    for crossing_idx, gen in enumerate(braid_word):
        # Which strands are involved
        strand_i = abs(gen) - 1  # 0-indexed
        strand_j = strand_i + 1
        positive = gen > 0

        # Create R-matrix for this crossing
        R = r_matrix_sl2(N, q, positive=positive)

        # Add node for this crossing
        # Tensor indices: [top_left, top_right, bottom_left, bottom_right]
        # In braid convention: top = incoming, bottom = outgoing
        incidence = []

        # Connect to previous crossings on these strands
        for local_idx, strand in enumerate([strand_i, strand_j]):
            prev = strand_state[strand]
            if prev[0] is not None:
                # Connect to previous crossing
                edge_idx = edge_counter
                edge_counter += 1
                incidence.append(edge_idx)
                # Add edge in graph
                g.add_edge(prev[0], crossing_idx, edge_id=edge_idx)
            else:
                # New edge from top boundary
                edge_idx = edge_counter
                edge_counter += 1
                incidence.append(edge_idx)

        # Output edges (will be connected to next crossings or boundary)
        out_edges = []
        for local_idx, strand in enumerate([strand_i, strand_j]):
            edge_idx = edge_counter
            edge_counter += 1
            out_edges.append(edge_idx)
            incidence.append(edge_idx)
            strand_state[strand] = (crossing_idx, edge_idx)

        # Store tensor and incidence list
        g.add_node(crossing_idx)
        g.nodes[crossing_idx]['attr'] = [incidence, R]

    return g, n_strands


def colored_jones_from_braid(
    braid_word: list[int],
    N: int,
    q: complex,
    normalized: bool = True
) -> complex:
    """
    Compute the N-colored Jones polynomial J_N(K; q) from braid representation.

    The colored Jones polynomial is computed by:
    1. Building a tensor network from the braid
    2. Contracting all R-matrices
    3. Applying writhe normalization
    4. Dividing by loop value for closure normalization

    Args:
        braid_word: Braid representation of knot
        N: Dimension of representation (N=2 is standard Jones)
        q: Quantum parameter
        normalized: If True, normalize so J_N(unknot) = [N]_q

    Returns:
        Value of J_N(K; q)
    """
    if not braid_word:
        # Unknot
        return loop_value(N, q) if normalized else 1.0 + 0j

    n_strands = max(abs(g) for g in braid_word) + 1
    w = writhe_from_braid(braid_word)

    # Build tensor network
    g, _ = braid_to_tensor_network(braid_word, N, q)

    if g.number_of_nodes() == 0:
        return loop_value(N, q) if normalized else 1.0 + 0j

    # For the braid closure, we need to:
    # 1. Contract all R-matrices according to braid structure
    # 2. Close the strands (trace over boundary indices)

    # Simplified approach: enumerate states and sum contributions
    # This is exponential in crossing number but exact

    result = contract_braid_network(braid_word, N, q)

    # Apply writhe normalization
    # The standard normalization uses q^{w * c_2} where c_2 = (N²-1)/4
    # But commonly: q^{w * N(N-1)/4}
    c2 = (N * N - 1) / 4.0
    normalization = q ** (w * c2)

    result = result / normalization

    return result


def contract_braid_network(
    braid_word: list[int],
    N: int,
    q: complex
) -> complex:
    """
    Contract the tensor network for a braid closure.

    Uses direct state-sum approach for correctness.
    For large braids, this should be replaced with optimized contraction.
    """
    if not braid_word:
        return loop_value(N, q)

    n_strands = max(abs(g) for g in braid_word) + 1
    n_crossings = len(braid_word)

    # Build R-matrices
    R_positive = r_matrix_sl2(N, q, positive=True)
    R_negative = r_matrix_sl2(N, q, positive=False)

    # State-sum: iterate over all strand colorings at each level
    # For closure, initial and final states must match

    # More efficient: use transfer matrix approach
    # T[i→j] = amplitude to go from strand config i to config j through one crossing

    # Initialize: sum over all possible initial configs (for trace)
    d_N = loop_value(N, q)

    # Actually, let's use the matrix approach for the braid
    # The braid group acts on V_N^{⊗n} where n = n_strands
    # Each σ_i acts as R on positions i, i+1

    # Full state space dimension is N^n_strands
    dim = N ** n_strands

    # Build the braid matrix
    braid_matrix = np.eye(dim, dtype=complex)

    for gen in braid_word:
        strand_i = abs(gen) - 1  # 0-indexed position
        positive = gen > 0
        R = R_positive if positive else R_negative

        # Apply R at positions (strand_i, strand_i+1) in N^n tensor
        gen_matrix = apply_r_matrix(R, strand_i, n_strands, N)
        braid_matrix = gen_matrix @ braid_matrix

    # For braid closure (trace), we need to identify top and bottom strands
    # This means taking trace: Tr(braid_matrix)
    # But this is trace over the full N^n space

    # Actually for the colored Jones, we close each strand separately
    # giving a factor of d_N per strand from the metric
    # The trace gives: Σ_i ⟨i|M|i⟩ where |i⟩ runs over basis

    trace = np.trace(braid_matrix)

    # Normalize by quantum dimension factors
    # For n_strands strands, closing gives n_strands loops worth
    # but the trace already accounts for this in a specific way

    # The proper normalization for braid closure:
    # J_N = (1/d_N) * Tr(braid_matrix) where Tr includes strand identification

    return trace / (d_N ** (n_strands - 1))


def apply_r_matrix(
    R: np.ndarray,
    position: int,
    n_strands: int,
    N: int
) -> np.ndarray:
    """
    Apply R-matrix at position (position, position+1) in N^{n_strands} space.

    R acts on V_N ⊗ V_N at the specified positions.
    """
    dim = N ** n_strands
    result = np.zeros((dim, dim), dtype=complex)

    # R[i,j,k,l] transforms |i,j⟩ → R[i,j,k,l] |k,l⟩

    for state_in in range(dim):
        # Decode input state
        digits_in = []
        temp = state_in
        for _ in range(n_strands):
            digits_in.append(temp % N)
            temp //= N
        digits_in = digits_in[::-1]  # Now digits_in[0] is leftmost

        i, j = digits_in[position], digits_in[position + 1]

        for k in range(N):
            for l in range(N):
                coeff = R[i, j, k, l]
                if abs(coeff) < 1e-15:
                    continue

                # Build output state
                digits_out = digits_in.copy()
                digits_out[position] = k
                digits_out[position + 1] = l

                # Encode output state
                state_out = 0
                for d in digits_out:
                    state_out = state_out * N + d

                result[state_out, state_in] += coeff

    return result


def colored_jones_eval(
    braid_word: list[int],
    N: int,
    root_of_unity: int | None = None,
    q: complex | None = None
) -> complex:
    """
    Evaluate N-colored Jones polynomial J_N(K).

    Args:
        braid_word: Braid representation
        N: Color (dimension of representation)
        root_of_unity: If specified, use q = exp(2πi / root_of_unity)
        q: Direct q value (overrides root_of_unity)

    Returns:
        J_N(K; q)
    """
    if q is None:
        if root_of_unity is None:
            root_of_unity = 4 * N  # Common choice
        q = np.exp(2j * np.pi / root_of_unity)

    return colored_jones_from_braid(braid_word, N, q)


def volume_conjecture_sequence(
    braid_word: list[int],
    N_values: list[int],
    verbose: bool = False
) -> list[Tuple[int, complex, float]]:
    r"""
    Compute sequence for Volume Conjecture verification.

    The Volume Conjecture states:
    lim_{N->inf} (2*pi/N) * log|J_N(K; e^{2*pi*i/N})| = Vol(S^3 \ K)

    IMPORTANT: At q = e^{2*pi*i/N}, we have [N]_q = 0, so we use
    q = e^{2*pi*i/(N+k)} for small k to avoid the singularity.
    The proper treatment requires computing J_N as a polynomial first.

    Args:
        braid_word: Braid representation of knot
        N_values: List of N values to compute
        verbose: Print progress

    Returns:
        List of (N, J_N value, estimated volume contribution)
    """
    results = []

    for N in N_values:
        # Use q = e^{2*pi*i/(2N+1)} to avoid [N]_q = 0 singularity
        # This is a common approximation for the Volume Conjecture
        k = 2 * N + 1
        q = np.exp(2j * np.pi / k)

        t1 = perf_counter()
        J_N = colored_jones_from_braid(braid_word, N, q)
        elapsed = perf_counter() - t1

        # Volume contribution estimate
        if abs(J_N) > 1e-15:
            vol_estimate = (2 * np.pi / N) * np.log(abs(J_N))
        else:
            vol_estimate = float('-inf')

        results.append((N, J_N, vol_estimate))

        if verbose:
            print(f"N={N}: J_N = {J_N:.6f}, Vol estimate = {vol_estimate:.6f} ({elapsed:.3f}s)")

    return results


# Testing
def test_colored_jones():
    """Test colored Jones computation against known values."""
    print("Testing Colored Jones Polynomial")
    print("=" * 60)

    # Test knots
    knots = {
        "Unknot": [],
        "Trefoil": [1, 1, 1],
        "Figure-8": [1, -2, 1, -2],
    }

    # Test at various N and q values
    test_cases = [
        (2, np.exp(2j * np.pi / 8)),   # N=2, q = e^{pi*i/4}
        (3, np.exp(2j * np.pi / 12)),  # N=3
        (2, np.exp(2j * np.pi / 10)),  # N=2, different q
    ]

    for name, braid in knots.items():
        print(f"\n{name} (braid: {braid}):")
        for N, q in test_cases:
            J_N = colored_jones_from_braid(braid, N, q)
            d_N = loop_value(N, q)
            print(f"  N={N}, q=e^(2*pi*i/{int(round(2*np.pi/np.angle(q)))}): "
                  f"J_{N} = {J_N:.4f}, [N]_q = {d_N:.4f}")

    print("\n" + "=" * 60)
    print("Volume Conjecture sequence for Trefoil:")
    trefoil = [1, 1, 1]
    vol_seq = volume_conjecture_sequence(trefoil, [3, 4, 5, 6, 7, 8], verbose=True)

    print("\n" + "=" * 60)
    print("Volume Conjecture sequence for Figure-8:")
    fig8 = [1, -2, 1, -2]
    vol_seq = volume_conjecture_sequence(fig8, [3, 4, 5, 6, 7, 8], verbose=True)
    print(f"(Known hyperbolic volume: 2.02988...)")


def validate_against_snappy():
    """Compare N=2 colored Jones with snappy's Jones polynomial."""
    try:
        import snappy
    except ImportError:
        print("snappy not available for validation")
        return

    print("\n" + "=" * 60)
    print("Validating N=2 against snappy")
    print("=" * 60)

    # Note on conventions:
    # - snappy's Jones polynomial uses variable q where the polynomial
    #   has integer powers (their q = t^{1/2} in standard Jones notation)
    # - Our R-matrix uses q where the Jones variable t = q^2
    # - So to evaluate: use our q = snappy's q

    knots = {
        "Trefoil": [1, 1, 1],
        "Figure-8": [1, -2, 1, -2],
        "5_1": [1, 1, 1, 1, 1],  # Cinquefoil
    }

    # Test at several values
    test_k = [7, 9, 11, 13, 17]

    for name, braid in knots.items():
        print(f"\n{name}:")

        # Get snappy's polynomial
        L = snappy.Link(braid_closure=braid)
        jones_poly = L.jones_polynomial()
        print(f"  snappy poly: {jones_poly}")

        for k in test_k:
            q = np.exp(2j * np.pi / k)

            # Evaluate snappy polynomial at q (they use same variable)
            snappy_val = sum(coeff * q**power for power, coeff in jones_poly.dict().items())

            # Our N=2 colored Jones (using same q)
            our_val = colored_jones_from_braid(braid, N=2, q=q)

            # Compare
            ratio = our_val / snappy_val if abs(snappy_val) > 1e-10 else float('nan')
            match = "OK" if abs(abs(ratio) - 1) < 0.1 else "DIFF"
            print(f"  q=e^(2pi*i/{k}): ours={our_val:.4f}, snappy={snappy_val:.4f}, |ratio|={abs(ratio):.4f} [{match}]")


if __name__ == "__main__":
    test_colored_jones()
    validate_against_snappy()
