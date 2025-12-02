"""
EXAMPLE: EVALUATING JONES POLYNOMIAL OF KNOTS (Modern Version)
==============================================================
( example_jones_modern.py )
original by Konstantinos Meichanetzidis (phykme@leeds.ac.uk)
modernized: 2024

Example script demonstrating the evaluation of the Jones
Polynomial in terms of the q-state Potts model partition
function via tensor network contraction. Example knots are
provided in a dictionary in Planar Diagram presentation
along with the analytic form of their Jones polynomial.

DEPENDENCIES: tensorcsp_modern.py, knut_modern.py
"""

from typing import Callable
import numpy as np

from knut_modern import (
    pd2tait, taitnumber, writhe, tpotts,
    DeltaH_greedy, Jones_greedy
)


# Jones polynomial functions for various knots
def jpoly_3_1(t: complex) -> complex:
    """Jones polynomial for trefoil knot (3_1)."""
    return -t**-4 + t**-3 + t**-1


def jpoly_4_1(t: complex) -> complex:
    """Jones polynomial for figure-eight knot (4_1)."""
    return t**2 + t**-2 - t - t**-1 + 1


def jpoly_5_2(t: complex) -> complex:
    """Jones polynomial for knot 5_2."""
    return -t**-6 + t**-5 - t**-4 + 2*t**-3 - t**-2 + t**-1


def jpoly_6_3(t: complex) -> complex:
    """Jones polynomial for knot 6_3."""
    return -t**3 + 2*t**2 - 2*t + 3 - 2*t**-1 + 2*t**-2 - t**-3


def jpoly_7_6(t: complex) -> complex:
    """Jones polynomial for knot 7_6."""
    return t - 2 + 3*t**-1 - 3*t**-2 + 4*t**-3 - 3*t**-4 + 2*t**-5 - t**-6


def jpoly_8_10(t: complex) -> complex:
    """Jones polynomial for knot 8_10."""
    return -t**6 + 2*t**5 - 4*t**4 + 5*t**3 - 4*t**2 + 5*t - 3 + 2*t**-1 - t**-2


def jpoly_9_14(t: complex) -> complex:
    """Jones polynomial for knot 9_14."""
    return (t**6 - 2*t**5 + 3*t**4 - 5*t**3 + 6*t**2 - 6*t + 6
            - 4*t**-1 + 3*t**-2 - t**-3)


# Dictionary of knots: knots['name'] = [planar_diagram, jones_function]
KNOTS: dict[str, tuple[list, Callable]] = {
    '3_1': (
        [[1, 4, 2, 5], [3, 6, 4, 1], [5, 2, 6, 3]],
        jpoly_3_1
    ),
    '4_1': (
        [[4, 2, 5, 1], [8, 6, 1, 5], [6, 3, 7, 4], [2, 7, 3, 8]],
        jpoly_4_1
    ),
    '5_2': (
        [[1, 4, 2, 5], [3, 8, 4, 9], [5, 10, 6, 1], [9, 6, 10, 7], [7, 2, 8, 3]],
        jpoly_5_2
    ),
    '6_3': (
        [[4, 2, 5, 1], [8, 4, 9, 3], [12, 9, 1, 10], [10, 5, 11, 6],
         [6, 11, 7, 12], [2, 8, 3, 7]],
        jpoly_6_3
    ),
    '7_6': (
        [[1, 4, 2, 5], [3, 8, 4, 9], [5, 12, 6, 13], [9, 1, 10, 14],
         [13, 11, 14, 10], [11, 6, 12, 7], [7, 2, 8, 3]],
        jpoly_7_6
    ),
    '8_10': (
        [[1, 4, 2, 5], [3, 8, 4, 9], [9, 15, 10, 14], [5, 13, 6, 12],
         [13, 7, 14, 6], [11, 1, 12, 16], [15, 11, 16, 10], [7, 2, 8, 3]],
        jpoly_8_10
    ),
    '9_14': (
        [[1, 4, 2, 5], [5, 12, 6, 13], [3, 11, 4, 10], [11, 3, 12, 2],
         [13, 18, 14, 1], [9, 15, 10, 14], [7, 17, 8, 16], [15, 9, 16, 8],
         [17, 7, 18, 6]],
        jpoly_9_14
    ),
}


def main():
    """Demonstrate Jones polynomial computation via tensor networks."""

    # Choose example knot by name
    knot_name = '5_2'

    print("=" * 70)
    print(f"Computing Jones Polynomial for Knot {knot_name}")
    print("=" * 70)
    print()

    # Get knot data
    X, jones = KNOTS[knot_name]
    X = np.array(X)

    # Convert to Tait graph (signed edge list)
    c = pd2tait(X)

    # Compute knot invariants
    tau = taitnumber(c)  # Tait number
    w = writhe(X)        # Writhe

    print(f"Knot: {knot_name}")
    print(f"  Number of crossings: {len(X)}")
    print(f"  Tait number (tau): {tau}")
    print(f"  Writhe (w): {w}")
    print()

    # Compute maximal degree during greedy contraction
    # (sets q=1 to minimize memory, contracts graph without tensors)
    DH_greedy = DeltaH_greedy(c)
    print(f"Maximal degree during contraction: {DH_greedy}")
    print()

    # Choose a q value and compute Jones polynomial
    q = 5

    print(f"Computing Jones polynomial at q = {q}...")
    jpoly_computed, runtime = Jones_greedy(c, tau, w, q)

    t = tpotts(q)
    jpoly_analytic = jones(t)

    print()
    print("Results:")
    print("-" * 50)
    print(f"  Computed V(t(q={q})): {jpoly_computed}")
    print(f"  Analytic V(t(q={q})): {jpoly_analytic}")
    print(f"  Runtime: {runtime:.6f} seconds")
    print()

    # Check agreement
    if np.isclose(jpoly_computed, jpoly_analytic, rtol=1e-10):
        print("SUCCESS: Computed and analytic values agree!")
    else:
        diff = abs(jpoly_computed - jpoly_analytic)
        print(f"WARNING: Values differ by {diff}")
    print()

    # Demonstrate with multiple knots
    print("=" * 70)
    print("Computing Jones Polynomial for All Available Knots")
    print("=" * 70)
    print()

    for name, (diagram, jones_fn) in KNOTS.items():
        X = np.array(diagram)
        c = pd2tait(X)
        tau = taitnumber(c)
        w = writhe(X)

        jpoly_computed, runtime = Jones_greedy(c, tau, w, q)
        jpoly_analytic = jones_fn(tpotts(q))

        match = "OK" if np.isclose(jpoly_computed, jpoly_analytic, rtol=1e-10) else "MISMATCH"
        print(f"  Knot {name}: {match} (computed in {runtime:.4f}s)")


if __name__ == "__main__":
    main()
