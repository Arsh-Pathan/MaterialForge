"""Heuristic property estimation, phase classification, and structural scoring."""

from typing import Dict

try:
    from .config import ATOM_TYPES, EMPTY, PROPERTY_NAMES
    from .lattice import Lattice
except ImportError:
    from config import ATOM_TYPES, EMPTY, PROPERTY_NAMES
    from lattice import Lattice


def estimate_properties(lattice: Lattice) -> Dict[str, float]:
    """Estimate material properties (0-100) from the current lattice state.

    Each property is a weighted combination of:
    - Atom fraction * base contribution
    - Neighbor bonding bonus
    - Density/packing bonus
    """
    total_cells = lattice.size * lattice.size
    n_atoms = lattice.atom_count()

    if n_atoms == 0:
        return {p: 0.0 for p in PROPERTY_NAMES}

    counts = lattice.count_atoms()
    density = n_atoms / total_cells

    # Base contributions from atom fractions
    props: Dict[str, float] = {p: 0.0 for p in PROPERTY_NAMES}
    for sym, count in counts.items():
        if count == 0:
            continue
        frac = count / n_atoms
        for prop in PROPERTY_NAMES:
            props[prop] += frac * ATOM_TYPES[sym]["contributions"][prop]

    # Neighbor bonding bonus: atoms with same-type neighbors boost their contributions
    bonding_bonus: Dict[str, float] = {p: 0.0 for p in PROPERTY_NAMES}
    for r in range(lattice.size):
        for c in range(lattice.size):
            cell = lattice.get(r, c)
            if cell == EMPTY:
                continue
            neighbors = lattice.get_neighbors(r, c)
            same_type = sum(1 for n in neighbors if n == cell)
            occupied = sum(1 for n in neighbors if n != EMPTY)
            if same_type > 0:
                # Same-type clustering boosts the atom's primary property
                for prop in PROPERTY_NAMES:
                    bonding_bonus[prop] += (
                        ATOM_TYPES[cell]["contributions"][prop]
                        * same_type
                        / (8.0 * n_atoms)
                    )
            if occupied > 0 and cell == "B":
                # B-B connectivity boosts conductivity specifically
                b_neighbors = sum(1 for n in neighbors if n == "B")
                bonding_bonus["conductivity"] += b_neighbors / (8.0 * n_atoms)

    # Density bonus: more filled grid generally means stronger properties
    density_mult = 0.6 + 0.4 * density

    # Void fraction helps elasticity (some voids = flexible)
    void_fraction = 1.0 - density
    elasticity_void_bonus = 0.15 * min(void_fraction / 0.5, 1.0) if void_fraction > 0.1 else 0.0

    # Scale to 0-100.  With primary contribution = 1.0, a full grid of one
    # atom type should be able to reach ~95 for its primary property.
    # base_max = 1.0 * 80 = 80, bonding_max ~ 15, density_max = 10 → ~105 clamped to 100.
    result: Dict[str, float] = {}
    for prop in PROPERTY_NAMES:
        base = props[prop] * 80.0  # base contribution (up to ~80)
        bonding = bonding_bonus[prop] * 25.0  # bonding bonus (up to ~15)
        dens = (density_mult - 0.6) * 25.0  # density bonus (up to ~10)
        val = base + bonding + dens
        if prop == "elasticity":
            val += elasticity_void_bonus * 100.0
        result[prop] = round(min(max(val, 0.0), 100.0), 2)

    return result


def classify_phase(lattice: Lattice) -> str:
    """Classify the crystal phase based on structural regularity.

    Returns "crystalline", "polycrystalline", or "amorphous".
    """
    n_atoms = lattice.atom_count()
    if n_atoms < 4:
        return "amorphous"

    grid = lattice.get_grid()
    size = lattice.size

    # Check for repeating 2x2 sub-patterns
    pattern_counts: Dict[str, int] = {}
    for r in range(size - 1):
        for c in range(size - 1):
            block = (grid[r][c], grid[r][c + 1], grid[r + 1][c], grid[r + 1][c + 1])
            # Skip all-empty blocks
            if all(cell == EMPTY for cell in block):
                continue
            key = "|".join(block)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1

    if not pattern_counts:
        return "amorphous"

    total_blocks = sum(pattern_counts.values())
    max_pattern = max(pattern_counts.values())
    top_patterns = sorted(pattern_counts.values(), reverse=True)

    # High regularity = crystalline
    if max_pattern / total_blocks > 0.35:
        return "crystalline"

    # Multiple distinct regions with local regularity = polycrystalline
    if len(top_patterns) >= 2 and top_patterns[1] / total_blocks > 0.15:
        return "polycrystalline"

    # Row/column periodicity check
    row_repeats = 0
    for r in range(size):
        occupied = [grid[r][c] for c in range(size) if grid[r][c] != EMPTY]
        if len(occupied) >= 2 and len(set(occupied)) == 1:
            row_repeats += 1
    col_repeats = 0
    for c in range(size):
        occupied = [grid[r][c] for r in range(size) if grid[r][c] != EMPTY]
        if len(occupied) >= 2 and len(set(occupied)) == 1:
            col_repeats += 1

    regularity = (row_repeats + col_repeats) / (2 * size)
    if regularity > 0.4:
        return "crystalline"
    if regularity > 0.2:
        return "polycrystalline"

    return "amorphous"


def compute_stability(lattice: Lattice) -> float:
    """Compute structural stability score (0.0 to 1.0).

    Penalizes isolated atoms, rewards well-connected atoms and symmetry.
    """
    n_atoms = lattice.atom_count()
    if n_atoms == 0:
        return 0.0

    # Connectivity score
    connectivity = 0.0
    for r in range(lattice.size):
        for c in range(lattice.size):
            if lattice.get(r, c) == EMPTY:
                continue
            neighbors = lattice.get_neighbors(r, c)
            occupied_neighbors = sum(1 for n in neighbors if n != EMPTY)
            if occupied_neighbors == 0:
                connectivity -= 1.0  # isolated penalty
            elif occupied_neighbors >= 3:
                connectivity += 1.0  # well-connected bonus
            else:
                connectivity += 0.3 * occupied_neighbors

    connectivity_score = min(max(connectivity / n_atoms, -1.0), 1.0)
    connectivity_score = (connectivity_score + 1.0) / 2.0  # normalize to 0-1

    # Symmetry bonus: compare horizontal and vertical mirrors
    grid = lattice.get_grid()
    size = lattice.size
    h_match = 0
    v_match = 0
    total_pairs = 0

    for r in range(size):
        for c in range(size // 2):
            mirror_c = size - 1 - c
            total_pairs += 1
            if grid[r][c] == grid[r][mirror_c]:
                h_match += 1

    for c in range(size):
        for r in range(size // 2):
            mirror_r = size - 1 - r
            if grid[r][c] == grid[mirror_r][c]:
                v_match += 1

    if total_pairs > 0:
        symmetry = max(h_match, v_match) / total_pairs
    else:
        symmetry = 0.0

    return round(min(0.7 * connectivity_score + 0.3 * symmetry, 1.0), 4)


def compute_lattice_quality(lattice: Lattice) -> float:
    """Compute lattice structural quality (0.0 to 1.0).

    Measures structural order and uniformity of atom distribution.
    """
    n_atoms = lattice.atom_count()
    if n_atoms == 0:
        return 0.0

    size = lattice.size
    grid = lattice.get_grid()

    # Measure regularity: count atoms in even spacing patterns
    regular_atoms = 0
    for r in range(size):
        for c in range(size):
            if grid[r][c] == EMPTY:
                continue
            neighbors = lattice.get_neighbors(r, c)
            occupied = sum(1 for n in neighbors if n != EMPTY)
            # Atoms with 2-4 neighbors are in regular structures
            if 2 <= occupied <= 4:
                regular_atoms += 1

    regularity = regular_atoms / n_atoms if n_atoms > 0 else 0.0

    # Distribution uniformity: split grid into quadrants, compare atom counts
    half = size // 2
    quadrants = [0, 0, 0, 0]
    for r in range(size):
        for c in range(size):
            if grid[r][c] == EMPTY:
                continue
            qi = (0 if r < half else 2) + (0 if c < half else 1)
            quadrants[qi] += 1

    if n_atoms > 0:
        expected = n_atoms / 4.0
        deviation = sum(abs(q - expected) for q in quadrants) / (4.0 * max(expected, 1))
        uniformity = max(1.0 - deviation, 0.0)
    else:
        uniformity = 0.0

    return round(min(0.6 * regularity + 0.4 * uniformity, 1.0), 4)
