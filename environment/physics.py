# Heuristic property estimation, phase classification, and structural scoring.
# This module acts as the "Physics Engine" of the environment.

from typing import Dict

try:
    from .config import ATOM_TYPES, EMPTY, PROPERTY_NAMES
    from .lattice import Lattice
except ImportError:
    from config import ATOM_TYPES, EMPTY, PROPERTY_NAMES
    from lattice import Lattice

from collections import deque


# Main entry point for property calculation: converts grid state to physical metrics.
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
            
            # Connectivity-driven conductivity: B-B bonds boost significantly
            if cell == "B":
                b_neighbors = sum(1 for n in neighbors if n == "B")
                bonding_bonus["conductivity"] += b_neighbors * 0.05

    # Percolation check: Does a continuous path of B atoms span the grid?
    # This simulates long-range conductivity pathways.
    percolation_bonus = 0.0
    if counts.get("B", 0) >= lattice.size:
        b_coords = [(r, c) for r in range(lattice.size) for c in range(lattice.size) if lattice.get(r, c) == "B"]
        if b_coords:
            # BFS to find if top spans to bottom
            start_nodes = [node for node in b_coords if node[0] == 0]
            if start_nodes:
                visited = set(start_nodes)
                queue = deque(start_nodes)
                while queue:
                    r, c = queue.popleft()
                    if r == lattice.size - 1:
                        percolation_bonus = 15.0 # Found path
                        break
                    for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                        if (nr, nc) in b_coords and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

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
        
        if prop == "conductivity":
            val += percolation_bonus
        if prop == "elasticity":
            val += elasticity_void_bonus * 100.0
        result[prop] = round(min(max(val, 0.0), 100.0), 2)

    return result


# Determines the crystalline quality of the lattice based on pattern repetition.
def classify_phase(lattice: Lattice) -> str:
    """Classify the crystal phase based on structural regularity.

    Returns "crystalline", "polycrystalline", or "amorphous".
    """
    n_atoms = lattice.atom_count()
    if n_atoms < 4:
        return "amorphous"

    grid = lattice.get_grid()
    size = lattice.size

    # Check for repeating 2x2 sub-patterns to detect long-range order.
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

    # Row/column periodicity check for 1D symmetry.
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


# Calculates structural integrity based on bond density and mirror symmetry.
def compute_structural_stability(lattice: Lattice) -> float:
    """Compute structural stability score (0.0 to 1.0).
    
    Analyses the Gibbs-like stability of the lattice configuration.
    This logic specifically penalizes 1D 'linear' chains and rewards 2D clusters.
    """
    n_atoms = lattice.atom_count()
    if n_atoms == 0:
        return 0.0

    # Coordination-based stability: Non-linear rewards for high neighbor density.
    # Penalizes dangling bonds (1 neighbor) and isolated atoms (0 neighbors).
    coordination_energy = 0.0
    for r in range(lattice.size):
        for c in range(lattice.size):
            if lattice.get(r, c) == EMPTY:
                continue
            neighbors = lattice.get_neighbors(r, c)
            cn = sum(1 for n in neighbors if n != EMPTY)
            
            if cn == 0:
                coordination_energy -= 1.5  # Isolated atom
            elif cn == 1:
                coordination_energy -= 0.5  # Dangling bond (end of a line)
            elif cn == 2:
                coordination_energy += 0.3  # Simple line (low stability)
            elif cn == 3:
                coordination_energy += 1.0  # Cluster edge (good stability)
            else:
                coordination_energy += 1.6  # Crystal core (excellent stability)

    # Normalize based on atom count and clamp to [0, 1].
    # A perfectly packed 3x3 square would have high cn, maximizing this score.
    stability_norm = min(max(coordination_energy / n_atoms, -1.0), 1.0)
    stability_norm = (stability_norm + 1.0) / 2.0

    # Point-group Symmetry (Mirror Planes) check.
    grid = lattice.get_grid()
    size = lattice.size
    h_mirrors = 0
    v_mirrors = 0
    total_checks = 0

    for r in range(size):
        for c in range(size // 2):
            total_checks += 1
            if grid[r][c] == grid[r][size - 1 - c]:
                h_mirrors += 1

    for c in range(size):
        for r in range(size // 2):
            if grid[r][c] == grid[size - 1 - r][c]:
                v_mirrors += 1

    symmetry_factor = (h_mirrors + v_mirrors) / (2 * total_checks) if total_checks > 0 else 0.0

    return round(min(0.65 * stability_norm + 0.35 * symmetry_factor, 1.0), 4)


# Measures the positional entropy of the atomic arrangement.
def compute_lattice_order(lattice: Lattice) -> float:
    """Compute lattice structural order (0.0 to 1.0).
    
    Measures the positional entropy and distribution uniformity of the 
    atomic ensemble.
    """
    n_atoms = lattice.atom_count()
    if n_atoms == 0:
        return 0.0

    size = lattice.size
    grid = lattice.get_grid()

    # Bragg-like Order: rewards alignment with periodic lattice sites.
    order_metric = 0
    for r in range(size):
        for c in range(size):
            if grid[r][c] == EMPTY:
                continue
            # Atoms on even/odd intersections contribute to cubic order
            if (r + c) % 2 == 0:
                order_metric += 1

    order_factor = order_metric / n_atoms if n_atoms > 0 else 0.0

    # Quadrant Distribution Entropy: measures how evenly atoms are distributed.
    half = size // 2
    quads = [0, 0, 0, 0]
    for r in range(size):
        for c in range(size):
            if grid[r][c] == EMPTY:
                continue
            idx = (0 if r < half else 2) + (0 if c < half else 1)
            quads[idx] += 1

    expected_q = n_atoms / 4.0
    q_deviation = sum(abs(q - expected_q) for q in quads) / (4.0 * max(expected_q, 1))
    homogeneity = max(1.0 - q_deviation, 0.0)

    return round(min(0.55 * order_factor + 0.45 * homogeneity, 1.0), 4)
