"""Lattice grid engine for the MaterialForge environment."""

from __future__ import annotations

import copy
from typing import Dict, List

from .config import ATOM_TYPES, EMPTY, GRID_SIZE


class Lattice:
    """An 8x8 grid representing atomic crystal structure placement."""

    def __init__(self, size: int = GRID_SIZE):
        self.size = size
        self._grid: List[List[str]] = [[EMPTY] * size for _ in range(size)]

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def place(self, row: int, col: int, atom: str) -> bool:
        """Place an atom on an empty cell. Returns True if successful."""
        if not self._in_bounds(row, col):
            return False
        if self._grid[row][col] != EMPTY:
            return False
        if atom not in ATOM_TYPES:
            return False
        self._grid[row][col] = atom
        return True

    def replace(self, row: int, col: int, atom: str) -> bool:
        """Replace an existing atom with a different one. Returns True if successful."""
        if not self._in_bounds(row, col):
            return False
        if self._grid[row][col] == EMPTY:
            return False
        if atom not in ATOM_TYPES:
            return False
        if self._grid[row][col] == atom:
            return False
        self._grid[row][col] = atom
        return True

    def remove(self, row: int, col: int) -> bool:
        """Remove an atom from the grid. Returns True if successful."""
        if not self._in_bounds(row, col):
            return False
        if self._grid[row][col] == EMPTY:
            return False
        self._grid[row][col] = EMPTY
        return True

    def get(self, row: int, col: int) -> str:
        """Get the content of a cell."""
        if not self._in_bounds(row, col):
            return EMPTY
        return self._grid[row][col]

    def get_grid(self) -> List[List[str]]:
        """Return a copy of the grid."""
        return [row[:] for row in self._grid]

    def count_atoms(self) -> Dict[str, int]:
        """Count each atom type on the grid."""
        counts: Dict[str, int] = {sym: 0 for sym in ATOM_TYPES}
        for row in self._grid:
            for cell in row:
                if cell in ATOM_TYPES:
                    counts[cell] += 1
        return counts

    def atom_count(self) -> int:
        """Total number of non-empty cells."""
        return sum(
            1 for row in self._grid for cell in row if cell != EMPTY
        )

    def total_cost(self) -> float:
        """Sum of atom costs on the grid."""
        total = 0.0
        for row in self._grid:
            for cell in row:
                if cell in ATOM_TYPES:
                    total += ATOM_TYPES[cell]["cost"]
        return total

    def get_neighbors(self, row: int, col: int) -> List[str]:
        """Get contents of 8-connected neighbor cells (excluding out-of-bounds)."""
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if self._in_bounds(nr, nc):
                    neighbors.append(self._grid[nr][nc])
        return neighbors

    def clone(self) -> Lattice:
        """Create a deep copy of this lattice."""
        new = Lattice(self.size)
        new._grid = copy.deepcopy(self._grid)
        return new

    @classmethod
    def from_grid(cls, grid: List[List[str]]) -> Lattice:
        """Reconstruct a Lattice from a grid (e.g., from serialized state)."""
        size = len(grid)
        lattice = cls(size)
        lattice._grid = [row[:] for row in grid]
        return lattice
