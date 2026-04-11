/**
 * LatticeRenderer Module
 * Handles professional canvas drawing without glow.
 */
export class LatticeRenderer {
  constructor(canvasEl, atomColors) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext('2d');
    this.atomColors = atomColors;
    this.gridSize = 8;
  }

  draw(grid) {
    const { ctx, canvas } = this;
    const size = canvas.width;
    const cellSize = size / this.gridSize;

    ctx.clearRect(0, 0, size, size);
    
    // Draw grid lines
    ctx.strokeStyle = '#2d3748'; /* Match bg-border */
    ctx.lineWidth = 1;
    for (let i = 0; i <= this.gridSize; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0); ctx.lineTo(i * cellSize, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * cellSize); ctx.lineTo(size, i * cellSize);
      ctx.stroke();
    }

    // Draw atoms
    grid.forEach((row, r) => {
      row.forEach((cell, c) => {
        if (cell !== '.') {
          this.drawAtom(r, c, cell, cellSize);
        }
      });
    });
  }

  drawAtom(r, c, type, cellSize) {
    const { ctx } = this;
    const x = c * cellSize + cellSize / 2;
    const y = r * cellSize + cellSize / 2;
    const radius = cellSize * 0.38;

    // Draw a clean flat circle
    ctx.fillStyle = this.atomColors[type] || '#fff';
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    
    // Subtle border for the atom
    ctx.strokeStyle = 'rgba(0,0,0,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Small highlight for depth (not a glow)
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    ctx.beginPath();
    ctx.arc(x - radius * 0.3, y - radius * 0.3, radius * 0.2, 0, Math.PI * 2);
    ctx.fill();
  }
}
