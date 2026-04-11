/**
 * LatticeRenderer Module
 * Handles high-fidelity canvas drawing.
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
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
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
    const radius = cellSize * 0.35;

    ctx.fillStyle = this.atomColors[type] || '#fff';
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    
    // Glow effect
    ctx.shadowBlur = 10;
    ctx.shadowColor = ctx.fillStyle;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}
