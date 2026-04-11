import { LabAPI } from './modules/api.js';
import { LatticeRenderer } from './modules/renderer.js';
import { TelemetryUI } from './modules/ui.js';

class DiscoveryLab {
  constructor() {
    this.api = new LabAPI(window.location.host);
    this.ui  = new TelemetryUI();
    this.renderer = new LatticeRenderer(
      document.getElementById('lattice-canvas'),
      { 'A': '#ef4444', 'B': '#3b82f6', 'C': '#f59e0b', 'P': '#8b5cf6' }
    );

    this.state = {
      selectedAtom: 'A',
      bestReward: 0
    };

    this.init();
  }

  async init() {
    this.bindEvents();
    try {
      await this.api.connect('/ws');
      this.ui.log('Lab connection established', 'ok');
      this.reset();
    } catch (e) {
      this.ui.log('Failed to connect to lab engine', 'err');
    }

    this.api.onMessage((msg) => this.handleMessage(msg));
  }

  bindEvents() {
    document.getElementById('btn-reset').onclick = () => this.reset();
    
    // Canvas interaction
    this.renderer.canvas.onclick = (e) => this.handleCanvasClick(e);
    this.renderer.canvas.oncontextmenu = (e) => {
      e.preventDefault();
      this.handleCanvasClick(e, true);
    };

    // Atom selection
    document.querySelectorAll('.atom-btn').forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll('.atom-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        this.state.selectedAtom = btn.dataset.atom;
      };
    });
  }

  handleMessage(msg) {
    if (msg.type === 'observation') {
      const { observation, reward } = msg.data;
      if (reward > this.state.bestReward) this.state.bestReward = reward;
      
      this.renderer.draw(observation.grid);
      this.ui.updateKPIs(observation, reward, this.state.bestReward);
      this.ui.updatePropertyBars(observation);
    }
  }

  reset() {
    const scenario = document.getElementById('sel-scenario').value;
    const difficulty = document.getElementById('sel-difficulty').value;
    this.api.send('reset', { scenario_name: scenario, difficulty });
    this.state.bestReward = 0;
    this.ui.log(`Workspace re-initialized: ${scenario || 'random'}`, 'info');
  }

  handleCanvasClick(e, isRightClick = false) {
    const rect = this.renderer.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cellSize = this.renderer.canvas.width / 8;
    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);

    const action = isRightClick ? 'remove' : 'place';
    this.api.send('step', {
      action_type: action,
      row, col,
      atom: this.state.selectedAtom
    });
  }
}

// Start lab
window.addEventListener('load', () => {
  window.lab = new DiscoveryLab();
  document.getElementById('loading-overlay').classList.add('hidden');
});
