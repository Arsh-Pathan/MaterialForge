/**
 * MaterialForge — Crystal Structure Design Lab
 * Modular Frontend Application
 */

class MaterialForgeApp {
  constructor() {
    this.API_BASE = window.location.origin;
    this.SESSION_ID = null;
    this.state = {
      rewardHistory: [],
      bestReward: 0,
      currentObs: null,
      selectedAtom: 'A',
      selectedAction: 'place',
      hoverPos: { r: -1, c: -1 }
    };

    // Configuration
    this.ATOM_COLORS = { A: '#e74c3c', B: '#3b82f6', C: '#f59e0b', P: '#10b981', '.': '#111827' };
    this.ATOM_GLOW   = { A: '#e74c3c', B: '#3b82f6', C: '#f59e0b', P: '#10b981' };
    this.CELL_SIZE = 48;
    this.GRID_SIZE = 8;
    this.PAD = 4;

    this.init();
  }

  /* ── Initialization ── */
  async init() {
    this.cacheElements();
    this.bindEvents();
    this.setupCharts();
    this.drawLattice(null);
    
    await this.checkHealth();
    this.setLoading(false);
    this.log('Laboratory Protocol Initialized', 'ok');
    if (this.SESSION_ID) this.log(`Active Session: ${this.SESSION_ID}`, 'ts');

    // Health telemetry polling
    setInterval(() => this.checkHealth(), 30000);
  }

  cacheElements() {
    this.els = {
      loading: document.getElementById('loading-overlay'),
      statusDot: document.getElementById('status-dot'),
      statusText: document.getElementById('status-text'),
      apiBase: document.getElementById('api-base'),
      toast: document.getElementById('toast'),
      logBox: document.getElementById('log-box'),
      canvas: document.getElementById('lattice-canvas'),
      ctx: document.getElementById('lattice-canvas').getContext('2d'),
      
      // KPIs
      kpi: {
        step: document.getElementById('kpi-step'),
        reward: document.getElementById('kpi-reward'),
        best: document.getElementById('kpi-best'),
        phase: document.getElementById('kpi-phase'),
        cost: document.getElementById('kpi-cost'),
        atoms: document.getElementById('kpi-atoms'),
        atomCount: document.getElementById('lat-atom-count'),
        episodeId: document.getElementById('episode-id')
      },
      
      // Controls
      difficulty: document.getElementById('sel-difficulty'),
      scenario: document.getElementById('sel-scenario'),
      rowInput: document.getElementById('inp-row'),
      colInput: document.getElementById('inp-col'),
      rowVal: document.getElementById('row-val'),
      colVal: document.getElementById('col-val'),
      btnReset: document.getElementById('btn-reset'),
      btnStep: document.getElementById('btn-step'),
      atomSection: document.getElementById('atom-section')
    };

    // Resize canvas
    this.els.canvas.width = this.GRID_SIZE * this.CELL_SIZE + this.PAD * 2;
    this.els.canvas.height = this.GRID_SIZE * this.CELL_SIZE + this.PAD * 2;
  }

  bindEvents() {
    this.els.btnReset.onclick = () => this.doReset();
    this.els.btnStep.onclick = () => this.doStep();
    
    // Canvas interaction
    this.els.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
    this.els.canvas.addEventListener('mousemove', (e) => this.handleCanvasHover(e));
    this.els.canvas.addEventListener('mouseleave', () => {
      this.state.hoverPos = { r: -1, c: -1 };
      this.drawLattice(this.state.currentObs ? this.state.currentObs.grid : null);
    });

    // Keyboard
    document.addEventListener('keydown', (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === 'Enter') this.doStep();
      if (e.key.toLowerCase() === 'r') this.doReset();
    });

    // Coordination
    this.els.rowInput.oninput = (e) => this.els.rowVal.textContent = e.target.value;
    this.els.colInput.oninput = (e) => this.els.colVal.textContent = e.target.value;
  }

  /* ── API Interaction ── */
  async apiCall(path, body = null) {
    const opts = {
      method: body ? 'POST' : 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    };
    if (body) opts.body = JSON.stringify(body);
    if (this.SESSION_ID) opts.headers['X-Session-ID'] = this.SESSION_ID;

    const res = await fetch(this.API_BASE + path, opts);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt.slice(0, 200)}`);
    }
    return res.json();
  }

  async checkHealth() {
    try {
      this.els.apiBase.textContent = window.location.host;
      const h = await fetch(this.API_BASE + '/health').then(r => r.json());
      this.els.statusDot.classList.remove('offline');
      this.els.statusText.textContent = (h.status === 'ok' || h.status === 'healthy') ? 'API Online' : 'Warning';
      return true;
    } catch {
      this.els.statusDot.classList.add('offline');
      this.els.statusText.textContent = 'API Offline';
      return false;
    }
  }

  /* ── Business Logic ── */
  async doReset() {
    this.setLoading(true);
    try {
      const payload = { 
        difficulty: this.els.difficulty.value,
        scenario_name: this.els.scenario.value || undefined
      };
      const result = await this.apiCall('/reset', payload);
      
      const obs = result.observation ?? result;
      if (result.session_id) this.SESSION_ID = result.session_id;

      this.state.rewardHistory = [];
      this.state.bestReward = 0;
      this.resetCharts();
      this.updateUI(obs, 0);
      
      this.log(`Protocol Reset → Diff: ${payload.difficulty} | Scenario: ${payload.scenario_name || 'Random'}`, 'ok');
      this.toast('Laboratory environment re-initialized', 'success');
    } catch (e) {
      this.log(`Initialization Failure: ${e.message}`, 'err');
      this.toast(`Reset failed: ${e.message}`, 'error');
    } finally {
      this.setLoading(false);
    }
  }

  async doStep() {
    if (!this.state.currentObs) return this.toast('Reset the environment first', 'error');
    
    this.els.btnStep.disabled = true;
    const row = parseInt(this.els.rowInput.value);
    const col = parseInt(this.els.colInput.value);
    
    const action = {
      action_type: this.state.selectedAction,
      row, col,
      atom: this.state.selectedAction !== 'remove' ? this.state.selectedAtom : undefined
    };

    try {
      const result = await this.apiCall('/step', { action });
      const obs    = result.observation ?? result;
      const reward = result.reward ?? obs.reward ?? 0;
      const done   = result.done   ?? obs.done   ?? false;

      this.state.rewardHistory.push(reward);
      this.state.bestReward = Math.max(this.state.bestReward, reward);
      
      this.updateCharts(reward);
      this.updateUI(obs, reward);

      const actionStr = `${action.action_type.toUpperCase()} [${row},${col}] ${action.atom ? 'Species: '+action.atom : ''}`;
      this.log(`Step ${obs.step_number}: ${actionStr} → Rwd: ${reward.toFixed(4)}${done ? ' [COMPLETE]' : ''}`, done ? 'ok' : '');

      if (done) this.toast(`Episode complete. Optimal reward: ${this.state.bestReward.toFixed(4)}`, 'success');
    } catch (e) {
      this.log(`Step error: ${e.message}`, 'err');
      this.toast(`Step failed: ${e.message}`, 'error');
    } finally {
      this.els.btnStep.disabled = false;
    }
  }

  /* ── UI Rendering ── */
  updateUI(obs, reward) {
    this.state.currentObs = obs;
    this.drawLattice(obs.grid);

    // KPIs
    this.els.kpi.step.textContent = `${obs.step_number} / ${obs.max_steps}`;
    this.els.kpi.reward.textContent = (reward ?? 0).toFixed(4);
    this.els.kpi.best.textContent = this.state.bestReward.toFixed(4);
    this.els.kpi.cost.textContent = `${Math.round(obs.total_cost)} / ${Math.round(obs.cost_budget)}`;

    const atomCount = obs.grid.flat().filter(c => c !== '.').length;
    this.els.kpi.atoms.textContent = atomCount;
    this.els.kpi.atomCount.textContent = `${atomCount} atoms`;

    const phase = obs.phase || 'amorphous';
    this.els.kpi.phase.textContent = phase;
    this.els.kpi.phase.className = 'kpi-value phase-' + phase;

    // Property bars
    ['hardness', 'conductivity', 'thermal_resistance', 'elasticity'].forEach(p => {
      const curr = obs.current_properties[p] ?? 0;
      const tgt  = obs.target[p] ?? 0;
      document.getElementById(`pv-${p}`).textContent = `${curr.toFixed(0)} / ${tgt.toFixed(0)}`;
      document.getElementById(`pb-${p}`).style.width = Math.min(curr, 100) + '%';
      document.getElementById(`pt-${p}`).style.left  = Math.min(tgt, 100) + '%';
    });

    const sb = obs.score_breakdown || {};
    document.getElementById('sc-stability').textContent = (sb.stability ?? 0).toFixed(3);
    document.getElementById('sc-quality').textContent   = (sb.lattice_quality ?? 0).toFixed(3);
    document.getElementById('sb-stability').style.width = ((sb.stability ?? 0) * 100) + '%';
    document.getElementById('sb-quality').style.width   = ((sb.lattice_quality ?? 0) * 100) + '%';

    if (obs.metadata) {
      this.els.kpi.episodeId.textContent = `scenario: ${obs.metadata.scenario_name || 'random'} · diff: ${obs.metadata.difficulty || '—'}`;
    }
  }

  drawLattice(grid) {
    const { ctx, canvas } = this.els;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#080c14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < this.GRID_SIZE; r++) {
      for (let c = 0; c < this.GRID_SIZE; c++) {
        const cell = (grid && grid[r] && grid[r][c]) || '.';
        const x = this.PAD + c * this.CELL_SIZE;
        const y = this.PAD + r * this.CELL_SIZE;
        const color = this.ATOM_COLORS[cell] || this.ATOM_COLORS['.'];

        ctx.fillStyle = color;
        ctx.globalAlpha = cell === '.' ? 0.25 : 0.9;
        this.roundRect(ctx, x + 1, y + 1, this.CELL_SIZE - 2, this.CELL_SIZE - 2, 6);
        ctx.fill();
        ctx.globalAlpha = 1;

        if (cell !== '.') {
          ctx.shadowBlur = 14;
          ctx.shadowColor = this.ATOM_GLOW[cell];
          ctx.font = 'bold 16px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = '#fff';
          ctx.fillText(cell, x + this.CELL_SIZE / 2, y + this.CELL_SIZE / 2);
          ctx.shadowBlur = 0;
        }

        ctx.strokeStyle = 'rgba(30,45,69,0.5)';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x + 0.5, y + 0.5, this.CELL_SIZE - 1, this.CELL_SIZE - 1);
      }
    }
    this.drawHover();
  }

  drawHover() {
    const { r, c } = this.state.hoverPos;
    if (r < 0 || c < 0) return;
    const x = this.PAD + c * this.CELL_SIZE;
    const y = this.PAD + r * this.CELL_SIZE;
    this.els.ctx.strokeStyle = 'rgba(0,212,170,0.8)';
    this.els.ctx.lineWidth = 2;
    this.roundRect(this.els.ctx, x + 2, y + 2, this.CELL_SIZE - 4, this.CELL_SIZE - 4, 5);
    this.els.ctx.stroke();
  }

  /* ── Interaction Handlers ── */
  handleCanvasClick(e) {
    const coords = this.getCanvasCoords(e);
    if (coords) {
      this.els.rowInput.value = coords.r;
      this.els.colInput.value = coords.c;
      this.els.rowVal.textContent = coords.r;
      this.els.colVal.textContent = coords.c;
      this.drawLattice(this.state.currentObs ? this.state.currentObs.grid : null);
    }
  }

  handleCanvasHover(e) {
    const coords = this.getCanvasCoords(e);
    if (coords && (coords.r !== this.state.hoverPos.r || coords.c !== this.state.hoverPos.c)) {
      this.state.hoverPos = coords;
      this.drawLattice(this.state.currentObs ? this.state.currentObs.grid : null);
    }
  }

  getCanvasCoords(e) {
    const rect = this.els.canvas.getBoundingClientRect();
    const scaleX = this.els.canvas.width / rect.width;
    const scaleY = this.els.canvas.height / rect.height;
    const col = Math.floor(((e.clientX - rect.left) * scaleX - this.PAD) / this.CELL_SIZE);
    const row = Math.floor(((e.clientY - rect.top) * scaleY - this.PAD) / this.CELL_SIZE);
    return (row >= 0 && row < 8 && col >= 0 && col < 8) ? { r: row, c: col } : null;
  }

  /* ── Charts ── */
  setupCharts() {
    const rewardChartCtx = document.getElementById('reward-chart').getContext('2d');
    this.rewardChart = new Chart(rewardChartCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          data: [],
          borderColor: '#00d4aa',
          backgroundColor: 'rgba(0,212,170,0.08)',
          fill: true,
          tension: 0.35,
          pointRadius: 3,
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(30,45,69,0.3)' }, ticks: { font: { size: 10 } } },
          y: { min: -0.05, max: 1.05, grid: { color: 'rgba(30,45,69,0.3)' }, ticks: { font: { size: 10 } } }
        }
      }
    });
  }

  updateCharts(reward) {
    this.rewardChart.data.labels.push(this.state.rewardHistory.length);
    this.rewardChart.data.datasets[0].data.push(reward);
    this.rewardChart.update();
  }

  resetCharts() {
    this.rewardChart.data.labels = [];
    this.rewardChart.data.datasets[0].data = [];
    this.rewardChart.update();
  }

  /* ── Utilities ── */
  setLoading(on) {
    this.els.loading.classList.toggle('hidden', !on);
    this.els.btnReset.disabled = on;
    this.els.btnStep.disabled = on;
  }

  log(msg, type = '') {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-ts">${new Date().toLocaleTimeString()}</span><span class="log-${type}">${msg}</span>`;
    this.els.logBox.appendChild(entry);
    this.els.logBox.scrollTop = this.els.logBox.scrollHeight;
  }

  toast(msg, type = '') {
    this.els.toast.textContent = msg;
    this.els.toast.className = `toast show ${type}`;
    clearTimeout(this.toastTimeout);
    this.toastTimeout = setTimeout(() => this.els.toast.className = 'toast', 3000);
  }

  roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  selectAction(action) {
    this.state.selectedAction = action;
    document.querySelectorAll('.action-tab').forEach(b => b.classList.toggle('active', b.dataset.action === action));
    this.els.atomSection.style.opacity = action === 'remove' ? '0.4' : '1';
  }

  selectAtom(atom) {
    this.state.selectedAtom = atom;
    document.querySelectorAll('.atom-btn').forEach(b => b.classList.toggle('selected', b.dataset.atom === atom));
  }
}

// Analytics Logic
async function loadBenchmarks() {
  const status = document.getElementById('benchmark-status');
  try {
    const res = await fetch('/playground/benchmarks.json');
    if (!res.ok) throw new Error('Benchmark data not found. Please run agent_benchmark.py');
    const data = await res.json();
    
    status.textContent = `Successfully loaded ${data.length} discovery episodes. Updating visualization...`;
    
    // 1. Phase Distribution (Pie)
    const phases = data.reduce((acc, run) => {
      acc[run.phase] = (acc[run.phase] || 0) + 1;
      return acc;
    }, {});
    
    if (window.phaseChart) window.phaseChart.destroy();
    window.phaseChart = new Chart(document.getElementById('phase-pie-chart'), {
      type: 'pie',
      data: {
        labels: Object.keys(phases),
        datasets: [{
          data: Object.values(phases),
          backgroundColor: ['#00d4aa', '#3b82f6', '#f59e0b', '#8b5cf6'],
          borderWidth: 0
        }]
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', font: { family: 'Inter' } } } } }
    });

    // 2. Scenario Performance (Bar)
    if (window.barChart) window.barChart.destroy();
    window.barChart = new Chart(document.getElementById('scenario-bar-chart'), {
      type: 'bar',
      data: {
        labels: data.map(r => `${r.scenario} (${r.difficulty})`),
        datasets: [
          { label: 'Optimal Reward', data: data.map(r => r.best_reward), backgroundColor: '#00d4aa' },
          { label: 'Cost %', data: data.map(r => r.total_cost / 1.2), backgroundColor: 'rgba(59, 130, 246, 0.4)' }
        ]
      },
      options: { 
        responsive: true, 
        maintainAspectRatio: false, 
        scales: { 
          y: { grid: { color: 'rgba(30,45,69,0.3)' }, ticks: { color: '#64748b' } },
          x: { grid: { display : false }, ticks: { color: '#64748b', font: { size: 10 } } }
        },
        plugins: { legend: { labels: { color: '#94a3b8' } } }
      }
    });

  } catch (e) {
    status.textContent = `Notice: ${e.message}`;
    status.style.color = '#ef4444';
  }
}

// Instantiate app
window.MF_APP = new MaterialForgeApp();
setTimeout(loadBenchmarks, 1000); // Initial load attempt

// Link external clicks for the instantiated class
function selectAction(btn) { window.MF_APP.selectAction(btn.dataset.action); }
function selectAtom(btn) { window.MF_APP.selectAtom(btn.dataset.atom); }
