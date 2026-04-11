/**
 * TelemetryUI Module
 * Synchronizes scientific metrics with the dashboard.
 */
export class TelemetryUI {
  constructor() {
    this.els = {
      kpi: {
        step: document.getElementById('kpi-step'),
        reward: document.getElementById('kpi-reward'),
        best: document.getElementById('kpi-best'),
        orderIndex: document.getElementById('kpi-order'),
        atoms: document.getElementById('kpi-atoms'),
        cost: document.getElementById('kpi-cost'),
        episodeId: document.getElementById('episode-id')
      },
      log: document.getElementById('log-box'),
      statusText: document.getElementById('status-text'),
      statusDot: document.getElementById('status-dot')
    };
  }

  updateKPIs(obs, reward, bestReward) {
    const { kpi } = this.els;
    if (!kpi.step) return;

    kpi.step.textContent = `${obs.step_number} / ${obs.max_steps}`;
    kpi.reward.textContent = (reward ?? 0).toFixed(4);
    kpi.best.textContent = bestReward.toFixed(4);
    kpi.cost.textContent = `${Math.round(obs.total_cost)} / ${Math.round(obs.cost_budget)}`;
    kpi.atoms.textContent = obs.grid.flat().filter(c => c !== '.').length;
    
    if (kpi.orderIndex) {
        const orderIdx = obs.score_breakdown?.lattice_order_index ?? 0;
        kpi.orderIndex.textContent = orderIdx.toFixed(3);
    }
    
    if (obs.metadata && kpi.episodeId) {
      kpi.episodeId.textContent = `Scenario: ${obs.metadata.scenario_name} | Difficulty: ${obs.metadata.difficulty}`;
    }
  }

  updatePropertyBars(obs) {
    ['hardness', 'conductivity', 'thermal_resistance', 'elasticity'].forEach(p => {
      const valText = document.getElementById(`pv-${p}`);
      const fillBar = document.getElementById(`pb-${p}`);
      const targetMarker = document.getElementById(`pt-${p}`);
      
      if (!valText || !fillBar) return;

      const curr = obs.current_properties[p] ?? 0;
      const tgt  = obs.target[p] ?? 0;
      
      valText.textContent = `${curr.toFixed(0)} / ${tgt.toFixed(0)}`;
      fillBar.style.width = Math.min(curr, 100) + '%';
      if (targetMarker) targetMarker.style.left = Math.min(tgt, 100) + '%';
    });

    const sb = obs.score_breakdown || {};
    const stabText = document.getElementById('sc-stability');
    const orderText = document.getElementById('sc-order');
    if (stabText) stabText.textContent = (sb.structural_stability ?? 0).toFixed(3);
    if (orderText) orderText.textContent = (sb.lattice_order_index ?? 0).toFixed(3);
    
    const stabFill = document.getElementById('sb-stability');
    const orderFill = document.getElementById('sb-order');
    if (stabFill) stabFill.style.width = ((sb.structural_stability ?? 0) * 100) + '%';
    if (orderFill) orderFill.style.width = ((sb.lattice_order_index ?? 0) * 100) + '%';
  }

  setStatus(online) {
    if (this.els.statusText) {
        this.els.statusText.textContent = online ? 'LIVE' : 'OFFLINE';
    }
    if (this.els.statusDot) {
        this.els.statusDot.classList.toggle('offline', !online);
    }
  }

  log(msg, type = 'info') {
    if (!this.els.log) return;
    const entry = document.createElement('div');
    entry.className = `log-line ${type}`; /* changed to match playground.css line 86 */
    const ts = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    entry.innerHTML = `<span style="opacity:0.4">[${ts}]</span> <span>${msg}</span>`;
    this.els.log.prepend(entry);
  }
}
