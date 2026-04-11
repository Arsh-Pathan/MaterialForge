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
        orderIndex: document.getElementById('kpi-quality'),
        atoms: document.getElementById('kpi-atoms'),
        cost: document.getElementById('kpi-cost'),
        episodeId: document.getElementById('episode-id')
      },
      log: document.getElementById('log-box')
    };
  }

  updateKPIs(obs, reward, bestReward) {
    const { kpi } = this.els;
    kpi.step.textContent = `${obs.step_number} / ${obs.max_steps}`;
    kpi.reward.textContent = (reward ?? 0).toFixed(4);
    kpi.best.textContent = bestReward.toFixed(4);
    kpi.cost.textContent = `${Math.round(obs.total_cost)} / ${Math.round(obs.cost_budget)}`;
    kpi.atoms.textContent = obs.grid.flat().filter(c => c !== '.').length;
    
    const orderIdx = obs.score_breakdown?.lattice_order_index ?? 0;
    kpi.orderIndex.textContent = orderIdx.toFixed(3);
    
    if (obs.metadata) {
      kpi.episodeId.textContent = `Scenario: ${obs.metadata.scenario_name} | Difficulty: ${obs.metadata.difficulty}`;
    }
  }

  updatePropertyBars(obs) {
    ['hardness', 'conductivity', 'thermal_resistance', 'elasticity'].forEach(p => {
      const curr = obs.current_properties[p] ?? 0;
      const tgt  = obs.target[p] ?? 0;
      document.getElementById(`pv-${p}`).textContent = `${curr.toFixed(0)} / ${tgt.toFixed(0)}`;
      document.getElementById(`pb-${p}`).style.width = Math.min(curr, 100) + '%';
      document.getElementById(`pt-${p}`).style.left  = Math.min(tgt, 100) + '%';
    });

    const sb = obs.score_breakdown || {};
    document.getElementById('sc-stability').textContent = (sb.structural_stability ?? 0).toFixed(3);
    document.getElementById('sc-quality').textContent   = (sb.lattice_order_index ?? 0).toFixed(3);
    document.getElementById('sb-stability').style.width = ((sb.structural_stability ?? 0) * 100) + '%';
    document.getElementById('sb-quality').style.width   = ((sb.lattice_order_index ?? 0) * 100) + '%';
  }

  log(msg, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    const ts = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    entry.innerHTML = `<span class="log-ts">[${ts}]</span> <span>${msg}</span>`;
    this.els.log.prepend(entry);
  }
}
