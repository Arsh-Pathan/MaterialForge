"""Custom Gradio frontend for MaterialForge — Crystal Structure Visualizer.

Provides an interactive dashboard with:
- 8x8 lattice grid heatmap
- Property comparison radar chart (target vs current)
- Reward history line chart
- Phase/cost/step KPI indicators
"""

import json
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# Color scheme — scientific dark theme
ATOM_COLORS = {
    "A": "#e74c3c",  # Metal — red
    "B": "#3498db",  # Conductor — blue
    "C": "#f39c12",  # Ceramic — amber
    "P": "#2ecc71",  # Polymer — green
    ".": "#1a1a2e",  # Empty — dark
}
ATOM_LABELS = {
    "A": "Metal (A)",
    "B": "Conductor (B)",
    "C": "Ceramic (C)",
    "P": "Polymer (P)",
    ".": "Empty",
}
BG_COLOR = "#0f0f1a"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"
ACCENT = "#00d4aa"


def render_lattice(grid):
    """Render the 8x8 lattice as a colored grid with atom labels."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    size = len(grid)
    data = np.zeros((size, size, 4))

    for r in range(size):
        for c in range(size):
            cell = grid[r][c] if r < len(grid) and c < len(grid[r]) else "."
            rgba = mcolors.to_rgba(ATOM_COLORS.get(cell, ATOM_COLORS["."]))
            data[r, c] = rgba

    ax.imshow(data, interpolation="nearest", aspect="equal")

    # Draw atom symbols
    for r in range(size):
        for c in range(size):
            cell = grid[r][c] if r < len(grid) and c < len(grid[r]) else "."
            if cell != ".":
                ax.text(c, r, cell, ha="center", va="center",
                        fontsize=13, fontweight="bold", color="white",
                        fontfamily="monospace")

    # Grid lines
    for i in range(size + 1):
        ax.axhline(i - 0.5, color=GRID_COLOR, linewidth=0.5, alpha=0.6)
        ax.axvline(i - 0.5, color=GRID_COLOR, linewidth=0.5, alpha=0.6)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(size), color=TEXT_COLOR, fontsize=8)
    ax.set_yticklabels(range(size), color=TEXT_COLOR, fontsize=8)
    ax.tick_params(length=0)
    ax.set_title("Crystal Lattice Structure", color=ACCENT, fontsize=12,
                  fontweight="bold", pad=10)

    # Legend
    patches = [mpatches.Patch(color=ATOM_COLORS[k], label=ATOM_LABELS[k])
               for k in ["A", "B", "C", "P"]]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=8, facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)

    plt.tight_layout()
    return fig


def render_properties(target, current):
    """Render target vs current properties as a grouped bar chart."""
    props = ["hardness", "conductivity", "thermal_resistance", "elasticity"]
    labels = ["Hardness", "Conductivity", "Thermal\nResistance", "Elasticity"]

    t_vals = [target.get(p, 0) for p in props]
    c_vals = [current.get(p, 0) for p in props]

    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = np.arange(len(labels))
    width = 0.35

    bars_t = ax.bar(x - width / 2, t_vals, width, label="Target",
                     color="#ff6b6b", alpha=0.85, edgecolor="none")
    bars_c = ax.bar(x + width / 2, c_vals, width, label="Current",
                     color=ACCENT, alpha=0.85, edgecolor="none")

    ax.set_ylabel("Value (0-100)", color=TEXT_COLOR, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_COLOR, fontsize=8)
    ax.set_ylim(0, 105)
    ax.tick_params(colors=TEXT_COLOR, length=0)
    ax.spines[:].set_visible(False)
    ax.legend(fontsize=8, facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)
    ax.set_title("Target vs Current Properties", color=ACCENT, fontsize=11,
                  fontweight="bold", pad=8)

    # Value labels on bars
    for bar in bars_t:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", color="#ff6b6b", fontsize=7)
    for bar in bars_c:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}",
                    ha="center", va="bottom", color=ACCENT, fontsize=7)

    ax.grid(axis="y", color=GRID_COLOR, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    return fig


def render_reward_history(rewards):
    """Render reward history as a line chart with area fill."""
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    if not rewards:
        ax.text(0.5, 0.5, "No steps yet", ha="center", va="center",
                color=TEXT_COLOR, fontsize=12, transform=ax.transAxes)
        plt.tight_layout()
        return fig

    steps = list(range(1, len(rewards) + 1))
    ax.fill_between(steps, rewards, alpha=0.2, color=ACCENT)
    ax.plot(steps, rewards, color=ACCENT, linewidth=2, marker="o",
            markersize=3, markerfacecolor="white", markeredgecolor=ACCENT)

    # Best reward marker
    if rewards:
        best_idx = int(np.argmax(rewards))
        ax.annotate(f"Best: {rewards[best_idx]:.3f}",
                    xy=(steps[best_idx], rewards[best_idx]),
                    xytext=(10, 10), textcoords="offset points",
                    color="#ff6b6b", fontsize=8, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1))

    ax.set_xlabel("Step", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Reward", color=TEXT_COLOR, fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(colors=TEXT_COLOR, length=0)
    ax.spines[:].set_visible(False)
    ax.grid(color=GRID_COLOR, alpha=0.3, linewidth=0.5)
    ax.set_title("Reward History", color=ACCENT, fontsize=11,
                  fontweight="bold", pad=8)

    plt.tight_layout()
    return fig


def render_score_breakdown(breakdown, phase, cost, budget):
    """Render score components as horizontal bar chart + phase/cost info."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.8), facecolor=BG_COLOR,
                                     gridspec_kw={"width_ratios": [3, 2]})

    # Score breakdown bars
    ax1.set_facecolor(BG_COLOR)
    components = {
        "Stability": breakdown.get("stability", 0),
        "Lattice Quality": breakdown.get("lattice_quality", 0),
    }
    names = list(components.keys())
    vals = list(components.values())
    colors_bar = [ACCENT, "#3498db"]

    bars = ax1.barh(names, vals, color=colors_bar, alpha=0.85, height=0.5)
    ax1.set_xlim(0, 1.05)
    ax1.tick_params(colors=TEXT_COLOR, length=0)
    ax1.spines[:].set_visible(False)
    ax1.set_title("Score Components", color=ACCENT, fontsize=10,
                   fontweight="bold")

    for bar, val in zip(bars, vals):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", color=TEXT_COLOR, fontsize=9)
    ax1.set_yticklabels(names, color=TEXT_COLOR, fontsize=9)

    # Phase + Cost gauge
    ax2.set_facecolor(BG_COLOR)
    ax2.axis("off")

    phase_colors = {
        "crystalline": "#2ecc71",
        "polycrystalline": "#f39c12",
        "amorphous": "#e74c3c",
    }
    pc = phase_colors.get(phase, TEXT_COLOR)

    ax2.text(0.5, 0.85, "Phase", ha="center", va="center",
             color=TEXT_COLOR, fontsize=9, transform=ax2.transAxes)
    ax2.text(0.5, 0.65, phase.upper(), ha="center", va="center",
             color=pc, fontsize=11, fontweight="bold", transform=ax2.transAxes)

    cost_pct = min(cost / max(budget, 1), 1.5)
    cost_color = ACCENT if cost <= budget else "#e74c3c"
    ax2.text(0.5, 0.40, "Cost Budget", ha="center", va="center",
             color=TEXT_COLOR, fontsize=9, transform=ax2.transAxes)
    ax2.text(0.5, 0.20, f"{cost:.0f} / {budget:.0f}", ha="center",
             va="center", color=cost_color, fontsize=11, fontweight="bold",
             transform=ax2.transAxes)

    # Cost bar
    bar_y = 0.05
    ax2.barh([bar_y], [1.0], height=0.06, color=GRID_COLOR,
             transform=ax2.transAxes, left=0.1)
    ax2.barh([bar_y], [min(cost_pct, 1.0) * 0.8], height=0.06,
             color=cost_color, alpha=0.8, transform=ax2.transAxes, left=0.1)

    plt.tight_layout()
    return fig


def build_gradio_frontend(web_manager, action_fields, metadata,
                          is_chat_env, title, quick_start_md):
    """Build the custom MaterialForge visualization dashboard."""

    # State tracking
    reward_history = gr.State([])
    last_obs = gr.State(None)

    with gr.Blocks() as blocks:

        gr.Markdown(
            "## MaterialForge — Crystal Structure Design Lab\n"
            "*Design atomic structures to match target material properties*"
        )

        # --- KPI Strip ---
        with gr.Row():
            kpi_step = gr.Markdown(
                "<div class='kpi-box'><div class='kpi-value'>0 / 50</div>"
                "<div class='kpi-label'>Step</div></div>"
            )
            kpi_reward = gr.Markdown(
                "<div class='kpi-box'><div class='kpi-value'>0.000</div>"
                "<div class='kpi-label'>Reward</div></div>"
            )
            kpi_phase = gr.Markdown(
                "<div class='kpi-box'><div class='kpi-value'>—</div>"
                "<div class='kpi-label'>Phase</div></div>"
            )
            kpi_cost = gr.Markdown(
                "<div class='kpi-box'><div class='kpi-value'>0 / 80</div>"
                "<div class='kpi-label'>Cost</div></div>"
            )
            kpi_best = gr.Markdown(
                "<div class='kpi-box'><div class='kpi-value'>0.000</div>"
                "<div class='kpi-label'>Best Reward</div></div>"
            )

        # --- Charts Row 1: Lattice + Properties ---
        with gr.Row():
            lattice_plot = gr.Plot(label="Crystal Lattice")
            props_plot = gr.Plot(label="Properties")

        # --- Charts Row 2: Rewards + Score Breakdown ---
        with gr.Row():
            reward_plot = gr.Plot(label="Reward History")
            score_plot = gr.Plot(label="Score Breakdown")

        # --- Controls ---
        with gr.Row():
            with gr.Column(scale=1):
                difficulty = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="medium", label="Difficulty"
                )
                scenario = gr.Dropdown(
                    choices=["random", "diamond-like", "conductor",
                             "heat-shield", "flexible-polymer", "balanced-alloy"],
                    value="random", label="Scenario"
                )
                reset_btn = gr.Button("Reset Environment", variant="secondary",
                                       size="lg")

            with gr.Column(scale=2):
                with gr.Row():
                    action_type = gr.Dropdown(
                        choices=["place", "replace", "remove"],
                        value="place", label="Action"
                    )
                    atom = gr.Dropdown(
                        choices=["A (Metal)", "B (Conductor)",
                                 "C (Ceramic)", "P (Polymer)"],
                        value="A (Metal)", label="Atom"
                    )
                with gr.Row():
                    row_input = gr.Slider(0, 7, value=0, step=1, label="Row")
                    col_input = gr.Slider(0, 7, value=0, step=1, label="Col")
                step_btn = gr.Button("Step", variant="primary", size="lg")

        status_box = gr.Markdown("")

        # --- Event handlers ---
        def do_reset(diff, scen, rh):
            try:
                kwargs = {"difficulty": diff}
                if scen != "random":
                    kwargs["scenario_name"] = scen
                result = web_manager.reset_environment(**kwargs)
                obs_data = result if isinstance(result, dict) else (
                    result.model_dump() if hasattr(result, "model_dump") else {}
                )
                obs = obs_data.get("observation", obs_data)
                rh = []  # reset reward history

                grid = obs.get("grid", [["." for _ in range(8)] for _ in range(8)])
                target = obs.get("target", {})
                current = obs.get("current_properties", {})
                phase = obs.get("phase", "amorphous")
                cost = obs.get("total_cost", 0)
                budget = obs.get("cost_budget", 80)
                step_n = obs.get("step_number", 0)
                max_s = obs.get("max_steps", 50)

                return (
                    render_lattice(grid),
                    render_properties(target, current),
                    render_reward_history([]),
                    render_score_breakdown(obs.get("score_breakdown", {}),
                                           phase, cost, budget),
                    f"<div class='kpi-box'><div class='kpi-value'>{step_n} / {max_s}</div><div class='kpi-label'>Step</div></div>",
                    f"<div class='kpi-box'><div class='kpi-value'>0.000</div><div class='kpi-label'>Reward</div></div>",
                    f"<div class='kpi-box'><div class='kpi-value'>{phase.upper()}</div><div class='kpi-label'>Phase</div></div>",
                    f"<div class='kpi-box'><div class='kpi-value'>{cost:.0f} / {budget:.0f}</div><div class='kpi-label'>Cost</div></div>",
                    f"<div class='kpi-box'><div class='kpi-value'>0.000</div><div class='kpi-label'>Best Reward</div></div>",
                    rh,
                    obs,
                    f"**Reset** — {diff} | {scen} | Target: {target}",
                )
            except Exception as e:
                return (None, None, None, None,
                        gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), rh, None, f"**Error:** {e}")

        def do_step(atype, atom_sel, row, col, rh, obs_state):
            try:
                atom_char = atom_sel[0] if atom_sel else "A"
                action_data = {
                    "action_type": atype,
                    "row": int(row),
                    "col": int(col),
                }
                if atype != "remove":
                    action_data["atom"] = atom_char

                result = web_manager.step_environment(action_data)
                res_data = result if isinstance(result, dict) else (
                    result.model_dump() if hasattr(result, "model_dump") else {}
                )
                obs = res_data.get("observation", res_data)
                reward = res_data.get("reward", obs.get("reward", 0)) or 0
                done = res_data.get("done", obs.get("done", False))

                rh = list(rh) + [float(reward)]

                grid = obs.get("grid", [["." for _ in range(8)] for _ in range(8)])
                target = obs.get("target", {})
                current = obs.get("current_properties", {})
                phase = obs.get("phase", "amorphous")
                cost = obs.get("total_cost", 0)
                budget = obs.get("cost_budget", 80)
                step_n = obs.get("step_number", 0)
                max_s = obs.get("max_steps", 50)
                breakdown = obs.get("score_breakdown", {})
                best_r = max(rh) if rh else 0

                done_str = " | **DONE**" if done else ""
                action_str = f"{atype}({int(row)},{int(col)},{atom_char})"

                return (
                    render_lattice(grid),
                    render_properties(target, current),
                    render_reward_history(rh),
                    render_score_breakdown(breakdown, phase, cost, budget),
                    f"<div class='kpi-box'><div class='kpi-value'>{step_n} / {max_s}</div><div class='kpi-label'>Step</div></div>",
                    f"<div class='kpi-box'><div class='kpi-value'>{reward:.3f}</div><div class='kpi-label'>Reward</div></div>",
                    f"<div class='kpi-box'><div class='kpi-value'>{phase.upper()}</div><div class='kpi-label'>Phase</div></div>",
                    f"<div class='kpi-box'><div class='kpi-value'>{cost:.0f} / {budget:.0f}</div><div class='kpi-label'>Cost</div></div>",
                    f"<div class='kpi-box'><div class='kpi-value'>{best_r:.3f}</div><div class='kpi-label'>Best Reward</div></div>",
                    rh,
                    obs,
                    f"**Step {step_n}** — {action_str} → reward={reward:.3f}{done_str}",
                )
            except Exception as e:
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), rh, obs_state, f"**Error:** {e}")

        outputs = [
            lattice_plot, props_plot, reward_plot, score_plot,
            kpi_step, kpi_reward, kpi_phase, kpi_cost, kpi_best,
            reward_history, last_obs, status_box,
        ]

        reset_btn.click(
            fn=do_reset,
            inputs=[difficulty, scenario, reward_history],
            outputs=outputs,
        )
        step_btn.click(
            fn=do_step,
            inputs=[action_type, atom, row_input, col_input,
                    reward_history, last_obs],
            outputs=outputs,
        )

    return blocks
