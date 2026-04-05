"""Analysis and visualization of test results."""

import json

# Load results
with open("test_results.json") as f:
    results = json.load(f)

# Create summary tables
print("=" * 70)
print("TEST RESULTS ANALYSIS")
print("=" * 70)

# Per-task summary
print("\n📊 PERFORMANCE BY TASK")
print("-" * 50)
print(f"{'Task':<20} {'Max Reward':<15} {'Avg Gap':<12} {'Phase':<15}")
print("-" * 50)

by_task = {}
for r in results:
    t = r["task"]
    if t not in by_task:
        by_task[t] = []
    by_task[t].append(r)

for task, runs in sorted(by_task.items()):
    avg_max = sum(x["max_reward"] for x in runs) / len(runs)
    avg_gap = sum(x["avg_gap"] for x in runs) / len(runs)
    phases = ", ".join(set(x["phase"] for x in runs))
    print(f"{task:<20} {avg_max:<15.3f} {avg_gap:<12.1f} {phases:<15}")

# Overall stats
print("\n📈 OVERALL STATISTICS")
print("-" * 50)
total_runs = len(results)
total_llm_success = sum(r["llm_success"] for r in results)
total_llm_fail = sum(r["llm_fail"] for r in results)
avg_max_all = sum(r["max_reward"] for r in results) / total_runs
avg_gap_all = sum(r["avg_gap"] for r in results) / total_runs

print(f"Total runs: {total_runs}")
print(
    f"LLM success rate: {total_llm_success}/{total_llm_success + total_llm_fail} ({100 * total_llm_success / (total_llm_success + total_llm_fail):.1f}%)"
)
print(f"Average max reward: {avg_max_all:.3f}")
print(f"Average property gap: {avg_gap_all:.1f}")

# Atom usage
print("\n🔬 ATOM USAGE")
print("-" * 50)
from collections import Counter

atom_counts = Counter()
for r in results:
    for atom, count in r["atoms"].items():
        atom_counts[atom] += count

for atom, count in sorted(atom_counts.items()):
    names = {"A": "Metal", "B": "Conductor", "C": "Ceramic", "P": "Polymer"}
    print(f"{atom} ({names.get(atom, atom)}): {count} placements")

# Property gap analysis
print("\n📐 PROPERTY GAP ANALYSIS")
print("-" * 50)
prop_gaps = {
    "hardness": [],
    "conductivity": [],
    "thermal_resistance": [],
    "elasticity": [],
}
for r in results:
    for prop, gap in r["prop_gaps"].items():
        prop_gaps[prop].append(gap)

for prop, gaps in prop_gaps.items():
    avg = sum(gaps) / len(gaps)
    print(f"{prop}: avg gap = {avg:.1f}")

# Reward trajectory
print("\n📉 REWARD TRAJECTORY (avg across runs)")
print("-" * 50)
max_len = max(len(r["rewards"]) for r in results)
avg_by_step = []
for i in range(max_len):
    vals = [r["rewards"][i] for r in results if i < len(r["rewards"])]
    if vals:
        avg_by_step.append(sum(vals) / len(vals))

for i in range(0, len(avg_by_step), 5):
    print(
        f"Steps {i + 1}-{i + 5}: {avg_by_step[i]:.3f} -> {avg_by_step[min(i + 4, len(avg_by_step) - 1)]:.3f}"
    )

print("\n" + "=" * 70)
print("DATA READY FOR PLOTTING")
print("=" * 70)

# Export simplified data for plotting
plot_data = {
    "tasks": list(by_task.keys()),
    "task_max_rewards": {
        t: [r["max_reward"] for r in runs] for t, runs in by_task.items()
    },
    "task_avg_gaps": {t: [r["avg_gap"] for r in runs] for t, runs in by_task.items()},
    "atom_usage": dict(atom_counts),
    "property_gaps": {k: sum(v) / len(v) for k, v in prop_gaps.items()},
    "reward_trajectory": avg_by_step,
}

with open("plot_data.json", "w") as f:
    json.dump(plot_data, f, indent=2)

print("Saved plot_data.json for visualization")
