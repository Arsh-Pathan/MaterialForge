"""Create charts from test results."""

import json

# Load data
with open("plot_data.json") as f:
    data = json.load(f)

# Try to use matplotlib, fallback to text if not available
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if HAS_MATPLOTLIB:
    # 1. Bar chart - Max Reward by Task
    plt.figure(figsize=(10, 6))
    tasks = data["tasks"]
    rewards = [
        sum(data["task_max_rewards"][t]) / len(data["task_max_rewards"][t])
        for t in tasks
    ]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    bars = plt.bar(tasks, rewards, color=colors)
    plt.ylabel("Average Max Reward")
    plt.title("Performance by Task")
    plt.ylim(0, 1)
    for bar, r in zip(bars, rewards):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{r:.3f}",
            ha="center",
        )
    plt.tight_layout()
    plt.savefig("chart_reward_by_task.png", dpi=150)
    print("Created chart_reward_by_task.png")
    plt.close()

    # 2. Line chart - Reward Trajectory
    plt.figure(figsize=(12, 6))
    plt.plot(data["reward_trajectory"], linewidth=2, color="#2196F3")
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Reward Trajectory Over Time")
    plt.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% threshold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chart_reward_trajectory.png", dpi=150)
    print("Created chart_reward_trajectory.png")
    plt.close()

    # 3. Bar chart - Property Gaps
    plt.figure(figsize=(10, 6))
    props = list(data["property_gaps"].keys())
    gaps = list(data["property_gaps"].values())
    colors = ["#f44336" if g > 10 else "#ff9800" if g > 5 else "#4CAF50" for g in gaps]
    plt.bar(props, gaps, color=colors)
    plt.ylabel("Average Gap")
    plt.title("Property Gaps (lower is better)")
    plt.axhline(y=10, color="r", linestyle="--", alpha=0.5)
    plt.axhline(y=5, color="orange", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("chart_property_gaps.png", dpi=150)
    print("Created chart_property_gaps.png")
    plt.close()

    # 4. Pie chart - Atom Usage
    plt.figure(figsize=(8, 8))
    atoms = list(data["atom_usage"].keys())
    counts = list(data["atom_usage"].values())
    labels = [f"{a}\n({c})" for a, c in zip(atoms, counts)]
    colors = ["#607D8B", "#795548", "#9C27B0"]
    plt.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title("Atom Usage Distribution")
    plt.tight_layout()
    plt.savefig("chart_atom_usage.png", dpi=150)
    print("Created chart_atom_usage.png")
    plt.close()

    # 5. Grouped bar - Task comparison
    plt.figure(figsize=(12, 6))
    x = range(len(tasks))
    max_rewards = [
        sum(data["task_max_rewards"][t]) / len(data["task_max_rewards"][t])
        for t in tasks
    ]
    avg_gaps = [
        sum(data["task_avg_gaps"][t]) / len(data["task_avg_gaps"][t]) for t in tasks
    ]
    width = 0.35
    plt.bar(
        [i - width / 2 for i in x],
        max_rewards,
        width,
        label="Max Reward",
        color="#2196F3",
    )
    plt.bar(
        [i + width / 2 for i in x],
        [g / 100 for g in avg_gaps],
        width,
        label="Avg Gap (scaled)",
        color="#FF9800",
    )
    plt.xlabel("Task")
    plt.ylabel("Score")
    plt.title("Task Performance Comparison")
    plt.xticks(x, tasks)
    plt.legend()
    plt.tight_layout()
    plt.savefig("chart_task_comparison.png", dpi=150)
    print("Created chart_task_comparison.png")
    plt.close()

    print("\nAll charts created successfully!")
else:
    # Create text-based visualization
    print("\n" + "=" * 60)
    print("TEXT-BASED VISUALIZATION")
    print("=" * 60)

    print("\n📊 MAX REWARD BY TASK")
    print("-" * 40)
    for task in data["tasks"]:
        avg = sum(data["task_max_rewards"][task]) / len(data["task_max_rewards"][task])
        bar = "█" * int(avg * 20)
        print(f"{task:<15} {avg:.3f} {bar}")

    print("\n📐 PROPERTY GAPS (lower is better)")
    print("-" * 40)
    for prop, gap in data["property_gaps"].items():
        status = "✓" if gap < 8 else "⚠" if gap < 12 else "✗"
        bar = "▓" * int(gap / 2)
        print(f"{prop:<20} {gap:>5.1f} {bar} {status}")

    print("\n🔬 ATOM USAGE")
    print("-" * 40)
    total = sum(data["atom_usage"].values())
    for atom, count in data["atom_usage"].items():
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        print(f"{atom:<10} {count:>3} ({pct:>5.1f}%) {bar}")

    print("\n📈 REWARD TRAJECTORY")
    print("-" * 40)
    traj = data["reward_trajectory"]
    for i in range(0, len(traj), 5):
        vals = traj[i : min(i + 5, len(traj))]
        avg = sum(vals) / len(vals)
        bar = "█" * int(avg * 20)
        print(f"Step {i + 1:>2}-{i + 5:<2}: {avg:.3f} {bar}")
