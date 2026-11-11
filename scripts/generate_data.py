"""Generate synthetic bar, pie, line, and infographic samples.

The generator is intentionally independent of the extraction code: it writes
images and ground-truth JSON files that can be used for training and evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "samples"
DATASET_DIR = ROOT / "data" / "dataset"

PALETTES = [
    ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"],
    ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B"],
    ["#0F4C5C", "#E36414", "#9A031E", "#5F0F40", "#F4A261", "#2A9D8F"],
    ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#6D597A"],
]

BAR_TITLES = [
    ("Quarterly Revenue by Region", "Revenue (USD millions)", ["North", "South", "East", "West", "Central"]),
    ("Units Sold by Product", "Units sold", ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]),
    (
        "Average Wait Time by Department",
        "Minutes",
        ["Emergency", "Radiology", "Cardiology", "Oncology", "Pediatrics"],
    ),
    (
        "Student Enrollment by Programme",
        "Students",
        ["Data Science", "AI", "Cybersecurity", "Software Eng.", "Design"],
    ),
    ("Monthly Energy Use by Campus", "Energy (MWh)", ["Main", "North", "South", "East", "West"]),
    ("Customer Satisfaction by Channel", "Score", ["Phone", "Chat", "Email", "Store", "App"]),
    ("Crop Yield by District", "Yield (tonnes)", ["A", "B", "C", "D", "E"]),
    ("Hiring by Team in 2025", "New hires", ["Research", "Engineering", "Sales", "Support", "Operations"]),
]

PIE_TITLES = [
    ("Browser Market Share", ["Chrome", "Safari", "Edge", "Firefox", "Other"]),
    ("Household Budget Allocation", ["Housing", "Food", "Transport", "Health", "Savings"]),
    ("Global Energy Mix", ["Coal", "Gas", "Hydro", "Nuclear", "Solar", "Wind"]),
    ("Customer Channel Mix", ["Online", "Retail", "Partner", "Direct"]),
    ("Time Spent on Social Media", ["Video", "Messaging", "Feeds", "Search", "Other"]),
    ("Cloud Spending by Service", ["Compute", "Storage", "Database", "Network", "Analytics"]),
]

LINE_TITLES = [
    ("Monthly Active Users", "Users (millions)", ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]),
    ("Temperature Anomaly", "Temperature (°C)", ["2019", "2020", "2021", "2022", "2023", "2024"]),
    ("Website Visits", "Visits (thousands)", ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"]),
]


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path, metadata: dict) -> None:
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_bar(seed: int, path: Path, value_labels: bool = False, horizontal: bool = False) -> dict:
    rng = random.Random(seed)
    title, ylabel, categories = rng.choice(BAR_TITLES)
    values = [rng.randint(12, 96) for _ in categories]
    values[rng.randrange(len(values))] = max(values) + rng.randint(5, 25)
    palette = rng.choice(PALETTES)
    fig, ax = plt.subplots(figsize=(8, 5))
    if horizontal:
        bars = ax.barh(categories, values, color=palette[: len(categories)], height=0.62)
        ax.set_xlabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.invert_yaxis()
        if value_labels:
            for bar, value in zip(bars, values):
                ax.text(value + 1, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=8)
    else:
        bars = ax.bar(categories, values, color=palette[: len(categories)], width=0.62)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim(0, max(values) * 1.18)
        if value_labels:
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max(values) * 0.02,
                    str(value),
                    ha="center",
                    fontsize=8,
                )
    _clean_axes(ax)
    _save(
        fig,
        path,
        {
            "chart_type": "bar",
            "title": title,
            "metric": ylabel,
            "labels": categories,
            "values": values,
            "horizontal": horizontal,
            "value_labels": value_labels,
        },
    )
    return json.loads(path.with_suffix(".json").read_text())


def make_pie(seed: int, path: Path, donut: bool = False, legend: bool = False) -> dict:
    rng = random.Random(seed)
    title, labels = rng.choice(PIE_TITLES)
    raw = [rng.randint(8, 50) for _ in labels]
    total = sum(raw)
    values = [round(value / total * 100, 1) for value in raw]
    correction = 100 - sum(values)
    values[0] = round(values[0] + correction, 1)
    palette = rng.choice(PALETTES)[: len(labels)]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    wedges, _texts, autotexts = ax.pie(
        values,
        labels=None if legend else labels,
        autopct="%1.0f%%",
        colors=palette,
        startangle=rng.choice([90, 60, 120, 0]),
        counterclock=False,
        pctdistance=0.72 if not donut else 0.78,
        textprops={"fontsize": 9},
        wedgeprops={"linewidth": 1.2, "edgecolor": "white", **({"width": 0.62} if donut else {})},
    )
    for text in autotexts:
        text.set_fontsize(8)
        text.set_fontweight("bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    if legend:
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    ax.axis("equal")
    _save(
        fig,
        path,
        {
            "chart_type": "pie",
            "title": title,
            "labels": labels,
            "values": values,
            "donut": donut,
            "legend": legend,
        },
    )
    return json.loads(path.with_suffix(".json").read_text())


def make_line(seed: int, path: Path) -> dict:
    rng = random.Random(seed)
    title, ylabel, labels = rng.choice(LINE_TITLES)
    base = rng.randint(20, 70)
    values = []
    value = base
    for _ in labels:
        value += rng.randint(-8, 14)
        value = max(3, value)
        values.append(value)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, values, marker="o", linewidth=2.7, markersize=6, color=rng.choice(PALETTES)[0])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.2)
    _clean_axes(ax)
    _save(
        fig,
        path,
        {
            "chart_type": "line",
            "title": title,
            "metric": ylabel,
            "labels": labels,
            "values": values,
        },
    )
    return json.loads(path.with_suffix(".json").read_text())


def make_infographic(seed: int, path: Path) -> dict:
    rng = random.Random(seed)
    topics = [
        (
            "AI Adoption in 2026",
            [
                ("73%", "of teams use AI weekly"),
                ("2.4x", "faster prototyping"),
                ("41%", "report lower costs"),
            ],
        ),
        (
            "Remote Work Snapshot",
            [
                ("68%", "prefer hybrid work"),
                ("3.1x", "more video meetings"),
                ("27%", "use co-working spaces"),
            ],
        ),
        (
            "Startup Growth Report",
            [("120M", "active users"), ("$48M", "annual revenue"), ("18", "countries served")],
        ),
        (
            "Learning Outcomes",
            [("89%", "course completion"), ("4.6/5", "average rating"), ("32K", "learners enrolled")],
        ),
    ]
    title, metrics = rng.choice(topics)
    palette = rng.choice(PALETTES)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.text(0.55, 5.25, title, fontsize=20, fontweight="bold", color="#17324d")
    ax.text(0.55, 4.92, "Key performance indicators", fontsize=10, color="#5a6b7b")
    positions = [(0.55, 2.9), (3.65, 2.9), (6.75, 2.9)]
    for index, ((value, label), (x, y)) in enumerate(zip(metrics, positions)):
        card = FancyBboxPatch(
            (x, y),
            2.7,
            1.55,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=0,
            facecolor=palette[index % len(palette)],
            alpha=0.14,
            transform=ax.transData,
        )
        ax.add_patch(card)
        ax.text(x + 0.18, y + 1.0, value, fontsize=30, fontweight="bold", color=palette[index % len(palette)])
        ax.text(x + 0.18, y + 0.52, label, fontsize=9, color="#3a4a5a", wrap=True)
    ax.text(0.55, 0.55, "Source: synthetic demonstration dataset", fontsize=8, color="#6b7c8c")
    _save(
        fig,
        path,
        {
            "chart_type": "infographic",
            "title": title,
            "labels": [label for _, label in metrics],
            "values": [value for value, _ in metrics],
        },
    )
    return json.loads(path.with_suffix(".json").read_text())


def make_showcase(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    make_bar(101, output_dir / "bar_revenue.png", value_labels=False)
    make_bar(102, output_dir / "bar_products.png", value_labels=True)
    make_pie(201, output_dir / "pie_browser.png")
    make_pie(202, output_dir / "pie_energy_donut.png", donut=True)
    make_pie(203, output_dir / "pie_budget_legend.png", legend=True)
    make_line(301, output_dir / "line_users.png")
    make_infographic(401, output_dir / "infographic_ai.png")


def make_dataset(output_dir: Path, per_class: int, seed_start: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for offset in range(per_class):
        seed = seed_start + offset
        name = f"bar_{offset:03d}"
        path = output_dir / f"{name}.png"
        make_bar(seed, path, value_labels=(offset % 4 == 0))
        index.append({"file": path.name, "chart_type": "bar", "seed": seed})
    for offset in range(per_class):
        seed = seed_start + 1000 + offset
        name = f"pie_{offset:03d}"
        path = output_dir / f"{name}.png"
        make_pie(seed, path, donut=(offset % 3 == 1), legend=(offset % 5 == 0))
        index.append({"file": path.name, "chart_type": "pie", "seed": seed})
    for offset in range(per_class):
        seed = seed_start + 2000 + offset
        name = f"line_{offset:03d}"
        path = output_dir / f"{name}.png"
        make_line(seed, path)
        index.append({"file": path.name, "chart_type": "line", "seed": seed})
    for offset in range(per_class):
        seed = seed_start + 3000 + offset
        name = f"infographic_{offset:03d}"
        path = output_dir / f"{name}.png"
        make_infographic(seed, path)
        index.append({"file": path.name, "chart_type": "infographic", "seed": seed})
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["showcase", "dataset", "both"], default="both")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.mode in {"showcase", "both"}:
        make_showcase(SAMPLE_DIR)
    if args.mode in {"dataset", "both"}:
        make_dataset(args.output_dir or DATASET_DIR, args.per_class, args.seed_start)


if __name__ == "__main__":
    main()
