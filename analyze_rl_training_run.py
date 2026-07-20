#!/usr/bin/env python3
"""Generate a standalone HTML report from a gym-duckietown PPO run."""

from __future__ import annotations

import argparse
import csv
import html
import math
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

from cli_completion import parse_args_with_completion


REQUIRED_FILES = (
    "eval_history.csv",
    "eval_scenarios.csv",
    "history.csv",
    "ppo_diagnostics.csv",
    "reward_components_history.csv",
    "rollout_history.csv",
)

COLORS = ("#147d92", "#d1495b", "#2d6a4f", "#e09f3e", "#6d597a", "#577590")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a self-contained HTML analysis of a PPO training run.",
    )
    parser.add_argument("run_dir", type=Path, help="Run directory containing the PPO CSV files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML file (default: RUN_DIR/training_report.html).",
    )
    parser.add_argument(
        "--eval-window",
        type=int,
        default=50,
        help="Number of first/last evaluations used for comparisons.",
    )
    parser.add_argument(
        "--episode-window",
        type=int,
        default=500,
        help="Number of chronologically first/last training episodes to compare.",
    )
    parser.add_argument(
        "--diagnostic-window",
        type=int,
        default=100,
        help="Number of first/last PPO updates and rollouts to compare.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Window used for rolling-average chart lines.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated report in the default browser.",
    )
    args = parse_args_with_completion(parser)
    for name in ("eval_window", "episode_window", "diagnostic_window", "rolling_window"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def number(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def integer(row: dict[str, str], key: str, default: int = 0) -> int:
    value = number(row, key, float(default))
    return int(value) if math.isfinite(value) else default


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def average(values: Iterable[float]) -> float:
    values = finite(values)
    return fmean(values) if values else math.nan


def linear_trend(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    points = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(points) < 2:
        return math.nan, math.nan
    mean_x = average(x for x, _ in points)
    mean_y = average(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0.0:
        return math.nan, math.nan
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    r_squared = 1.0 - residual / total if total > 0.0 else math.nan
    return slope, r_squared


def rolling_points(points: Sequence[tuple[float, float]], window: int) -> list[tuple[float, float]]:
    result = []
    for index in range(window - 1, len(points)):
        values = finite(value for _, value in points[index - window + 1 : index + 1])
        if values:
            result.append((points[index][0], average(values)))
    return result


def fmt(value: float, digits: int = 2, suffix: str = "") -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}{suffix}"


def format_step(value: float) -> str:
    return f"{int(value):,}".replace(",", " ") if math.isfinite(value) else "-"


def html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def svg_line_chart(
    series: Sequence[tuple[str, Sequence[tuple[float, float]], str]],
    *,
    width: int = 980,
    height: int = 340,
) -> str:
    all_points = [point for _, points, _ in series for point in points if all(math.isfinite(v) for v in point)]
    if not all_points:
        return "<p class='muted'>No chart data available.</p>"
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0
    y_padding = 0.06 * (max_y - min_y)
    min_y -= y_padding
    max_y += y_padding
    left, right, top, bottom = 74, 24, 24, 58
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_width

    def sy(value: float) -> float:
        return top + (max_y - value) / (max_y - min_y) * plot_height

    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for index in range(6):
        fraction = index / 5
        y_value = max_y - fraction * (max_y - min_y)
        y = top + fraction * plot_height
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' class='grid'/>")
        parts.append(f"<text x='{left-9}' y='{y+4:.1f}' text-anchor='end' class='tick'>{y_value:.2f}</text>")
    for index in range(6):
        fraction = index / 5
        x_value = min_x + fraction * (max_x - min_x)
        x = left + fraction * plot_width
        parts.append(f"<text x='{x:.1f}' y='{height-31}' text-anchor='middle' class='tick'>{x_value:.0f}</text>")
    parts.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' class='axis'/>")
    parts.append(f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' class='axis'/>")
    for name, points, color in series:
        clean = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
        coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in clean)
        parts.append(f"<polyline points='{coordinates}' fill='none' stroke='{color}' stroke-width='2'/>")
    legend_x = left + 8
    for index, (name, _, color) in enumerate(series):
        y = 15 + index * 18
        parts.append(f"<line x1='{legend_x}' y1='{y}' x2='{legend_x+22}' y2='{y}' stroke='{color}' stroke-width='3'/>")
        parts.append(f"<text x='{legend_x+28}' y='{y+4}' class='legend'>{html.escape(name)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def svg_grouped_bars(
    categories: Sequence[str],
    first_values: Sequence[float],
    last_values: Sequence[float],
    first_label: str,
    last_label: str,
    *,
    width: int = 980,
    height: int = 390,
) -> str:
    if not categories:
        return "<p class='muted'>No chart data available.</p>"
    left, right, top, bottom = 64, 20, 35, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max([100.0, *finite(first_values), *finite(last_values)])
    group_width = plot_width / len(categories)
    bar_width = min(26.0, group_width * 0.32)
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' class='chart'>"]
    for index in range(6):
        value = max_value * index / 5
        y = top + plot_height - value / max_value * plot_height
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' class='grid'/>")
        parts.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' class='tick'>{value:.0f}%</text>")
    for index, category in enumerate(categories):
        center = left + (index + 0.5) * group_width
        for offset, value, color in ((-bar_width, first_values[index], COLORS[0]), (0, last_values[index], COLORS[1])):
            if not math.isfinite(value):
                continue
            bar_height = value / max_value * plot_height
            parts.append(
                f"<rect x='{center+offset:.1f}' y='{top+plot_height-bar_height:.1f}' "
                f"width='{bar_width:.1f}' height='{bar_height:.1f}' fill='{color}'/>"
            )
        parts.append(
            f"<text x='{center:.1f}' y='{height-bottom+19}' transform='rotate(38 {center:.1f} {height-bottom+19})' "
            f"text-anchor='start' class='tick'>{html.escape(category)}</text>"
        )
    parts.append(f"<rect x='{left}' y='8' width='12' height='12' fill='{COLORS[0]}'/><text x='{left+18}' y='19' class='legend'>{html.escape(first_label)}</text>")
    parts.append(f"<rect x='{left+180}' y='8' width='12' height='12' fill='{COLORS[1]}'/><text x='{left+198}' y='19' class='legend'>{html.escape(last_label)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def scenario_statistics(rows: Sequence[dict[str, str]], window: int) -> list[dict[str, float]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["scenario_seed"]].append(row)
    result = []
    for seed, group in groups.items():
        group.sort(key=lambda row: integer(row, "eval_index"))
        size = min(window, len(group))
        first = group[:size]
        last = group[-size:]
        xs = [number(row, "train_step") for row in group]
        ys = [number(row, "scenario_return") for row in group]
        slope, r_squared = linear_trend(xs, ys)
        result.append(
            {
                "seed": float(seed),
                "count": float(len(group)),
                "mean": average(ys),
                "first": average(number(row, "scenario_return") for row in first),
                "last": average(number(row, "scenario_return") for row in last),
                "slope_per_million": slope * 1_000_000,
                "r_squared": r_squared,
                "complete": 100.0 * average(1.0 if integer(row, "terminated") == 0 else 0.0 for row in group),
                "first_complete": 100.0 * average(1.0 if integer(row, "terminated") == 0 else 0.0 for row in first),
                "last_complete": 100.0 * average(1.0 if integer(row, "terminated") == 0 else 0.0 for row in last),
                "mean_steps": average(number(row, "scenario_steps") for row in group),
                "minimum": min(ys),
                "maximum": max(ys),
            }
        )
    return sorted(result, key=lambda item: item["seed"])


def start_label(row: dict[str, str]) -> str:
    if row.get("start_type") == "hard_seed":
        return f"seed {integer(row, 'start_seed')}"
    if row.get("start_type") == "hard_pose":
        return f"pose {row.get('start_name') or 'unnamed'}"
    return "random (combined)"


def start_statistics(rows: Sequence[dict[str, str]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[start_label(row)].append(row)
    total_steps = sum(number(row, "episode_length", 0.0) for row in rows)
    result = {}
    for label, group in groups.items():
        steps = sum(number(row, "episode_length", 0.0) for row in group)
        result[label] = {
            "episodes": float(len(group)),
            "invalid_pct": 100.0 * average(1.0 if row.get("done_reason") == "invalid-pose" else 0.0 for row in group),
            "mean_length": average(number(row, "episode_length") for row in group),
            "step_share_pct": 100.0 * steps / total_steps if total_steps > 0.0 else math.nan,
            "reward_per_step": average(number(row, "episode_return_per_step") for row in group),
        }
    return result


def start_sort_key(label: str) -> tuple[int, object]:
    if label.startswith("seed "):
        return 0, int(label.split()[1])
    if label.startswith("pose "):
        return 1, label
    return 2, label


def metric_window_table(
    rows: Sequence[dict[str, str]], metrics: Sequence[tuple[str, str]], window: int
) -> list[list[str]]:
    size = min(window, len(rows))
    first = rows[:size]
    last = rows[-size:]
    result = []
    for key, label in metrics:
        first_mean = average(number(row, key) for row in first)
        last_mean = average(number(row, key) for row in last)
        result.append([label, fmt(first_mean, 4), fmt(last_mean, 4), fmt(last_mean - first_mean, 4)])
    return result


def reward_component_means(
    rows: Sequence[dict[str, str]], phase: str, window: int
) -> list[list[str]]:
    selected = (
        "Reward",
        "Velocity",
        "Pose",
        "Pose.HeadingQuality",
        "Pose.LaneDistancePenalty",
        "Pose.ScaledAbsLaneDistance",
        "InvalidPosePenalty",
    )
    phase_rows = [row for row in rows if row.get("phase") == phase]
    identifiers = sorted({integer(row, "train_rollout") for row in phase_rows})
    size = min(window, len(identifiers))
    first_ids = set(identifiers[:size])
    last_ids = set(identifiers[-size:])
    values: dict[tuple[int, str], float] = {}
    for row in phase_rows:
        values[(integer(row, "train_rollout"), row.get("component", ""))] = number(
            row, "component_mean_per_step", 0.0
        )
    result = []
    for component in selected:
        first = average(values.get((identifier, component), 0.0) for identifier in first_ids)
        last = average(values.get((identifier, component), 0.0) for identifier in last_ids)
        result.append([component, fmt(first, 4), fmt(last, 4), fmt(last - first, 4)])
    return result


def build_report(
    run_dir: Path,
    eval_history: list[dict[str, str]],
    eval_scenarios: list[dict[str, str]],
    history: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    reward_components: list[dict[str, str]],
    rollouts: list[dict[str, str]],
    args: argparse.Namespace,
) -> str:
    eval_history.sort(key=lambda row: integer(row, "eval_index"))
    eval_scenarios.sort(key=lambda row: (integer(row, "eval_index"), integer(row, "scenario_index")))
    history.sort(key=lambda row: integer(row, "episode"))
    diagnostics.sort(key=lambda row: integer(row, "rollout"))
    rollouts.sort(key=lambda row: integer(row, "rollout"))

    eval_size = min(args.eval_window, len(eval_history))
    first_eval = eval_history[:eval_size]
    last_eval = eval_history[-eval_size:]
    mean_return_first = average(number(row, "eval_mean_scenario_return") for row in first_eval)
    mean_return_last = average(number(row, "eval_mean_scenario_return") for row in last_eval)
    complete_first = average(number(row, "eval_safe_scenarios") for row in first_eval)
    complete_last = average(number(row, "eval_safe_scenarios") for row in last_eval)
    eval_x = [number(row, "train_step") for row in eval_history]
    eval_y = [number(row, "eval_mean_scenario_return") for row in eval_history]
    eval_slope, eval_r_squared = linear_trend(eval_x, eval_y)
    best_eval = max(eval_history, key=lambda row: number(row, "eval_mean_scenario_return"))
    latest_eval = eval_history[-1]
    scenario_stats = scenario_statistics(eval_scenarios, args.eval_window)

    overall_starts = start_statistics(history)
    episode_size = min(args.episode_window, len(history))
    first_starts = start_statistics(history[:episode_size])
    last_starts = start_statistics(history[-episode_size:])
    start_labels = sorted(set(overall_starts) | set(first_starts) | set(last_starts), key=start_sort_key)

    eval_points = [(number(row, "eval_index"), number(row, "eval_mean_scenario_return")) for row in eval_history]
    eval_chart = svg_line_chart(
        (
            ("Mean scenario return", eval_points, "#9aa4ad"),
            (f"Rolling mean ({args.rolling_window})", rolling_points(eval_points, args.rolling_window), COLORS[0]),
        )
    )
    scenario_groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in eval_scenarios:
        scenario_groups[row["scenario_seed"]].append(
            (number(row, "eval_index"), number(row, "scenario_return"))
        )
    scenario_chart = svg_line_chart(
        tuple(
            (f"Seed {seed}", points, COLORS[index % len(COLORS)])
            for index, (seed, points) in enumerate(sorted(scenario_groups.items(), key=lambda item: int(item[0])))
        )
    )
    completion_points = [
        (number(row, "eval_index"), number(row, "eval_safe_scenarios")) for row in eval_history
    ]
    completion_chart = svg_line_chart((("Completed scenarios", completion_points, COLORS[2]),), height=260)
    invalid_chart = svg_grouped_bars(
        start_labels,
        [first_starts.get(label, {}).get("invalid_pct", math.nan) for label in start_labels],
        [last_starts.get(label, {}).get("invalid_pct", math.nan) for label in start_labels],
        f"First {episode_size} episodes",
        f"Last {episode_size} episodes",
    )
    rollout_points = [(number(row, "rollout"), number(row, "rollout_reward_per_step")) for row in rollouts]
    rollout_chart = svg_line_chart(
        (
            ("Reward per step", rollout_points, "#b5bdc4"),
            (f"Rolling mean ({args.rolling_window})", rolling_points(rollout_points, args.rolling_window), COLORS[3]),
        )
    )
    std_chart = svg_line_chart(
        (
            (
                "Raw policy std, control 0",
                [(number(row, "rollout"), number(row, "std_left")) for row in diagnostics],
                COLORS[0],
            ),
            (
                "Raw policy std, control 1",
                [(number(row, "rollout"), number(row, "std_right")) for row in diagnostics],
                COLORS[1],
            ),
        ),
        height=280,
    )

    scenario_rows = []
    for item in scenario_stats:
        scenario_rows.append(
            [
                str(int(item["seed"])),
                fmt(item["first"]),
                fmt(item["last"]),
                fmt(item["last"] - item["first"]),
                fmt(item["first_complete"], 1, "%"),
                fmt(item["last_complete"], 1, "%"),
                fmt(item["slope_per_million"]),
                fmt(item["r_squared"], 3),
                fmt(item["minimum"]),
                fmt(item["maximum"]),
            ]
        )

    start_overall_rows = []
    start_comparison_rows = []
    for label in start_labels:
        overall = overall_starts.get(label, {})
        first = first_starts.get(label, {})
        last = last_starts.get(label, {})
        start_overall_rows.append(
            [
                label,
                str(int(overall.get("episodes", 0))),
                fmt(overall.get("invalid_pct", math.nan), 2, "%"),
                fmt(overall.get("mean_length", math.nan), 1),
                fmt(overall.get("step_share_pct", math.nan), 2, "%"),
                fmt(overall.get("reward_per_step", math.nan), 3),
            ]
        )
        start_comparison_rows.append(
            [
                label,
                f"{int(first.get('episodes', 0))} / {int(last.get('episodes', 0))}",
                f"{fmt(first.get('invalid_pct', math.nan), 1, '%')} -> {fmt(last.get('invalid_pct', math.nan), 1, '%')}",
                f"{fmt(first.get('mean_length', math.nan), 1)} -> {fmt(last.get('mean_length', math.nan), 1)}",
                f"{fmt(first.get('step_share_pct', math.nan), 2, '%')} -> {fmt(last.get('step_share_pct', math.nan), 2, '%')}",
            ]
        )

    diagnostic_metrics = (
        ("approx_kl", "Approximate KL"),
        ("clip_fraction", "Clip fraction"),
        ("ratio_mean", "Ratio mean"),
        ("log_std_left", "Log std, control 0"),
        ("std_left", "Std, control 0"),
        ("policy_control_noise_0_std", "Control-noise std 0"),
        ("policy_control_noise_1_std", "Control-noise std 1"),
        ("sampled_action_saturation_fraction", "Sampled wheel saturation"),
        ("squashed_entropy_estimate", "Squashed entropy estimate"),
    )
    rollout_metrics = (
        ("rollout_reward_per_step", "Rollout reward per step"),
        ("policy_loss", "Policy loss"),
        ("value_loss", "Value loss"),
        ("entropy", "Gaussian entropy"),
        ("environment_steps_per_second", "Environment steps/s"),
        ("cycle_steps_per_second", "Rollout + update steps/s"),
        ("rollout_seconds", "Rollout seconds"),
        ("update_seconds", "Update seconds"),
    )

    latest_step = number(latest_eval, "train_step")
    first_step = number(eval_history[0], "train_step")
    cards = (
        ("Evaluations", str(len(eval_history))),
        ("Observed step range", f"{format_step(first_step)} - {format_step(latest_step)}"),
        ("Latest mean return", fmt(number(latest_eval, "eval_mean_scenario_return"))),
        ("Best mean return", fmt(number(best_eval, "eval_mean_scenario_return"))),
        ("Best eval", f"#{integer(best_eval, 'eval_index')} at step {format_step(number(best_eval, 'train_step'))}"),
        ("Latest completed", f"{integer(latest_eval, 'eval_safe_scenarios')} / {integer(latest_eval, 'eval_scenarios')}"),
    )
    card_html = "".join(
        f"<div class='card'><div class='card-label'>{html.escape(label)}</div><div class='card-value'>{html.escape(value)}</div></div>"
        for label, value in cards
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PPO training report - {html.escape(run_dir.name)}</title>
<style>
:root {{ color-scheme: light; --ink:#172126; --muted:#637078; --line:#d8dee2; --panel:#f5f7f8; --accent:#147d92; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:white; }}
main {{ max-width:1180px; margin:0 auto; padding:36px 28px 72px; }}
h1 {{ margin:0 0 6px; font-size:30px; letter-spacing:0; }}
h2 {{ margin:42px 0 14px; padding-bottom:7px; border-bottom:2px solid var(--ink); font-size:21px; letter-spacing:0; }}
h3 {{ margin:25px 0 8px; font-size:16px; letter-spacing:0; }}
p {{ max-width:900px; }}
.muted {{ color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin:20px 0; }}
.card {{ border:1px solid var(--line); border-radius:6px; padding:13px 15px; background:var(--panel); }}
.card-label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
.card-value {{ margin-top:3px; font-size:20px; font-weight:650; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:5px; }}
table {{ border-collapse:collapse; width:100%; min-width:720px; }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ background:var(--panel); font-size:12px; text-transform:uppercase; }}
tbody tr:last-child td {{ border-bottom:0; }}
.chart {{ display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:5px; background:white; }}
.grid {{ stroke:#e7ebed; stroke-width:1; }} .axis {{ stroke:#67747b; stroke-width:1; }}
.tick,.legend {{ fill:#536169; font-size:11px; }}
code {{ background:var(--panel); padding:2px 5px; border-radius:3px; }}
@media print {{ main {{ max-width:none; padding:0; }} .chart {{ break-inside:avoid; }} h2 {{ break-before:page; }} }}
</style>
</head>
<body><main>
<h1>PPO training report</h1>
<p class="muted"><code>{html.escape(str(run_dir))}</code><br>Generated {datetime.now().astimezone().isoformat(timespec='seconds')}</p>
<div class="cards">{card_html}</div>

<h2>Evaluation</h2>
<p>The first/last comparison uses {eval_size} evaluations. The linear trend is
<strong>{fmt(eval_slope * 1_000_000)}</strong> mean-scenario-return per million training steps
with R-squared <strong>{fmt(eval_r_squared, 3)}</strong>. This is descriptive; sequential evaluations are not independent samples.</p>
{html_table(("Metric", f"First {eval_size}", f"Last {eval_size}", "Change"), (
    ("Mean scenario return", fmt(mean_return_first), fmt(mean_return_last), fmt(mean_return_last-mean_return_first)),
    ("Mean completed scenarios", fmt(complete_first), fmt(complete_last), fmt(complete_last-complete_first)),
    ("Mean scenario length", fmt(average(number(row, 'eval_mean_scenario_length') for row in first_eval), 1), fmt(average(number(row, 'eval_mean_scenario_length') for row in last_eval), 1), fmt(average(number(row, 'eval_mean_scenario_length') for row in last_eval)-average(number(row, 'eval_mean_scenario_length') for row in first_eval), 1)),
))}
<h3>Aggregate return</h3>{eval_chart}
<h3>Per-scenario return</h3>{scenario_chart}
{html_table(("Seed", f"Return first {eval_size}", f"Return last {eval_size}", "Change", "Complete first", "Complete last", "Slope / 1M", "R-squared", "Minimum", "Maximum"), scenario_rows)}
<h3>Completed scenarios per evaluation</h3>{completion_chart}

<h2>Training starts</h2>
<p>The comparison uses the chronologically first and last {episode_size} completed training episodes. Step shares use all steps inside the respective period, exposing the difference between reset probability and actual PPO sample probability.</p>
{html_table(("Start", "Episodes", "Invalid pose", "Mean length", "Step share", "Reward / step"), start_overall_rows)}
<h3>First versus last episodes</h3>
{html_table(("Start", "Episodes first / last", "Invalid pose", "Mean length", "Step share"), start_comparison_rows)}
{invalid_chart}

<h2>Rollouts and PPO</h2>
<h3>Rollout reward</h3>{rollout_chart}
{html_table(("Metric", f"First {min(args.diagnostic_window, len(rollouts))}", f"Last {min(args.diagnostic_window, len(rollouts))}", "Change"), metric_window_table(rollouts, rollout_metrics, args.diagnostic_window))}
<h3>PPO diagnostics</h3>
{html_table(("Metric", f"First {min(args.diagnostic_window, len(diagnostics))}", f"Last {min(args.diagnostic_window, len(diagnostics))}", "Change"), metric_window_table(diagnostics, diagnostic_metrics, args.diagnostic_window))}
<h3>Policy exploration</h3>{std_chart}

<h2>Reward components</h2>
<p>Means are calculated per rollout step. Missing terminal penalties are treated as zero.</p>
{html_table(("Component", f"First {min(args.diagnostic_window, len(rollouts))}", f"Last {min(args.diagnostic_window, len(rollouts))}", "Change"), reward_component_means(reward_components, 'train_rollout', args.diagnostic_window))}

<h2>Data provenance</h2>
{html_table(("File", "Rows"), tuple((name, str(count)) for name, count in (
    ('eval_history.csv', len(eval_history)), ('eval_scenarios.csv', len(eval_scenarios)),
    ('history.csv', len(history)), ('ppo_diagnostics.csv', len(diagnostics)),
    ('reward_components_history.csv', len(reward_components)), ('rollout_history.csv', len(rollouts)),
)))}
<p class="muted">The report is self-contained: charts are inline SVG and no network connection is required.</p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{run_dir} is missing required files: {', '.join(missing)}"
        )
    output = args.output.expanduser() if args.output is not None else run_dir / "training_report.html"
    if not output.is_absolute():
        output = Path.cwd() / output
    report = build_report(
        run_dir,
        read_csv_rows(run_dir / "eval_history.csv"),
        read_csv_rows(run_dir / "eval_scenarios.csv"),
        read_csv_rows(run_dir / "history.csv"),
        read_csv_rows(run_dir / "ppo_diagnostics.csv"),
        read_csv_rows(run_dir / "reward_components_history.csv"),
        read_csv_rows(run_dir / "rollout_history.csv"),
        args,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"Wrote PPO training report: {output}")
    if not args.no_open:
        try:
            opened = webbrowser.open(output.resolve().as_uri(), new=2)
        except webbrowser.Error as error:
            print(f"Could not open the report in a browser: {error}")
        else:
            if not opened:
                print(
                    "Could not open the report in a browser. "
                    "Open it manually or use --no-open on headless systems."
                )


if __name__ == "__main__":
    main()
