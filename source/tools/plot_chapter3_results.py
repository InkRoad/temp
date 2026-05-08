import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METRICS = [
    ("Average ETE delay (s)", "chapter3_fig_3_2_avg_ete_delay.png", "Average end-to-end delay (s)"),
    ("Average execution delay (s)", "chapter3_fig_3_4_avg_execution_delay.png", "Average execution delay (s)"),
    ("Average energy consumption (W/Data center)", "chapter3_fig_3_5_avg_energy.png", "Average energy consumption (W/Data center)"),
    ("Tasks success rate(%)", "chapter3_fig_3_6_success_rate.png", "Task success rate (%)"),
    ("Tasks failed rate(delay)(%)", "chapter3_fig_3_7_delay_failure_rate.png", "Failure rate due to delay (%)"),
    ("Tasks failed rate(mobility)(%)", "chapter3_fig_3_8_mobility_failure_rate.png", "Failure rate due to mobility (%)"),
    ("Total tasks executed (Cloud)", "chapter3_fig_3_3a_cloud_tasks.png", "Cloud processed tasks"),
    ("Total tasks executed (Edge)", "chapter3_fig_3_3b_edge_tasks.png", "Edge processed tasks"),
    ("Total tasks executed (Mist)", "chapter3_fig_3_3c_mist_tasks.png", "Mist processed tasks"),
]

COLORS = [
    (0, 0, 255),
    (0, 128, 0),
    (255, 0, 0),
    (0, 188, 188),
    (188, 0, 188),
    (188, 188, 0),
]

ALGORITHM_ORDER = [
    "TASK_CLF_Q_LEARNING",
    "WEIGHT_GREEDY",
    "ROUND_ROBIN",
    "TRADE_OFF",
    "TRADI_POLLING",
    "RANDOM_VM",
]

FALLBACK_COLORS = [
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (255, 127, 14),
    (148, 103, 189),
]


def latest_chapter3_csv() -> Path:
    files = list(Path("SatEdgeSim/output/chapter3").glob("*/chapter3_results.csv"))
    if not files:
        files = list(Path("SatEdgeSim/output").glob("*/chapter3_results.csv"))
    if not files:
        raise FileNotFoundError("No chapter3_results.csv file found under SatEdgeSim/output")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def nice_label(algorithm: str) -> str:
    return "TaskClfQLearning" if algorithm == "TASK_CLF_Q_LEARNING" else algorithm


def ordered_algorithms(rows):
    present = {row["Orchestration algorithm"] for row in rows}
    ordered = [algorithm for algorithm in ALGORITHM_ORDER if algorithm in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def transform(value: float, low: float, high: float, top: int, bottom: int) -> int:
    if high == low:
        return (top + bottom) // 2
    return int(bottom - (value - low) * (bottom - top) / (high - low))


def plot_metric(rows, metric: str, output: Path, ylabel: str) -> None:
    width, height = 1280, 760
    left, right, top, bottom = 120, 310, 70, 640
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(28)
    text_font = font(20)
    small_font = font(17)

    algorithms = ordered_algorithms(rows)
    series = []
    all_y = []
    for algorithm in algorithms:
        alg_rows = sorted(
            [row for row in rows if row["Orchestration algorithm"] == algorithm],
            key=lambda row: int(row["Edge devices count"]),
        )
        points = [(int(row["Edge devices count"]), float(row[metric])) for row in alg_rows]
        series.append((algorithm, points))
        all_y.extend(y for _, y in points)

    x_values = sorted({x for _, points in series for x, _ in points})
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = (y_max - y_min) * 0.08 if y_max != y_min else 1
    y_min -= y_pad
    y_max += y_pad

    plot_right = width - right
    draw.rectangle((left, top, plot_right, bottom), outline=(80, 80, 80), width=2)
    draw.text((left, 24), ylabel, fill=(20, 20, 20), font=title_font)

    for i in range(6):
        y = top + i * (bottom - top) // 5
        value = y_max - i * (y_max - y_min) / 5
        draw.line((left, y, plot_right, y), fill=(220, 220, 220), width=1)
        draw.text((18, y - 10), f"{value:.2g}", fill=(70, 70, 70), font=small_font)

    for x in x_values:
        px = int(left + (x - x_min) * (plot_right - left) / (x_max - x_min))
        draw.line((px, bottom, px, bottom + 6), fill=(80, 80, 80), width=2)
        draw.text((px - 22, bottom + 12), str(x), fill=(60, 60, 60), font=small_font)

    draw.text(((left + plot_right) // 2 - 100, height - 55), "Number of edge devices", fill=(40, 40, 40), font=text_font)
    draw.text((12, top - 38), ylabel, fill=(40, 40, 40), font=small_font)

    for idx, (algorithm, points) in enumerate(series):
        color = COLORS[idx] if idx < len(COLORS) else FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]
        pixels = []
        for x, y_value in points:
            px = int(left + (x - x_min) * (plot_right - left) / (x_max - x_min))
            py = transform(y_value, y_min, y_max, top, bottom)
            pixels.append((px, py))
        if len(pixels) > 1:
            draw.line(pixels, fill=color, width=4, joint="curve")
        for px, py in pixels:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline="white", width=2)

        legend_x = plot_right + 35
        legend_y = top + idx * 42
        draw.line((legend_x, legend_y + 12, legend_x + 30, legend_y + 12), fill=color, width=4)
        draw.ellipse((legend_x + 10, legend_y + 7, legend_x + 20, legend_y + 17), fill=color)
        draw.text((legend_x + 42, legend_y), nice_label(algorithm), fill=(40, 40, 40), font=small_font)

    output.parent.mkdir(exist_ok=True)
    img.save(output)


def main() -> None:
    csv_path = latest_chapter3_csv()
    rows = load_rows(csv_path)
    output_dir = csv_path.parent / "chapter3_figures"
    for metric, filename, ylabel in METRICS:
        plot_metric(rows, metric, output_dir / filename, ylabel)
    print(f"Wrote {len(METRICS)} figures to {output_dir}")


if __name__ == "__main__":
    main()
