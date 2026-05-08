import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METRICS = [
    ("Average ETE delay (s)", "chapter4_fig_4_5_speed_avg_ete_delay.png", "Average end-to-end delay (s)", "speed"),
    ("Average ETE delay (s)", "chapter4_fig_4_6_avg_ete_delay.png", "Average end-to-end delay (s)"),
    ("Average energy consumption (W/Data center)", "chapter4_fig_4_7_avg_energy.png", "Average energy consumption (W/Data center)"),
    ("Tasks success rate(%)", "chapter4_fig_4_8_success_rate.png", "Task success rate (%)"),
    ("Tasks failed rate(delay)(%)", "chapter4_fig_4_9_delay_failure_rate.png", "Failure rate due to delay (%)"),
    ("Tasks failed rate(mobility)(%)", "chapter4_fig_4_10_mobility_failure_rate.png", "Failure rate due to mobility (%)"),
    ("Total tasks executed (Cloud)", "chapter4_fig_4_11a_cloud_tasks.png", "Cloud processed tasks"),
    ("Total tasks executed (Edge)", "chapter4_fig_4_11b_edge_tasks.png", "Edge processed tasks"),
    ("Total tasks executed (Mist)", "chapter4_fig_4_11c_mist_tasks.png", "Mist processed tasks"),
]

ALGORITHM_ORDER = [
    "DDLDO",
    "TASK_CLF_Q_LEARNING",
    "WEIGHT_GREEDY",
    "ROUND_ROBIN",
    "TRADE_OFF",
    "TRADI_POLLING",
    "RANDOM_VM",
]

COLORS = [
    (0, 0, 255),
    (0, 128, 0),
    (255, 0, 0),
    (0, 188, 188),
    (188, 0, 188),
    (188, 188, 0),
    (64, 64, 64),
]


def latest_results_csv() -> Path:
    files = list(Path("SatEdgeSim/output/chapter4").glob("*/chapter4_results.csv"))
    if not files:
        raise FileNotFoundError("No chapter4_results.csv found")
    return max(files, key=lambda path: path.stat().st_mtime)


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def latest_speed_csv(results_csv: Path) -> Path:
    speed_csv = results_csv.with_name("chapter4_speed_results.csv")
    if not speed_csv.exists():
        raise FileNotFoundError(f"No speed results file found at {speed_csv}")
    return speed_csv


def font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(algorithm: str) -> str:
    if algorithm == "TASK_CLF_Q_LEARNING":
        return "TaskClfQLearning"
    return algorithm


def algorithms(rows):
    present = {row["Orchestration algorithm"] for row in rows}
    ordered = [name for name in ALGORITHM_ORDER if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def plot_metric(rows, metric: str, output: Path, ylabel: str, x_column: str = "Edge devices count",
        x_label: str = "Number of edge devices") -> None:
    width, height = 1280, 760
    left, right, top, bottom = 120, 330, 70, 640
    plot_right = width - right
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28)
    text_font = font(20)
    small_font = font(17)

    series = []
    values = []
    for algorithm in algorithms(rows):
        items = sorted(
            [row for row in rows if row["Orchestration algorithm"] == algorithm],
            key=lambda row: float(row[x_column]),
        )
        points = [(float(row[x_column]), float(row[metric])) for row in items]
        series.append((algorithm, points))
        values.extend(value for _, value in points)

    x_values = sorted({x for _, points in series for x, _ in points})
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(values), max(values)
    pad = (y_max - y_min) * 0.08 if y_max != y_min else 1
    y_min -= pad
    y_max += pad

    draw.text((left, 24), ylabel, fill=(20, 20, 20), font=title_font)
    draw.rectangle((left, top, plot_right, bottom), outline=(80, 80, 80), width=2)
    for i in range(6):
        y = top + i * (bottom - top) // 5
        value = y_max - i * (y_max - y_min) / 5
        draw.line((left, y, plot_right, y), fill=(220, 220, 220), width=1)
        draw.text((18, y - 10), f"{value:.2g}", fill=(70, 70, 70), font=small_font)

    for x in x_values:
        if x_max == x_min:
            px = (left + plot_right) // 2
        else:
            px = int(left + (x - x_min) * (plot_right - left) / (x_max - x_min))
        draw.line((px, bottom, px, bottom + 6), fill=(80, 80, 80), width=2)
        draw.text((px - 20, bottom + 12), f"{x:g}", fill=(60, 60, 60), font=small_font)

    draw.text(((left + plot_right) // 2 - 100, height - 55), x_label, fill=(40, 40, 40), font=text_font)

    for index, (algorithm, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        pixels = []
        for x, value in points:
            if x_max == x_min:
                px = (left + plot_right) // 2
            else:
                px = int(left + (x - x_min) * (plot_right - left) / (x_max - x_min))
            py = int(bottom - (value - y_min) * (bottom - top) / (y_max - y_min))
            pixels.append((px, py))
        if len(pixels) > 1:
            draw.line(pixels, fill=color, width=4, joint="curve")
        for px, py in pixels:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline="white", width=2)

        lx, ly = plot_right + 35, top + index * 42
        draw.line((lx, ly + 12, lx + 30, ly + 12), fill=color, width=4)
        draw.ellipse((lx + 10, ly + 7, lx + 20, ly + 17), fill=color)
        draw.text((lx + 42, ly), label(algorithm), fill=(40, 40, 40), font=small_font)

    output.parent.mkdir(exist_ok=True)
    image.save(output)


def main() -> None:
    csv_path = latest_results_csv()
    rows = load_rows(csv_path)
    speed_rows = load_rows(latest_speed_csv(csv_path))
    output_dir = csv_path.parent / "chapter4_figures"
    for item in METRICS:
        if len(item) == 4 and item[3] == "speed":
            metric, filename, ylabel, _ = item
            plot_metric(speed_rows, metric, output_dir / filename, ylabel, "Vehicle speed", "Vehicle speed")
        else:
            metric, filename, ylabel = item
            plot_metric(rows, metric, output_dir / filename, ylabel)
    print(f"Wrote {len(METRICS)} figures to {output_dir}")


if __name__ == "__main__":
    main()
