import csv
from collections import defaultdict
from pathlib import Path


METRICS = [
    "Average ETE delay (s)",
    "Average energy consumption (W/Data center)",
    "Tasks success rate(%)",
]


def chapter4_output_dirs():
    root = Path("SatEdgeSim/output/chapter4")
    return sorted({path.parent for path in root.glob("*/Sequential_simulation.csv")})


def speed_from_decision_log(output_dir: Path):
    decisions = output_dir / "chapter4_ddldo_decisions.csv"
    if not decisions.exists():
        return None
    with decisions.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("vehicleSpeed", "").strip()
            if value:
                return float(value)
    return None


def load_simulation_rows(output_dir: Path):
    source = output_dir / "Sequential_simulation.csv"
    with source.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def latest_output_dir() -> Path:
    dirs = chapter4_output_dirs()
    if not dirs:
        raise FileNotFoundError("No chapter4 Sequential_simulation.csv found")
    return max(dirs, key=lambda path: path.stat().st_mtime)


def main() -> None:
    grouped = defaultdict(lambda: defaultdict(list))
    scanned = 0
    tagged = 0
    for output_dir in chapter4_output_dirs():
        scanned += 1
        speed = speed_from_decision_log(output_dir)
        if speed is None:
            continue
        tagged += 1
        for row in load_simulation_rows(output_dir):
            algorithm = row.get("Orchestration algorithm", "").strip()
            if not algorithm:
                continue
            key = (algorithm, speed)
            for metric in METRICS:
                value = row.get(metric, "").strip()
                if value:
                    grouped[key][metric].append(float(value))

    if not grouped:
        raise RuntimeError(
            "No real chapter4 speed experiment rows found. Run chapter4 once per speed with "
            "DDLDO enabled, for example: mvn exec:java "
            "'-Dexec.mainClass=edu.weijunyong.satedgesim.MainApplication' "
            "'-Dexec.args=chapter4' '-Dvehicle.speed=30'."
        )

    output = latest_output_dir() / "chapter4_speed_results.csv"
    fieldnames = ["Orchestration algorithm", "Vehicle speed"] + METRICS
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm, speed in sorted(grouped, key=lambda item: (item[0], item[1])):
            values = grouped[(algorithm, speed)]
            row = {
                "Orchestration algorithm": algorithm,
                "Vehicle speed": speed,
            }
            for metric in METRICS:
                samples = values.get(metric, [])
                row[metric] = sum(samples) / len(samples) if samples else ""
            writer.writerow(row)

    print(f"Scanned {scanned} chapter4 output directories; used {tagged} speed-tagged real runs.")
    print(f"Wrote speed sensitivity results to {output}")


if __name__ == "__main__":
    main()
