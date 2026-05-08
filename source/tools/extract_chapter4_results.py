import csv
from pathlib import Path


COLUMNS = [
    "Orchestration algorithm",
    "Edge devices count",
    "Average ETE delay (s)",
    "Average execution delay (s)",
    "Average energy consumption (W/Data center)",
    "Tasks success rate(%)",
    "Tasks failed rate(delay)(%)",
    "Tasks failed rate(mobility)(%)",
    "Total tasks executed (Cloud)",
    "Total tasks executed (Edge)",
    "Total tasks executed (Mist)",
    "Tasks successfully executed (Cloud)",
    "Tasks successfully executed (Edge)",
    "Tasks successfully executed (Mist)",
]


def main() -> None:
    csv_files = list(Path("SatEdgeSim/output/chapter4").glob("*/Sequential_simulation.csv"))
    if not csv_files:
        raise FileNotFoundError("No chapter4 Sequential_simulation.csv found")

    source = max(csv_files, key=lambda path: path.stat().st_mtime)
    output = source.with_name("chapter4_results.csv")

    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
