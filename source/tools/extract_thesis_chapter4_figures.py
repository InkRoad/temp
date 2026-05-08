from pathlib import Path

from pypdf import PdfReader


FIGURES = [
    (93, 0, "figure_4_5_speed_avg_ete_delay.png"),
    (93, 1, "figure_4_6_device_count_avg_ete_delay.png"),
    (94, 0, "figure_4_7_average_energy.png"),
    (94, 1, "figure_4_8_success_rate.png"),
    (95, 0, "figure_4_9_delay_failure_rate.png"),
    (95, 1, "figure_4_10_mobility_failure_rate.png"),
    (96, 0, "figure_4_11a_cloud_tasks.png"),
    (96, 1, "figure_4_11b_edge_tasks.png"),
    (96, 2, "figure_4_11c_mist_tasks.png"),
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    pdf_path = workspace_root / "帅嘉琪-毕业论文-2025.4.28.pdf"
    output_dir = project_root / "SatEdgeSim" / "output" / "paper_chapter4_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    for page_number, image_index, filename in FIGURES:
        page = reader.pages[page_number - 1]
        image = list(page.images)[image_index]
        (output_dir / filename).write_bytes(image.data)

    print(f"Wrote {len(FIGURES)} thesis figures to {output_dir}")


if __name__ == "__main__":
    main()
