from pathlib import Path

from pypdf import PdfReader


FIGURES = [
    (68, 0, "figure_3_2_average_ete_delay.png"),
    (68, 1, "figure_3_3a_cloud_tasks.png"),
    (68, 2, "figure_3_3b_edge_tasks.png"),
    (68, 3, "figure_3_3c_mist_tasks.png"),
    (69, 0, "figure_3_4_average_execution_delay.png"),
    (70, 0, "figure_3_5_average_energy.png"),
    (70, 1, "figure_3_6_success_rate.png"),
    (71, 0, "figure_3_7_delay_failure_rate.png"),
    (72, 0, "figure_3_8_mobility_failure_rate.png"),
]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    workspace_root = project_root.parent
    pdf_path = workspace_root / "帅嘉琪-毕业论文-2025.4.28.pdf"
    output_dir = project_root / "SatEdgeSim" / "output" / "paper_chapter3_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    for page_number, image_index, filename in FIGURES:
        page = reader.pages[page_number - 1]
        image = list(page.images)[image_index]
        target = output_dir / filename
        target.write_bytes(image.data)

    print(f"Wrote {len(FIGURES)} thesis figures to {output_dir}")


if __name__ == "__main__":
    main()
