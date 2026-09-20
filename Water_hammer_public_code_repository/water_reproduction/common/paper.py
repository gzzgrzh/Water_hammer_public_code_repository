"""Extract one paper figure and pair it with one clearly labelled result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Crop:
    page: int
    box: Tuple[float, float, float, float]


CROPS: Dict[int, Crop] = {
    3: Crop(9, (0.27, 0.25, 0.73, 0.53)),
    5: Crop(10, (0.27, 0.27, 0.76, 0.57)),
    6: Crop(11, (0.27, 0.08, 0.71, 0.35)),
    7: Crop(12, (0.27, 0.09, 0.78, 0.42)),
    8: Crop(12, (0.27, 0.41, 0.82, 0.73)),
    9: Crop(13, (0.26, 0.23, 0.89, 0.63)),
    10: Crop(14, (0.08, 0.07, 0.94, 0.58)),
    11: Crop(15, (0.08, 0.52, 0.96, 0.80)),
    12: Crop(16, (0.27, 0.09, 0.71, 0.35)),
    13: Crop(16, (0.27, 0.63, 0.77, 0.89)),
    14: Crop(17, (0.07, 0.10, 0.95, 0.40)),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def extract_paper_figure(figure_number: int, output_path: Path, dpi: int = 220) -> Path:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("Install PyMuPDF to extract the paper figures") from exc
    spec = CROPS[figure_number]
    pdf_path = project_root() / "water-16-02668.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    try:
        page = document.load_page(spec.page - 1)
        x0, y0, x1, y1 = spec.box
        rect = page.rect
        clip = fitz.Rect(
            rect.x0 + x0 * rect.width,
            rect.y0 + y0 * rect.height,
            rect.x0 + x1 * rect.width,
            rect.y0 + y1 * rect.height,
        )
        scale = dpi / 72.0
        page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False).save(output_path)
    finally:
        document.close()
    return output_path


def save_comparison(figure_number: int, result_path: Path, output_path: Path, result_label: str) -> Path:
    paper_path = output_path.parent / f"figure{figure_number:02d}_paper.png"
    extract_paper_figure(figure_number, paper_path)
    paper_image = plt.imread(paper_path)
    result_image = plt.imread(result_path)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    axes[0].imshow(paper_image)
    axes[0].set_title(f"Paper Figure {figure_number}")
    axes[1].imshow(result_image)
    axes[1].set_title(result_label)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"Water 2024, 16, 2668 - Figure {figure_number}", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path
