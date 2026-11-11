"""
Punto de entrada principal de DocuStruct Studio.
Permite tanto el inicio de la interfaz gráfica PySide6 como la ejecución de un modo
batch/CLI (`--batch-demo`) para generar automáticamente los artefactos y reportes en `outputs/`.
"""

import argparse
import os
import sys

from PySide6.QtWidgets import QApplication

from src.exporter import ReportExporter
from src.gui import MainWindow
from src.icon_generator import ensure_app_icon, generate_sample_document_image
from src.ocr_engine import OCREngine
from src.structure_engine import StructureEngine


def get_base_dir() -> str:
    """
    Obtiene el directorio raíz del proyecto tanto en ejecución desde código fuente
    como cuando se ejecuta empaquetado dentro de un ejecutable de PyInstaller.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def run_batch_demo(base_dir: str) -> int:
    """
    Ejecuta el flujo completo de extracción OCR, reconstrucción estructural coherente y
    exportación de reportes (PDF maquetado, Markdown y TXT) sobre la imagen de muestra.
    """
    assets_dir = os.path.join(base_dir, "assets")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    ensure_app_icon(assets_dir)
    sample_img = generate_sample_document_image(os.path.join(assets_dir, "sample_document.png"))

    ocr = OCREngine(confidence_threshold=0.35)
    struct = StructureEngine()

    regions, img_size, elapsed_ms = ocr.extract_regions(sample_img, enhance_contrast=False, denoise=False)
    doc = struct.build_structured_document(
        raw_regions=regions,
        source_path=sample_img,
        image_size=img_size,
        processing_time_ms=elapsed_ms,
    )

    pdf_path = os.path.join(outputs_dir, "reporte_extraccion_estructurada.pdf")
    md_path = os.path.join(outputs_dir, "documento_estructurado.md")
    txt_path = os.path.join(outputs_dir, "documento_estructurado.txt")

    ReportExporter.export_pdf_report(doc, pdf_path, include_analytics=True)
    ReportExporter.export_markdown(doc, md_path)
    ReportExporter.export_plain_text(doc, txt_path)

    print(f"[OK] Bloques reconstruidos: {doc.analytics.total_blocks}")
    print(f"[OK] Reporte PDF generado : {pdf_path}")
    print(f"[OK] Archivo Markdown     : {md_path}")
    print(f"[OK] Archivo Texto Plano  : {txt_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DocuStruct Studio — Extracción Óptica y Organización Estructural de Texto"
    )
    parser.add_argument(
        "--batch-demo",
        action="store_true",
        help="Procesa el documento de muestra y genera los reportes en outputs/ sin abrir la ventana.",
    )
    args = parser.parse_args()

    base_dir = get_base_dir()
    if args.batch_demo:
        return run_batch_demo(base_dir)

    app = QApplication(sys.argv)
    app.setApplicationName("DocuStruct Studio")
    app.setOrganizationName("DocuStruct Studio")

    window = MainWindow(base_dir=base_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
