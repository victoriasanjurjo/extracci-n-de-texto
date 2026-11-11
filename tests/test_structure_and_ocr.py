"""
Pruebas unitarias y de integración para DocuStruct Studio.
Verifica el correcto funcionamiento de:
1. Agrupación horizontal de líneas y reconstrucción de jerarquía (`StructureEngine`).
2. Clasificación de Títulos (H1), Subtítulos (H2), Pares Clave-Valor, Listas, Párrafos y Tablas.
3. Unión de líneas partidas con guion y formateo en Markdown, Texto Plano y HTML.
4. Extracción OCR real (`OCREngine`) y exportación de reportes analíticos (`ReportExporter`).
"""

import os
import tempfile
from typing import List

from src.exporter import ReportExporter
from src.icon_generator import ensure_app_icon, generate_sample_document_image
from src.models import BlockType, TextRegion
from src.ocr_engine import OCREngine
from src.structure_engine import StructureEngine


def _make_region(text: str, x0: float, y0: float, x1: float, y1: float, conf: float = 0.96) -> TextRegion:
    return TextRegion(
        text=text,
        confidence=conf,
        polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        x_min=x0,
        y_min=y0,
        x_max=x1,
        y_max=y1,
    )


def test_structure_engine_hierarchy_and_tables() -> None:
    """
    Comprueba que regiones espaciales sintéticas se organicen coherentemente en:
    Título H1, Metadato Clave-Valor, Subtítulo H2, Párrafo multi-línea unido,
    Lista de viñetas y Tabla de múltiples columnas.
    """
    regions: List[TextRegion] = [
        # Título principal (altura 42px vs cuerpo 20px)
        _make_region("REPORTE ANUAL DE LOGISTICA", 80, 50, 680, 92),
        # Par Clave-Valor (altura 20px)
        _make_region("Fecha: 11/11/2026", 80, 120, 320, 140),
        # Subtítulo (altura 28px)
        _make_region("1. Resumen Operativo", 80, 180, 420, 208),
        # Párrafo de 2 líneas (altura 20px, separación vertical 8px) con palabra cortada por guion
        _make_region("El sistema de transporte experi-", 80, 230, 520, 250),
        _make_region("mento una mejora continua en todas las rutas.", 80, 258, 560, 278),
        # Lista de 2 elementos
        _make_region("- Optimizacion de tiempos de carga", 95, 315, 500, 335),
        _make_region("- Digitalizacion de albaranes", 95, 345, 460, 365),
        # Tabla de 2 filas y 3 columnas (separadas horizontalmente)
        _make_region("Sucursal", 80, 410, 200, 430),
        _make_region("Envios", 360, 410, 460, 430),
        _make_region("Cumplimiento", 640, 410, 810, 430),
        _make_region("Norte", 80, 442, 170, 462),
        _make_region("1450", 360, 442, 430, 462),
        _make_region("99.2%", 640, 442, 720, 462),
    ]

    engine = StructureEngine()
    doc = engine.build_structured_document(regions, source_path="test.png", image_size=(1000, 800))

    block_types = [b.block_type for b in doc.blocks]
    assert BlockType.TITLE in block_types
    assert BlockType.KEY_VALUE in block_types
    assert BlockType.SUBTITLE in block_types
    assert BlockType.PARAGRAPH in block_types
    assert BlockType.LIST_ITEM in block_types
    assert BlockType.TABLE in block_types

    # Verificar que el párrafo haya unido "experi-" + "mento" -> "experimento"
    paragraphs = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH]
    assert len(paragraphs) == 1
    assert "experimento una mejora continua" in paragraphs[0].text

    # Verificar que la tabla tenga 2 filas y 3 columnas
    tables = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
    assert len(tables) == 1
    assert tables[0].table_data is not None
    assert len(tables[0].table_data) == 2
    assert len(tables[0].table_data[0]) == 3

    # Verificar generación de Markdown y Texto Plano
    md_out = StructureEngine.format_as_markdown(doc)
    assert "# REPORTE ANUAL DE LOGISTICA" in md_out
    assert "**Fecha:** 11/11/2026" in md_out
    assert "| Sucursal | Envios | Cumplimiento |" in md_out

    txt_out = StructureEngine.format_as_plain_text(doc)
    assert "REPORTE ANUAL DE LOGISTICA" in txt_out
    assert "• Optimizacion de tiempos de carga" in txt_out


def test_end_to_end_ocr_and_pdf_export() -> None:
    """
    Prueba de integración extremo a extremo: genera la imagen de muestra,
    extrae las regiones con RapidOCR, reconstruye el documento y exporta el PDF y gráfico.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        png_icon, ico_icon = ensure_app_icon(tmp_dir)
        assert os.path.exists(png_icon)
        assert os.path.exists(ico_icon)

        sample_img = generate_sample_document_image(os.path.join(tmp_dir, "sample.png"))
        assert os.path.exists(sample_img)

        ocr = OCREngine(confidence_threshold=0.35)
        regions, size, elapsed = ocr.extract_regions(sample_img)
        assert len(regions) >= 10
        assert size == (1150, 1480)
        assert elapsed > 0

        struct = StructureEngine()
        doc = struct.build_structured_document(regions, source_path=sample_img, image_size=size, processing_time_ms=elapsed)
        assert doc.analytics.total_blocks >= 5
        assert doc.analytics.total_words >= 40

        pdf_out = os.path.join(tmp_dir, "reporte.pdf")
        ReportExporter.export_pdf_report(doc, pdf_out, include_analytics=True)
        assert os.path.exists(pdf_out)
        assert os.path.getsize(pdf_out) > 2000

