"""
Módulo de Exportación y Generación de Reportes Analíticos (`ReportExporter`).

Decisiones técnicas:
- **Análisis tabular con `pandas` y `numpy`:** Convierte los bloques y regiones del
  `StructuredDocument` en un `pandas.DataFrame` estructurado para calcular métricas
  agregadas por categoría jerárquica (recuento de bloques, palabras, caracteres y
  confianza media/mínima/máxima).
- **Visualización estadística con `matplotlib` y `seaborn`:** Genera un gráfico dual
  en alta resolución (220 DPI) con estética limpia ("whitegrid", paleta editorial cálida/azul
  pizarra) que muestra:
  1. La distribución de tipos de bloques detectados en el documento.
  2. El nivel de confianza OCR por cada bloque secuencial.
- **Maquetación de PDF Profesional con `reportlab`:** Construye un documento PDF en `outputs/`
  que incluye:
  - Cabecera formal y resumen ejecutivo del documento analizado.
  - Tabla de métricas clave y desglose estadístico por tipo de bloque.
  - Gráfico analítico embebido.
  - Reconstrucción íntegra y coherente del texto extraído respetando su jerarquía
    (Títulos, Subtítulos, Metadatos Clave-Valor, Listas, Tablas estilizadas y Párrafos).
"""

import datetime
import os
from typing import Dict, List, Optional
from xml.sax.saxutils import escape as xml_escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.models import BlockType, StructuredDocument
from src.structure_engine import StructureEngine


BLOCK_LABELS_ES: Dict[str, str] = {
    BlockType.TITLE.value: "Título Principal (H1)",
    BlockType.SUBTITLE.value: "Subtítulo / Sección (H2)",
    BlockType.PARAGRAPH.value: "Párrafo Cohesivo",
    BlockType.LIST_ITEM.value: "Elemento de Lista",
    BlockType.KEY_VALUE.value: "Dato Clave-Valor",
    BlockType.TABLE.value: "Tabla Estructurada",
    BlockType.QUOTE_OR_NOTE.value: "Nota / Destacado",
}


class ReportExporter:
    """
    Gestiona la generación de DataFrames analíticos, gráficos estadísticos y
    la exportación del documento reconstruido a PDF, Markdown y Texto Plano.
    """

    @staticmethod
    def blocks_to_dataframe(doc: StructuredDocument) -> pd.DataFrame:
        """
        Convierte los bloques del documento reconstruido en un `pandas.DataFrame`
        para su inspección analítica y resumen estadístico.
        """
        if not doc.blocks:
            return pd.DataFrame(
                columns=[
                    "block_id",
                    "block_type",
                    "type_label",
                    "words",
                    "characters",
                    "confidence",
                    "y_top",
                    "text_preview",
                ]
            )

        rows = []
        for b in doc.blocks:
            words_count = len([w for w in b.text.split() if w.strip()])
            preview = b.text.replace("\n", " ")
            if len(preview) > 65:
                preview = preview[:62] + "..."
            rows.append(
                {
                    "block_id": b.block_id,
                    "block_type": b.block_type.value,
                    "type_label": BLOCK_LABELS_ES.get(b.block_type.value, b.block_type.value),
                    "words": words_count,
                    "characters": len(b.text),
                    "confidence": round(float(b.confidence) * 100.0, 2),
                    "y_top": round(float(b.bbox[1]), 1),
                    "text_preview": preview,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def generate_analytics_chart(cls, doc: StructuredDocument, output_chart_path: str) -> Optional[str]:
        """
        Genera un gráfico estadístico compuesto (distribución por tipo de bloque +
        perfil de confianza OCR por bloque) usando `matplotlib` y `seaborn`.

        Args:
            doc: Documento estructurado con métricas.
            output_chart_path: Ruta donde se guardará la imagen PNG del gráfico.

        Returns:
            Ruta del archivo PNG generado o `None` si no hay bloques.
        """
        df = cls.blocks_to_dataframe(doc)
        if df.empty:
            return None

        os.makedirs(os.path.dirname(os.path.abspath(output_chart_path)), exist_ok=True)

        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=200)
        fig.patch.set_facecolor("#FFFFFF")

        # Panel 1: Conteo de bloques por categoría jerárquica
        type_summary = (
            df.groupby("type_label", as_index=False)
            .agg(count=("block_id", "count"), words=("words", "sum"))
            .sort_values("count", ascending=True)
        )
        palette_colors = ["#0F766E", "#1D4ED8", "#475569", "#B45309", "#6D28D9", "#047857"]
        bars = ax1.barh(
            type_summary["type_label"],
            type_summary["count"],
            color=palette_colors[: len(type_summary)],
            height=0.58,
            edgecolor="#1E293B",
            linewidth=0.6,
        )
        ax1.set_title("Composición Estructural por Tipo de Bloque", fontsize=11, fontweight="bold", color="#0F172A", pad=10)
        ax1.set_xlabel("Cantidad de Bloques", fontsize=9.5, color="#334155")
        ax1.tick_params(axis="both", labelsize=8.5)
        for bar in bars:
            w = bar.get_width()
            ax1.text(
                w + 0.1,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(w)}",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#0F172A",
            )
        ax1.set_xlim(0, max(type_summary["count"]) * 1.22 + 0.5)

        # Panel 2: Confianza OCR (%) a lo largo de la secuencia de bloques
        sns.lineplot(
            data=df,
            x="block_id",
            y="confidence",
            marker="o",
            color="#0F766E",
            linewidth=2.0,
            markersize=6,
            ax=ax2,
        )
        mean_conf = float(df["confidence"].mean())
        ax2.axhline(
            mean_conf,
            color="#D97706",
            linestyle="--",
            linewidth=1.3,
            label=f"Media: {mean_conf:.1f}%",
        )
        ax2.set_title("Perfil de Confianza OCR por Bloque Secuencial", fontsize=11, fontweight="bold", color="#0F172A", pad=10)
        ax2.set_xlabel("ID de Bloque (Orden de Lectura)", fontsize=9.5, color="#334155")
        ax2.set_ylabel("Confianza OCR (%)", fontsize=9.5, color="#334155")
        ax2.set_ylim(max(0.0, float(df["confidence"].min()) - 10.0), 103.0)
        ax2.tick_params(axis="both", labelsize=8.5)
        ax2.legend(loc="lower right", frameon=True, fontsize=8.5)

        plt.tight_layout(pad=1.8)
        fig.savefig(output_chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_chart_path

    @classmethod
    def export_markdown(cls, doc: StructuredDocument, output_path: str) -> str:
        """Guarda el documento reconstruido en formato Markdown (.md)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        md_content = StructureEngine.format_as_markdown(doc)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        return output_path

    @classmethod
    def export_plain_text(cls, doc: StructuredDocument, output_path: str) -> str:
        """Guarda el documento reconstruido en formato Texto Plano (.txt)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        txt_content = StructureEngine.format_as_plain_text(doc)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        return output_path

    @classmethod
    def export_pdf_report(
        cls,
        doc: StructuredDocument,
        output_pdf_path: str,
        include_analytics: bool = True,
    ) -> str:
        """
        Genera un reporte PDF profesional y maquetado en `output_pdf_path` con:
        1. Cabecera institucional y metadatos de procesamiento.
        2. Resumen ejecutivo y tabla de métricas clave de extracción.
        3. Gráfico estadístico de composición estructural y confianza OCR.
        4. Contenido textual íntegro reconstruido con su jerarquía visual.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

        pdf = SimpleDocTemplate(
            output_pdf_path,
            pagesize=A4,
            leftMargin=1.8 * cm,
            rightMargin=1.8 * cm,
            topMargin=1.8 * cm,
            bottomMargin=1.8 * cm,
            title="Reporte de Extracción y Estructuración Documental - DocuStruct Studio",
            author="DocuStruct Studio",
        )

        base_styles = getSampleStyleSheet()
        style_report_header = ParagraphStyle(
            "ReportHeader",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=4,
        )
        style_report_sub = ParagraphStyle(
            "ReportSub",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#475569"),
            spaceAfter=10,
        )
        style_section_title = ParagraphStyle(
            "SectionTitle",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#0F766E"),
            spaceBefore=12,
            spaceAfter=6,
        )
        style_doc_h1 = ParagraphStyle(
            "DocH1",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14.5,
            leading=18,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=10,
            spaceAfter=6,
        )
        style_doc_h2 = ParagraphStyle(
            "DocH2",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#1E3A8A"),
            spaceBefore=8,
            spaceAfter=4,
        )
        style_doc_body = ParagraphStyle(
            "DocBody",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14.5,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=3,
            spaceAfter=6,
        )
        style_doc_list = ParagraphStyle(
            "DocList",
            parent=style_doc_body,
            leftIndent=14,
            firstLineIndent=-8,
            spaceBefore=2,
            spaceAfter=3,
        )
        style_doc_kv = ParagraphStyle(
            "DocKV",
            parent=style_doc_body,
            leftIndent=8,
            spaceBefore=2,
            spaceAfter=2,
        )
        style_doc_note = ParagraphStyle(
            "DocNote",
            parent=style_doc_body,
            fontName="Helvetica-Oblique",
            leftIndent=12,
            textColor=colors.HexColor("#881337"),
            backColor=colors.HexColor("#FFF1F2"),
            borderPadding=5,
            spaceBefore=5,
            spaceAfter=6,
        )
        style_table_cell = ParagraphStyle(
            "TableCell",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=11.5,
            textColor=colors.HexColor("#1E293B"),
        )
        style_table_header = ParagraphStyle(
            "TableHeader",
            parent=style_table_cell,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0F172A"),
        )

        story = []

        # 1. Cabecera Formal del Reporte
        now_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        src_name = os.path.basename(doc.source_path) if doc.source_path else "Imagen en memoria"
        story.append(Paragraph("DocuStruct Studio — Reporte de Extracción y Estructuración", style_report_header))
        story.append(
            Paragraph(
                f"<b>Archivo origen:</b> {xml_escape(src_name)} &nbsp;|&nbsp; "
                f"<b>Dimensiones:</b> {doc.image_size[0]}x{doc.image_size[1]} px &nbsp;|&nbsp; "
                f"<b>Generado:</b> {now_str}",
                style_report_sub,
            )
        )
        story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#CBD5E1"), spaceAfter=10))

        # 2. Resumen Ejecutivo y Métricas Clave (si include_analytics está activo)
        if include_analytics:
            story.append(Paragraph("1. Resumen Ejecutivo y Métricas de Calidad OCR", style_section_title))
            an = doc.analytics
            summary_text = (
                f"El motor de reconstrucción documental procesó un total de <b>{an.total_regions}</b> regiones "
                f"ópticas crudas en <b>{an.processing_time_ms:.0f} ms</b>, consolidándolas de manera coherente en "
                f"<b>{an.total_blocks}</b> bloques semánticos jerárquicos (<b>{an.total_words}</b> palabras y "
                f"<b>{an.total_characters}</b> caracteres). La confianza media de reconocimiento se situó en "
                f"<b>{an.mean_confidence * 100:.1f}%</b> (mediana: {an.median_confidence * 100:.1f}%)."
            )
            story.append(Paragraph(summary_text, style_doc_body))
            story.append(Spacer(1, 6))

            kpi_data = [
                ["Métrica de Evaluación", "Valor Obtenido", "Métrica de Evaluación", "Valor Obtenido"],
                [
                    "Regiones OCR Crudas",
                    str(an.total_regions),
                    "Confianza Media OCR",
                    f"{an.mean_confidence * 100:.1f}%",
                ],
                [
                    "Bloques Coherentes",
                    str(an.total_blocks),
                    "Rango de Confianza (Min - Max)",
                    f"{an.min_confidence * 100:.1f}% - {an.max_confidence * 100:.1f}%",
                ],
                [
                    "Total de Palabras / Caracteres",
                    f"{an.total_words} pal. / {an.total_characters} car.",
                    "Tiempo de Lectura Estimado",
                    f"~{an.estimated_reading_time_sec} seg",
                ],
            ]
            kpi_table = Table(kpi_data, colWidths=[4.6 * cm, 3.8 * cm, 4.8 * cm, 4.0 * cm])
            kpi_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#1E293B")),
                        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                        ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("PADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(kpi_table)
            story.append(Spacer(1, 10))

            # Gráfico estadístico embebido
            chart_path = os.path.splitext(output_pdf_path)[0] + "_analytics_chart.png"
            generated_chart = cls.generate_analytics_chart(doc, chart_path)
            if generated_chart and os.path.exists(generated_chart):
                story.append(RLImage(generated_chart, width=17.0 * cm, height=6.5 * cm))
                story.append(Spacer(1, 10))

            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E2E8F0"), spaceAfter=8))

        # 3. Contenido del Documento Reconstruido y Organizado
        section_num = "2." if include_analytics else "1."
        story.append(
            Paragraph(f"{section_num} Documento Extraído y Estructurado Coherentemente", style_section_title)
        )

        if not doc.blocks:
            story.append(Paragraph("<i>No se detectaron bloques de texto en la imagen.</i>", style_doc_body))
        else:
            for block in doc.blocks:
                escaped = xml_escape(block.text)
                if block.block_type == BlockType.TITLE:
                    story.append(Paragraph(escaped, style_doc_h1))
                elif block.block_type == BlockType.SUBTITLE:
                    story.append(Paragraph(escaped, style_doc_h2))
                elif block.block_type == BlockType.KEY_VALUE and block.key_value:
                    k_esc = xml_escape(block.key_value[0])
                    v_esc = xml_escape(block.key_value[1])
                    story.append(Paragraph(f"<b>{k_esc}:</b> {v_esc}", style_doc_kv))
                elif block.block_type == BlockType.LIST_ITEM:
                    story.append(Paragraph(f"&bull; {escaped}", style_doc_list))
                elif block.block_type == BlockType.QUOTE_OR_NOTE:
                    story.append(Paragraph(escaped, style_doc_note))
                elif block.block_type == BlockType.TABLE and block.table_data:
                    matrix = block.table_data
                    if matrix:
                        num_cols = max(1, len(matrix[0]))
                        col_w = (17.2 * cm) / float(num_cols)
                        rl_rows = []
                        for r_idx, row in enumerate(matrix):
                            cell_style = style_table_header if r_idx == 0 else style_table_cell
                            rl_rows.append([Paragraph(xml_escape(c), cell_style) for c in row])

                        doc_table = Table(rl_rows, colWidths=[col_w] * num_cols)
                        doc_table.setStyle(
                            TableStyle(
                                [
                                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                    ("PADDING", (0, 0), (-1, -1), 5),
                                ]
                            )
                        )
                        story.append(Spacer(1, 4))
                        story.append(KeepTogether([doc_table]))
                        story.append(Spacer(1, 6))
                else:
                    story.append(Paragraph(escaped, style_doc_body))

        pdf.build(story)
        return output_pdf_path

