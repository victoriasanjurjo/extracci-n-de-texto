"""
Motor de Reconstrucción Estructural y Organización Coherente de Texto (`StructureEngine`).

Decisiones técnicas y algorítmicas:
Un motor OCR convencional devuelve una lista desordenada o meramente secuencial de fragmentos
de texto aislados, perdiendo la jerarquía visual del documento (encabezados, párrafos continuos,
columnas de tablas, listas de viñetas y metadatos clave-valor).

Para transformar las cajas crudas (`TextRegion`) en un documento coherente y estructurado
(`StructuredDocument`), este módulo implementa un pipeline determinista de 5 fases:

1. **Agrupación en Líneas Horizontales (`_group_regions_into_lines`):**
   - Ordena las regiones por su coordenada vertical superior (`y_min`).
   - Agrupa dos regiones en la misma línea si su solapamiento vertical supera un umbral relativo
     a la menor altura de ambas o si la distancia entre sus centros verticales es menor a
     `line_merge_vertical_tolerance * altura_media`.
   - Dentro de cada línea, ordena las regiones de izquierda a derecha (`x_min`) y calcula las
     brechas horizontales (`gaps`) entre fragmentos consecutivos.
   - Fusiona fragmentos adyacentes cuya brecha sea pequeña (palabras de una misma oración
     separadas por el detector), pero conserva los segmentos separados cuando existe una brecha
     amplia (indicador de columnas tabulares).

2. **Detección de Bloques Tabulares (`_detect_and_group_tables`):**
   - Identifica secuencias de líneas que contienen múltiples celdas separadas por brechas
     horizontales significativas (`gap > table_column_gap_ratio * char_width`) o líneas
     consecutivas cuyos fragmentos se alinean verticalmente en 2 o más columnas.
   - Reconstruye una cuadrícula (`List[List[str]]`) proyectando las coordenadas X de las celdas
     sobre clústeres de columnas para que cada celda ocupe la columna correcta incluso si alguna
     fila tiene celdas vacías.

3. **Agrupación en Párrafos y Bloques Semánticos (`_segment_lines_into_blocks`):**
   - Calcula la altura tipográfica mediana del cuerpo del documento (`median_body_height`) y el
     interlineado típico (`median_line_gap`).
   - Decide cuándo cortar un bloque e iniciar uno nuevo basándose en:
     a) Saltos verticales mayores a `paragraph_gap_multiplier * altura_linea`.
     b) Cambios abruptos en la altura tipográfica (ej. pasar de un título a un párrafo).
     c) Patrones sintácticos de inicio de lista (`•`, `-`, `*`, `1.`, `a)`, etc.).
     d) Patrones de metadatos clave-valor cortos (`Clave: Valor`).
     e) Desplazamientos de sangría izquierda significativos o centrado de títulos.

4. **Clasificación Jerárquica (`_classify_block`):**
   - Asigna `BlockType.TITLE` (H1) y `BlockType.SUBTITLE` (H2/H3) combinando la proporción de
     altura respecto al cuerpo (`height_ratio`), posición superior o aislamiento vertical,
     longitud moderada y ausencia de punto final de oración.
   - Clasifica `BlockType.LIST_ITEM`, `BlockType.KEY_VALUE`, `BlockType.TABLE`,
     `BlockType.QUOTE_OR_NOTE` y `BlockType.PARAGRAPH`.
   - En los párrafos multi-línea, une inteligentemente palabras cortadas con guion al final de
     línea (`-\n`) y recompone el flujo de lectura continuo sin saltos de línea artificiales.

5. **Formateo Coherente y Cálculo Analítico (`format_as_markdown`, `format_as_plain_text`, `_compute_analytics`):**
   - Genera representaciones limpias en Markdown (con tablas GFM, encabezados `#`/`##`, listas y
     pares en negrita), texto plano indentado y HTML enriquecido para vista previa interactiva.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import (
    BlockType,
    DocumentAnalytics,
    DocumentBlock,
    ReconstructedLine,
    StructuredDocument,
    TextRegion,
)

# Expresiones regulares precompiladas para clasificación sintáctica y semántica
BULLET_PATTERN = re.compile(
    r"^\s*(?:[\u2022\u2023\u25E6\u2043\u2219\*\-\+]|(?:\d{1,3}|[a-zA-Z])[\.\)])\s+(.+)$"
)
KEY_VALUE_PATTERN = re.compile(
    r"^\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\s/._\-]{2,38})\s*:\s*(.+)$"
)
NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*(?:(?:\d+\.)+\d*|[IVXLCDM]+\.)\s+[A-ZÁÉÍÓÚÜÑ]"
)
NOTE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:nota|importante|advertencia|observaci[oó]n|aviso|note|warning|tip)\b[:\s\-]*",
    re.IGNORECASE,
)


class StructureEngine:
    """
    Motor encargado de reconstruir la estructura lógica, jerárquica y tabular
    de un documento a partir de regiones crudas de OCR.
    """

    def __init__(
        self,
        heading_height_ratio: float = 1.26,
        sub_heading_height_ratio: float = 1.11,
        line_merge_vertical_tolerance: float = 0.55,
        paragraph_gap_multiplier: float = 1.40,
        table_column_gap_ratio: float = 2.4,
        detect_tables: bool = True,
        detect_lists: bool = True,
        detect_key_value_pairs: bool = True,
    ) -> None:
        self.heading_height_ratio = heading_height_ratio
        self.sub_heading_height_ratio = sub_heading_height_ratio
        self.line_merge_vertical_tolerance = line_merge_vertical_tolerance
        self.paragraph_gap_multiplier = paragraph_gap_multiplier
        self.table_column_gap_ratio = table_column_gap_ratio
        self.detect_tables = detect_tables
        self.detect_lists = detect_lists
        self.detect_key_value_pairs = detect_key_value_pairs

    def build_structured_document(
        self,
        raw_regions: List[TextRegion],
        source_path: str = "",
        image_size: Tuple[int, int] = (1000, 1400),
        processing_time_ms: float = 0.0,
    ) -> StructuredDocument:
        """
        Orquesta la reconstrucción completa del documento a partir de las regiones OCR.

        Args:
            raw_regions: Lista de `TextRegion` detectadas en la imagen.
            source_path: Ruta de la imagen procesada.
            image_size: Tupla `(ancho, alto)` en píxeles.
            processing_time_ms: Tiempo de ejecución del OCR en milisegundos.

        Returns:
            Instancia completa de `StructuredDocument`.
        """
        if not raw_regions:
            empty_analytics = DocumentAnalytics(processing_time_ms=processing_time_ms)
            return StructuredDocument(
                source_path=source_path,
                image_size=image_size,
                raw_regions=[],
                reconstructed_lines=[],
                blocks=[],
                analytics=empty_analytics,
            )

        # 1. Reconstruir líneas horizontales y fusionar fragmentos cercanos
        lines = self._group_regions_into_lines(raw_regions)

        # 2. Calcular estadísticos globales de referencia tipográfica
        heights = np.array([line.height for line in lines], dtype=float)
        # Usamos el percentil 45 como estimador robusto del tamaño de fuente del cuerpo
        # para evitar que documentos con muchos títulos sesguen la mediana hacia arriba.
        body_height = float(np.percentile(heights, 45)) if len(heights) > 0 else 20.0
        body_height = max(8.0, body_height)

        doc_x_min = min(line.x_min for line in lines)
        doc_x_max = max(line.x_max for line in lines)
        doc_width = max(100.0, doc_x_max - doc_x_min)

        # 3. Segmentar líneas en bloques (detectando tablas, listas, encabezados y párrafos)
        blocks = self._segment_lines_into_blocks(
            lines=lines,
            body_height=body_height,
            doc_x_min=doc_x_min,
            doc_width=doc_width,
        )

        # 4. Calcular analíticas detalladas del documento
        analytics = self._compute_analytics(
            raw_regions=raw_regions,
            blocks=blocks,
            processing_time_ms=processing_time_ms,
        )

        return StructuredDocument(
            source_path=source_path,
            image_size=image_size,
            raw_regions=raw_regions,
            reconstructed_lines=lines,
            blocks=blocks,
            analytics=analytics,
        )

    def _group_regions_into_lines(self, regions: List[TextRegion]) -> List[ReconstructedLine]:
        """
        Agrupa regiones de texto que se encuentran en la misma banda horizontal
        y fusiona aquellas que pertenecen a la misma frase o celda.
        """
        sorted_regs = sorted(regions, key=lambda r: (r.center_y, r.x_min))
        raw_line_clusters: List[List[TextRegion]] = []

        for reg in sorted_regs:
            placed = False
            for cluster in raw_line_clusters:
                # Comparamos con el promedio de la línea en formación
                cluster_y_min = min(r.y_min for r in cluster)
                cluster_y_max = max(r.y_max for r in cluster)
                cluster_center_y = sum(r.center_y for r in cluster) / len(cluster)
                cluster_avg_h = sum(r.height for r in cluster) / len(cluster)

                overlap_y = max(0.0, min(cluster_y_max, reg.y_max) - max(cluster_y_min, reg.y_min))
                min_h = min(cluster_avg_h, reg.height)
                vertical_dist = abs(reg.center_y - cluster_center_y)

                # Criterio de pertenencia a la misma línea horizontal
                if (overlap_y / max(1.0, min_h) >= 0.42) or (
                    vertical_dist <= self.line_merge_vertical_tolerance * min_h
                ):
                    cluster.append(reg)
                    placed = True
                    break

            if not placed:
                raw_line_clusters.append([reg])

        reconstructed_lines: List[ReconstructedLine] = []
        for cluster in raw_line_clusters:
            # Ordenar de izquierda a derecha dentro de la línea
            cluster.sort(key=lambda r: r.x_min)

            # Fusionar regiones que están separadas sólo por un espacio normal de palabra,
            # manteniendo separadas aquellas con una brecha de columna tabular.
            merged_cells: List[TextRegion] = [cluster[0]]
            gaps: List[float] = []

            for next_reg in cluster[1:]:
                prev_reg = merged_cells[-1]
                gap = next_reg.x_min - prev_reg.x_max
                avg_char_w = (prev_reg.char_width + next_reg.char_width) / 2.0
                avg_h = (prev_reg.height + next_reg.height) / 2.0
                col_gap_threshold = max(
                    avg_char_w * self.table_column_gap_ratio,
                    avg_h * 1.15,
                )

                # Si la línea es una frase larga con punto intermedio (párrafo partido por el detector)
                # y el gap es menor a 3.2 veces la altura de línea, es texto continuo y no tabla
                total_words_in_pair = len(prev_reg.text.split()) + len(next_reg.text.split())
                is_sentence_continuation = (
                    total_words_in_pair >= 8 and gap < avg_h * 3.2
                )

                if gap < col_gap_threshold or is_sentence_continuation:
                    # Es continuación de la misma frase/celda
                    combined_text = f"{prev_reg.text.strip()} {next_reg.text.strip()}"
                    combined_conf = (prev_reg.confidence + next_reg.confidence) / 2.0
                    new_x_min = min(prev_reg.x_min, next_reg.x_min)
                    new_y_min = min(prev_reg.y_min, next_reg.y_min)
                    new_x_max = max(prev_reg.x_max, next_reg.x_max)
                    new_y_max = max(prev_reg.y_max, next_reg.y_max)
                    merged_cells[-1] = TextRegion(
                        text=combined_text,
                        confidence=combined_conf,
                        polygon=[
                            (new_x_min, new_y_min),
                            (new_x_max, new_y_min),
                            (new_x_max, new_y_max),
                            (new_x_min, new_y_max),
                        ],
                        x_min=new_x_min,
                        y_min=new_y_min,
                        x_max=new_x_max,
                        y_max=new_y_max,
                    )
                else:
                    merged_cells.append(next_reg)
                    gaps.append(gap)

            reconstructed_lines.append(
                ReconstructedLine(
                    regions=merged_cells,
                    x_min=min(r.x_min for r in merged_cells),
                    y_min=min(r.y_min for r in merged_cells),
                    x_max=max(r.x_max for r in merged_cells),
                    y_max=max(r.y_max for r in merged_cells),
                    gaps=gaps,
                )
            )

        # Ordenar las líneas finales de arriba hacia abajo
        reconstructed_lines.sort(key=lambda l: l.y_min)
        return reconstructed_lines

    def _segment_lines_into_blocks(
        self,
        lines: List[ReconstructedLine],
        body_height: float,
        doc_x_min: float,
        doc_width: float,
    ) -> List[DocumentBlock]:
        """
        Recorre las líneas reconstruidas y las agrupa en bloques coherentes:
        tablas, títulos, subtítulos, elementos de lista, pares clave-valor y párrafos.
        """
        blocks: List[DocumentBlock] = []
        n_lines = len(lines)
        idx = 0
        block_counter = 1

        while idx < n_lines:
            current_line = lines[idx]

            # 1. ¿Comienza aquí un bloque tabular (2 o más filas consecutivas multi-celda)?
            if self.detect_tables and len(current_line.regions) >= 2:
                table_lines: List[ReconstructedLine] = [current_line]
                lookahead = idx + 1
                while lookahead < n_lines:
                    candidate = lines[lookahead]
                    prev_in_table = table_lines[-1]
                    v_gap = candidate.y_min - prev_in_table.y_max

                    # Permitir que una fila de tabla continúe si tiene >= 2 columnas y está a una distancia razonable
                    if len(candidate.regions) >= 2 and v_gap <= body_height * 2.8:
                        table_lines.append(candidate)
                        lookahead += 1
                    else:
                        break

                if len(table_lines) >= 2:
                    table_matrix = self._build_table_matrix(table_lines)
                    table_text = self._table_matrix_to_text(table_matrix)
                    avg_conf = sum(l.avg_confidence for l in table_lines) / len(table_lines)
                    bbox = (
                        min(l.x_min for l in table_lines),
                        min(l.y_min for l in table_lines),
                        max(l.x_max for l in table_lines),
                        max(l.y_max for l in table_lines),
                    )
                    blocks.append(
                        DocumentBlock(
                            block_id=block_counter,
                            block_type=BlockType.TABLE,
                            text=table_text,
                            lines=table_lines,
                            confidence=avg_conf,
                            bbox=bbox,
                            table_data=table_matrix,
                            level=1,
                        )
                    )
                    block_counter += 1
                    idx = lookahead
                    continue

            # 2. Evaluar si la línea actual es un elemento unitario (Título, Subtítulo, Lista o Clave-Valor)
            line_type, kv_pair, heading_level, cleaned_text = self._inspect_single_line(
                line=current_line,
                body_height=body_height,
                doc_x_min=doc_x_min,
                doc_width=doc_width,
                is_first_line=(idx == 0),
            )

            if line_type in (BlockType.TITLE, BlockType.SUBTITLE, BlockType.KEY_VALUE):
                blocks.append(
                    DocumentBlock(
                        block_id=block_counter,
                        block_type=line_type,
                        text=cleaned_text,
                        lines=[current_line],
                        confidence=current_line.avg_confidence,
                        bbox=(
                            current_line.x_min,
                            current_line.y_min,
                            current_line.x_max,
                            current_line.y_max,
                        ),
                        key_value=kv_pair,
                        level=heading_level,
                    )
                )
                block_counter += 1
                idx += 1
                continue

            if line_type == BlockType.LIST_ITEM:
                # Agrupar líneas de continuación de la misma viñeta si están indentadas y muy próximas
                item_lines = [current_line]
                item_texts = [cleaned_text]
                lookahead = idx + 1
                while lookahead < n_lines:
                    next_line = lines[lookahead]
                    v_gap = next_line.y_min - item_lines[-1].y_max
                    next_type, _, _, _ = self._inspect_single_line(
                        line=next_line,
                        body_height=body_height,
                        doc_x_min=doc_x_min,
                        doc_width=doc_width,
                        is_first_line=False,
                    )
                    if (
                        next_type == BlockType.PARAGRAPH
                        and len(next_line.regions) == 1
                        and v_gap <= body_height * 0.95
                        and next_line.x_min >= current_line.x_min - body_height
                    ):
                        item_lines.append(next_line)
                        item_texts.append(next_line.text)
                        lookahead += 1
                    else:
                        break

                merged_item_text = self._join_wrapped_lines(item_texts)
                avg_conf = sum(l.avg_confidence for l in item_lines) / len(item_lines)
                bbox = (
                    min(l.x_min for l in item_lines),
                    min(l.y_min for l in item_lines),
                    max(l.x_max for l in item_lines),
                    max(l.y_max for l in item_lines),
                )
                blocks.append(
                    DocumentBlock(
                        block_id=block_counter,
                        block_type=BlockType.LIST_ITEM,
                        text=merged_item_text,
                        lines=item_lines,
                        confidence=avg_conf,
                        bbox=bbox,
                        level=1,
                    )
                )
                block_counter += 1
                idx = lookahead
                continue

            # 3. Agrupación de Párrafo o Nota/Cita multi-línea
            para_lines: List[ReconstructedLine] = [current_line]
            lookahead = idx + 1
            while lookahead < n_lines:
                next_line = lines[lookahead]
                prev_line = para_lines[-1]
                v_gap = next_line.y_min - prev_line.y_max
                avg_line_h = (prev_line.height + next_line.height) / 2.0

                # Si la siguiente línea es comienzo de tabla, encabezado, lista o clave-valor, cerramos párrafo
                if self.detect_tables and len(next_line.regions) >= 2:
                    if lookahead + 1 < n_lines and len(lines[lookahead + 1].regions) >= 2:
                        break

                next_type, _, _, _ = self._inspect_single_line(
                    line=next_line,
                    body_height=body_height,
                    doc_x_min=doc_x_min,
                    doc_width=doc_width,
                    is_first_line=False,
                )
                if next_type != BlockType.PARAGRAPH:
                    break

                # Si el salto vertical supera el umbral de párrafo nuevo, cortamos
                if v_gap > avg_line_h * self.paragraph_gap_multiplier:
                    break

                # Si hay un cambio fuerte de tamaño de fuente (> 20%), cortamos
                if abs(next_line.height - prev_line.height) / max(1.0, prev_line.height) > 0.22:
                    break

                para_lines.append(next_line)
                lookahead += 1

            raw_texts = [l.text for l in para_lines]
            merged_para_text = self._join_wrapped_lines(raw_texts)
            final_block_type = (
                BlockType.QUOTE_OR_NOTE
                if NOTE_PREFIX_PATTERN.match(merged_para_text)
                else BlockType.PARAGRAPH
            )
            avg_conf = sum(l.avg_confidence for l in para_lines) / len(para_lines)
            bbox = (
                min(l.x_min for l in para_lines),
                min(l.y_min for l in para_lines),
                max(l.x_max for l in para_lines),
                max(l.y_max for l in para_lines),
            )
            blocks.append(
                DocumentBlock(
                    block_id=block_counter,
                    block_type=final_block_type,
                    text=merged_para_text,
                    lines=para_lines,
                    confidence=avg_conf,
                    bbox=bbox,
                    level=1,
                )
            )
            block_counter += 1
            idx = lookahead

        return blocks

    def _inspect_single_line(
        self,
        line: ReconstructedLine,
        body_height: float,
        doc_x_min: float,
        doc_width: float,
        is_first_line: bool,
    ) -> Tuple[BlockType, Optional[Tuple[str, str]], int, str]:
        """
        Determina la naturaleza estructural de una línea individual basándose en su
        geometría (altura tipográfica, centrado) y sintaxis (viñetas, dos puntos, numeración).

        Returns:
            Tupla `(BlockType, key_value_pair, heading_level, cleaned_text)`.
        """
        raw_text = line.text.strip()
        if not raw_text:
            return BlockType.PARAGRAPH, None, 1, ""

        height_ratio = line.height / max(1.0, body_height)
        word_count = len(raw_text.split())

        # 1. ¿Es un Título Principal (H1) o Subtítulo (H2)?
        ends_with_sentence_dot = raw_text.endswith(".") and not NUMBERED_HEADING_PATTERN.match(raw_text)
        if word_count <= 16 and not ends_with_sentence_dot:
            # Si es una sección numerada corta (ej. "1. Resumen Operativo") y no es la primera línea H1
            if not is_first_line and NUMBERED_HEADING_PATTERN.match(raw_text) and word_count <= 9:
                return BlockType.SUBTITLE, None, 2, raw_text

            if height_ratio >= self.heading_height_ratio or (
                is_first_line and height_ratio >= self.sub_heading_height_ratio and word_count <= 12
            ):
                return BlockType.TITLE, None, 1, raw_text

            if height_ratio >= self.sub_heading_height_ratio:
                return BlockType.SUBTITLE, None, 2, raw_text

            # Encabezado en mayúsculas o numerado (ej. "1. INTRODUCCIÓN", "RESUMEN EJECUTIVO")
            alpha_chars = [c for c in raw_text if c.isalpha()]
            is_all_caps = len(alpha_chars) >= 4 and all(c.isupper() for c in alpha_chars)
            if (is_all_caps and word_count <= 10) or (
                NUMBERED_HEADING_PATTERN.match(raw_text) and word_count <= 9
            ):
                return BlockType.SUBTITLE, None, 2, raw_text

        # 2. ¿Es un elemento de lista (viñeta o lista numerada)?
        if self.detect_lists:
            bullet_match = BULLET_PATTERN.match(raw_text)
            if bullet_match:
                item_content = bullet_match.group(1).strip()
                return BlockType.LIST_ITEM, None, 1, item_content

        # 3. ¿Es un par Clave: Valor de metadatos (ej. "Fecha de emisión: 11/11/2026", "Cliente: ACME")?
        if self.detect_key_value_pairs and len(line.regions) == 2 and ":" not in raw_text:
            # Caso común en facturas/formularios: dos cajas en la misma línea donde la izquierda es una etiqueta corta
            left_text = line.regions[0].text.strip()
            right_text = line.regions[1].text.strip()
            if 1 <= len(left_text.split()) <= 4 and left_text.endswith(":"):
                key_clean = left_text[:-1].strip()
                return BlockType.KEY_VALUE, (key_clean, right_text), 1, f"{key_clean}: {right_text}"

        if self.detect_key_value_pairs and word_count <= 14:
            kv_match = KEY_VALUE_PATTERN.match(raw_text)
            if kv_match:
                key_part = kv_match.group(1).strip()
                val_part = kv_match.group(2).strip()
                # Evitar falsos positivos con horas ("10:30") o URLs ("https://...")
                if (
                    not key_part.isdigit()
                    and key_part.lower() not in ("http", "https", "nota", "importante", "aviso")
                    and len(key_part.split()) <= 5
                    and len(val_part) <= 95
                ):
                    return BlockType.KEY_VALUE, (key_part, val_part), 1, f"{key_part}: {val_part}"

        # 4. ¿Es una Nota o Advertencia de una sola línea?
        if NOTE_PREFIX_PATTERN.match(raw_text):
            return BlockType.QUOTE_OR_NOTE, None, 1, raw_text

        return BlockType.PARAGRAPH, None, 1, raw_text

    @staticmethod
    def _build_table_matrix(table_lines: List[ReconstructedLine]) -> List[List[str]]:
        """
        Alinea las regiones de múltiples líneas tabulares en columnas coherentes
        mediante clustering de sus coordenadas horizontales (`x_min`).
        """
        # Recolectar todos los inicios de columna (`x_min`) de todas las celdas
        all_x_starts: List[float] = []
        all_heights: List[float] = []
        for line in table_lines:
            all_heights.append(line.height)
            for reg in line.regions:
                all_x_starts.append(reg.x_min)

        all_x_starts.sort()
        median_h = float(np.median(all_heights)) if all_heights else 20.0
        col_cluster_tol = max(25.0, median_h * 1.8)

        # Agrupar coordenadas X cercanas en centros de columna
        column_anchors: List[float] = []
        cluster_members: List[List[float]] = []

        for x_val in all_x_starts:
            if not column_anchors or abs(x_val - column_anchors[-1]) > col_cluster_tol:
                cluster_members.append([x_val])
                column_anchors.append(x_val)
            else:
                cluster_members[-1].append(x_val)
                column_anchors[-1] = sum(cluster_members[-1]) / len(cluster_members[-1])

        num_cols = max(2, len(column_anchors))
        matrix: List[List[str]] = []

        for line in table_lines:
            row = ["" for _ in range(num_cols)]
            for reg in line.regions:
                # Encontrar la columna cuyo anclaje X sea más cercano al inicio de esta celda
                best_col = min(
                    range(num_cols),
                    key=lambda c_idx: abs(reg.x_min - column_anchors[c_idx]),
                )
                if row[best_col]:
                    row[best_col] = f"{row[best_col]} {reg.text.strip()}"
                else:
                    row[best_col] = reg.text.strip()
            matrix.append(row)

        return matrix

    @staticmethod
    def _table_matrix_to_text(matrix: List[List[str]]) -> str:
        """Convierte una matriz tabular en una representación textual legible separada por barras."""
        if not matrix:
            return ""
        return "\n".join(" | ".join(cell for cell in row) for row in matrix)

    @staticmethod
    def _join_wrapped_lines(line_texts: List[str]) -> str:
        """
        Une las líneas de un mismo párrafo en un flujo continuo y coherente,
        eliminando guiones de partición silábica al final de línea (`-\n`).
        """
        if not line_texts:
            return ""
        result = line_texts[0].strip()
        for next_part in line_texts[1:]:
            clean_next = next_part.strip()
            if not clean_next:
                continue
            # Si la línea anterior termina en guion de palabra partida (ej. "docu-")
            if result.endswith("-") and len(result) > 2 and result[-2].isalpha() and clean_next[0].islower():
                result = result[:-1] + clean_next
            else:
                result = f"{result} {clean_next}"
        return result

    @staticmethod
    def _compute_analytics(
        raw_regions: List[TextRegion],
        blocks: List[DocumentBlock],
        processing_time_ms: float,
    ) -> DocumentAnalytics:
        """Calcula métricas descriptivas y distribución de tipos de bloques."""
        confidences = [r.confidence for r in raw_regions] if raw_regions else [0.0]
        full_text = " ".join(b.text for b in blocks)
        words = [w for w in full_text.split() if w.strip()]
        total_words = len(words)
        total_chars = sum(len(b.text) for b in blocks)

        counts: Dict[str, int] = {bt.value: 0 for bt in BlockType}
        for b in blocks:
            counts[b.block_type.value] = counts.get(b.block_type.value, 0) + 1

        # Velocidad promedio de lectura adulta: 200 palabras por minuto
        reading_sec = max(1, int(round((total_words / 200.0) * 60.0))) if total_words > 0 else 0

        return DocumentAnalytics(
            total_regions=len(raw_regions),
            total_blocks=len(blocks),
            total_words=total_words,
            total_characters=total_chars,
            mean_confidence=float(np.mean(confidences)) if raw_regions else 0.0,
            median_confidence=float(np.median(confidences)) if raw_regions else 0.0,
            min_confidence=float(np.min(confidences)) if raw_regions else 0.0,
            max_confidence=float(np.max(confidences)) if raw_regions else 0.0,
            block_type_counts=counts,
            estimated_reading_time_sec=reading_sec,
            processing_time_ms=processing_time_ms,
        )

    @staticmethod
    def format_as_markdown(doc: StructuredDocument) -> str:
        """
        Transforma un `StructuredDocument` en un documento Markdown limpio,
        jerárquico y bien estructurado (GFM).
        """
        if not doc.blocks:
            return "_No se detectó texto estructurable en la imagen._"

        md_Parts: List[str] = []
        idx = 0
        n_blocks = len(doc.blocks)

        while idx < n_blocks:
            block = doc.blocks[idx]

            if block.block_type == BlockType.TITLE:
                md_Parts.append(f"# {block.text}\n")
            elif block.block_type == BlockType.SUBTITLE:
                prefix = "##" if block.level <= 2 else "###"
                md_Parts.append(f"{prefix} {block.text}\n")
            elif block.block_type == BlockType.KEY_VALUE and block.key_value:
                # Agrupar pares clave-valor consecutivos en un bloque compacto
                kv_lines: List[str] = []
                while idx < n_blocks and doc.blocks[idx].block_type == BlockType.KEY_VALUE:
                    curr_kv = doc.blocks[idx].key_value
                    if curr_kv:
                        kv_lines.append(f"- **{curr_kv[0]}:** {curr_kv[1]}")
                    else:
                        kv_lines.append(f"- {doc.blocks[idx].text}")
                    idx += 1
                md_Parts.append("\n".join(kv_lines) + "\n")
                continue
            elif block.block_type == BlockType.LIST_ITEM:
                # Agrupar ítems de lista consecutivos sin líneas en blanco intermedias
                list_lines: List[str] = []
                while idx < n_blocks and doc.blocks[idx].block_type == BlockType.LIST_ITEM:
                    list_lines.append(f"- {doc.blocks[idx].text}")
                    idx += 1
                md_Parts.append("\n".join(list_lines) + "\n")
                continue
            elif block.block_type == BlockType.TABLE and block.table_data:
                matrix = block.table_data
                if matrix:
                    num_cols = len(matrix[0])
                    header = "| " + " | ".join(cell or f"Col {i+1}" for i, cell in enumerate(matrix[0])) + " |"
                    sep = "| " + " | ".join(["---"] * num_cols) + " |"
                    rows = [
                        "| " + " | ".join(cell for cell in row) + " |"
                        for row in matrix[1:]
                    ]
                    md_Parts.append("\n".join([header, sep] + rows) + "\n")
            elif block.block_type == BlockType.QUOTE_OR_NOTE:
                md_Parts.append(f"> {block.text}\n")
            else:
                md_Parts.append(f"{block.text}\n")

            idx += 1

        return "\n".join(md_Parts).strip() + "\n"

    @staticmethod
    def format_as_plain_text(doc: StructuredDocument) -> str:
        """
        Transforma un `StructuredDocument` en texto plano claramente organizado
        con subrayados ASCII para títulos, viñetas uniformes y columnas alineadas.
        """
        if not doc.blocks:
            return ""

        out_lines: List[str] = []
        for block in doc.blocks:
            if block.block_type == BlockType.TITLE:
                out_lines.append(block.text.upper())
                out_lines.append("=" * min(72, max(12, len(block.text))))
                out_lines.append("")
            elif block.block_type == BlockType.SUBTITLE:
                out_lines.append(block.text)
                out_lines.append("-" * min(60, max(10, len(block.text))))
                out_lines.append("")
            elif block.block_type == BlockType.KEY_VALUE and block.key_value:
                out_lines.append(f"{block.key_value[0]}: {block.key_value[1]}")
            elif block.block_type == BlockType.LIST_ITEM:
                out_lines.append(f"  • {block.text}")
            elif block.block_type == BlockType.TABLE and block.table_data:
                matrix = block.table_data
                if matrix:
                    num_cols = len(matrix[0])
                    col_widths = [
                        max(len(row[c]) for row in matrix)
                        for c in range(num_cols)
                    ]
                    out_lines.append("")
                    for r_idx, row in enumerate(matrix):
                        formatted_row = "  |  ".join(
                            row[c].ljust(col_widths[c]) for c in range(num_cols)
                        )
                        out_lines.append(formatted_row)
                        if r_idx == 0:
                            out_lines.append("-" * len(formatted_row))
                    out_lines.append("")
            elif block.block_type == BlockType.QUOTE_OR_NOTE:
                out_lines.append(f"[NOTA] {block.text}")
                out_lines.append("")
            else:
                out_lines.append(block.text)
                out_lines.append("")

        return "\n".join(out_lines).strip() + "\n"

    @staticmethod
    def format_as_rich_html(doc: StructuredDocument) -> str:
        """
        Genera HTML estilizado con tipografía editorial cálida y etiquetas semánticas
        para la vista previa enriquecida dentro de la aplicación PySide6.
        """
        if not doc.blocks:
            return (
                "<div style='color:#64748B; font-style:italic; padding:20px;'>"
                "No hay contenido estructurado para mostrar. Suba o seleccione una imagen."
                "</div>"
            )

        import html

        parts: List[str] = [
            "<div style='font-family: \"Segoe UI\", Inter, sans-serif; color: #1E293B; line-height: 1.6;'>"
        ]

        badge_colors = {
            BlockType.TITLE: ("#0F766E", "#CCFBF1", "TÍTULO"),
            BlockType.SUBTITLE: ("#1D4ED8", "#DBEAFE", "SUBTÍTULO"),
            BlockType.PARAGRAPH: ("#475569", "#F1F5F9", "PÁRRAFO"),
            BlockType.LIST_ITEM: ("#B45309", "#FEF3C7", "LISTA"),
            BlockType.KEY_VALUE: ("#6D28D9", "#EDE9FE", "DATO CLAVE"),
            BlockType.TABLE: ("#047857", "#D1FAE5", "TABLA"),
            BlockType.QUOTE_OR_NOTE: ("#BE123C", "#FFE4E6", "NOTA"),
        }

        for block in doc.blocks:
            fg, bg, label = badge_colors.get(
                block.block_type, ("#475569", "#F1F5F9", block.block_type.value)
            )
            conf_pct = int(round(block.confidence * 100))
            badge_html = (
                f"<span style='background-color:{bg}; color:{fg}; font-size:10px; "
                f"font-weight:700; padding:2px 6px; border-radius:4px; margin-right:8px;'>"
                f"{label} · {conf_pct}%</span>"
            )

            escaped_text = html.escape(block.text)

            if block.block_type == BlockType.TITLE:
                parts.append(
                    f"<div style='margin-top:14px; margin-bottom:10px; border-bottom:2px solid #E2E8F0; padding-bottom:6px;'>"
                    f"{badge_html}<span style='font-size:20px; font-weight:800; color:#0F172A;'>{escaped_text}</span></div>"
                )
            elif block.block_type == BlockType.SUBTITLE:
                parts.append(
                    f"<div style='margin-top:12px; margin-bottom:6px;'>"
                    f"{badge_html}<span style='font-size:16px; font-weight:700; color:#1E3A8A;'>{escaped_text}</span></div>"
                )
            elif block.block_type == BlockType.KEY_VALUE and block.key_value:
                k_esc = html.escape(block.key_value[0])
                v_esc = html.escape(block.key_value[1])
                parts.append(
                    f"<div style='margin:4px 0; padding:6px 10px; background-color:#F8FAFC; border-left:3px solid #7C3AED;'>"
                    f"{badge_html}<strong style='color:#334155;'>{k_esc}:</strong> "
                    f"<span style='color:#0F172A;'>{v_esc}</span></div>"
                )
            elif block.block_type == BlockType.LIST_ITEM:
                parts.append(
                    f"<div style='margin:4px 0 4px 16px;'>"
                    f"{badge_html}<span style='color:#B45309; font-weight:bold;'>&#8226;</span> {escaped_text}</div>"
                )
            elif block.block_type == BlockType.TABLE and block.table_data:
                parts.append(f"<div style='margin:12px 0;'>{badge_html}</div>")
                parts.append(
                    "<table style='border-collapse:collapse; width:100%; margin-bottom:14px; font-size:13px;'>"
                )
                for r_idx, row in enumerate(block.table_data):
                    row_bg = "#F1F5F9" if r_idx == 0 else ("#FFFFFF" if r_idx % 2 == 1 else "#F8FAFC")
                    parts.append(f"<tr style='background-color:{row_bg};'>")
                    for cell in row:
                        cell_esc = html.escape(cell)
                        tag = "th" if r_idx == 0 else "td"
                        weight = "700" if r_idx == 0 else "400"
                        parts.append(
                            f"<{tag} style='border:1px solid #CBD5E1; padding:6px 10px; "
                            f"text-align:left; font-weight:{weight};'>{cell_esc}</{tag}>"
                        )
                    parts.append("</tr>")
                parts.append("</table>")
            elif block.block_type == BlockType.QUOTE_OR_NOTE:
                parts.append(
                    f"<div style='margin:10px 0; padding:10px 12px; background-color:#FFF1F2; "
                    f"border-left:4px solid #E11D48; font-style:italic;'>"
                    f"{badge_html}{escaped_text}</div>"
                )
            else:
                parts.append(
                    f"<p style='margin:8px 0; text-align:justify;'>{badge_html}{escaped_text}</p>"
                )

        parts.append("</div>")
        return "\n".join(parts)
