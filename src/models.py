"""
Modelos de datos para la representación espacial, semántica y analítica
de documentos extraídos mediante OCR.

Decisiones técnicas:
- Se utilizan `dataclasses` tipadas para garantizar trazabilidad e inmutabilidad
  parcial entre las etapas del pipeline (OCR -> Agrupación -> Clasificación -> Exportación).
- Cada `TextRegion` conserva tanto las coordenadas originales (polígono de 4 puntos)
  como el rectángulo envolvente normalizado (`bbox`: x_min, y_min, x_max, y_max)
  para facilitar cálculos vectorizados de solapamiento vertical y brechas horizontales.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class BlockType( str, Enum ):
    """Tipos jerárquicos de bloques de documento reconstruidos."""
    TITLE = "TITLE"                 # Título principal del documento (H1)
    SUBTITLE = "SUBTITLE"           # Subtítulo o encabezado de sección (H2/H3)
    PARAGRAPH = "PARAGRAPH"         # Párrafo cohesivo de texto continuo
    LIST_ITEM = "LIST_ITEM"         # Elemento de lista (viñeta o numerada)
    KEY_VALUE = "KEY_VALUE"         # Par clave-valor o metadato (ej. Fecha: 12/10/2026)
    TABLE = "TABLE"                 # Tabla reconstruida en filas y columnas
    QUOTE_OR_NOTE = "QUOTE_OR_NOTE" # Nota al pie, cita o bloque destacado


@dataclass
class TextRegion:
    """
    Representa una región cruda de texto detectada por el motor OCR.

    Attributes:
        text: Cadena de texto reconocida en la región.
        confidence: Nivel de confianza del reconocimiento en el rango [0.0, 1.0].
        polygon: Lista de 4 vértices [(x1,y1), (x2,y2), (x3,y3), (x4,y4)].
        x_min: Coordenada X izquierda del cuadro delimitador.
        y_min: Coordenada Y superior del cuadro delimitador.
        x_max: Coordenada X derecha del cuadro delimitador.
        y_max: Coordenada Y inferior del cuadro delimitador.
    """
    text: str
    confidence: float
    polygon: List[Tuple[float, float]]
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """Ancho de la caja delimitadora en píxeles."""
        return max(1.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        """Alto de la caja delimitadora en píxeles."""
        return max(1.0, self.y_max - self.y_min)

    @property
    def center_y(self) -> float:
        """Centro vertical de la región."""
        return (self.y_min + self.y_max) / 2.0

    @property
    def center_x(self) -> float:
        """Centro horizontal de la región."""
        return (self.x_min + self.x_max) / 2.0

    @property
    def char_width(self) -> float:
        """Ancho promedio estimado por carácter dentro de la región."""
        length = max(1, len(self.text.strip()))
        return self.width / float(length)


@dataclass
class ReconstructedLine:
    """
    Representa una línea horizontal reconstruida a partir de una o más regiones
    de texto (`TextRegion`) que comparten la misma banda vertical.

    Attributes:
        regions: Regiones ordenadas de izquierda a derecha.
        x_min: Límite izquierdo de la línea.
        y_min: Límite superior de la línea.
        x_max: Límite derecho de la línea.
        y_max: Límite inferior de la línea.
        gaps: Brechas horizontales en píxeles entre regiones consecutivas.
    """
    regions: List[TextRegion]
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    gaps: List[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Texto completo de la línea uniendo sus regiones."""
        return " ".join(r.text.strip() for r in self.regions if r.text.strip())

    @property
    def height(self) -> float:
        """Altura mediana de las regiones que componen la línea."""
        if not self.regions:
            return max(1.0, self.y_max - self.y_min)
        heights = sorted(r.height for r in self.regions)
        return heights[len(heights) // 2]

    @property
    def width(self) -> float:
        """Ancho total abarcado por la línea."""
        return max(1.0, self.x_max - self.x_min)

    @property
    def avg_confidence(self) -> float:
        """Confianza promedio ponderada de la línea."""
        if not self.regions:
            return 0.0
        return sum(r.confidence for r in self.regions) / len(self.regions)


@dataclass
class DocumentBlock:
    """
    Bloque semántico y estructural dentro del documento reconstruido.

    Attributes:
        block_id: Identificador secuencial único (1..N).
        block_type: Categoría jerárquica (`BlockType`).
        text: Contenido textual limpio y formateado.
        lines: Líneas reconstruidas que dieron origen al bloque.
        confidence: Confianza media del bloque [0.0, 1.0].
        bbox: Tupla `(x_min, y_min, x_max, y_max)` en coordenadas de la imagen.
        table_data: Matriz bidimensional de celdas si `block_type == BlockType.TABLE`.
        key_value: Tupla `(clave, valor)` si `block_type == BlockType.KEY_VALUE`.
        level: Nivel jerárquico para encabezados (1 para H1, 2 para H2, etc.).
    """
    block_id: int
    block_type: BlockType
    text: str
    lines: List[ReconstructedLine]
    confidence: float
    bbox: Tuple[float, float, float, float]
    table_data: Optional[List[List[str]]] = None
    key_value: Optional[Tuple[str, str]] = None
    level: int = 1


@dataclass
class DocumentAnalytics:
    """
    Métricas cuantitativas y estadísticas del documento procesado,
    utilizadas tanto en el panel de análisis de la interfaz como en el reporte PDF.
    """
    total_regions: int = 0
    total_blocks: int = 0
    total_words: int = 0
    total_characters: int = 0
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    block_type_counts: Dict[str, int] = field(default_factory=dict)
    estimated_reading_time_sec: int = 0
    processing_time_ms: float = 0.0


@dataclass
class StructuredDocument:
    """
    Resultado integral del pipeline de extracción y estructuración coherente.

    Attributes:
        source_path: Ruta del archivo de imagen original.
        image_size: Dimensiones `(ancho, alto)` en píxeles de la imagen procesada.
        raw_regions: Regiones crudas devueltas por el OCR.
        reconstructed_lines: Líneas horizontales reconstruidas.
        blocks: Bloques semánticos ordenados jerárquicamente y por lectura natural.
        analytics: Resumen estadístico del documento.
    """
    source_path: str
    image_size: Tuple[int, int]
    raw_regions: List[TextRegion]
    reconstructed_lines: List[ReconstructedLine]
    blocks: List[DocumentBlock]
    analytics: DocumentAnalytics

