"""
Interfaz Gráfica de Usuario (GUI) para DocuStruct Studio construida con PySide6.

Decisiones de diseño estético y funcional:
- **Paleta Editorial Cálida y Luminosa ("Warm Paper & Slate Studio"):**
  Se evita deliberadamente el cliché oscuro/cyberpunk con neones. En su lugar, se emplea
  una estética inspirada en herramientas editoriales modernas (fondo marfil cálido `#F8F7F4`,
  tarjetas blancas `#FFFFFF` con bordes suaves `#E2E8F0`, acento principal verde azulado
  profundo `#0F766E`, acento secundario ámbar terracota `#D97706` y tipografía gris pizarra
  oscuro `#0F172A` de alto contraste).
- **Arquitectura No Bloqueante (`QThread`):**
  El procesamiento OCR y la reconstrucción estructural se ejecutan en un hilo trabajador
  dedicado (`OCRWorkerThread`) para que la ventana principal permanezca fluida (`Responding: True`)
  en todo momento.
- **Visor de Imagen con Superposición Estructural (`AnnotatedImageView`):**
  Permite alternar entre la imagen limpia y la vista con cajas delimitadoras coloreadas según
  el tipo jerárquico del bloque detectado (Título, Subtítulo, Párrafo, Lista, Clave-Valor, Tabla),
  demostrando visualmente cómo el algoritmo agrupa y comprende el documento.
"""

import os
import traceback
from typing import Dict, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import (
    QAction,
    QClipboard,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QGuiApplication,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.exporter import BLOCK_LABELS_ES, ReportExporter
from src.icon_generator import ensure_app_icon, generate_sample_document_image
from src.models import BlockType, StructuredDocument
from src.ocr_engine import OCREngine
from src.structure_engine import StructureEngine


# Mapa de colores RGB por tipo de bloque para la superposición geométrica en el visor
BLOCK_OVERLAY_COLORS: Dict[BlockType, Tuple[int, int, int]] = {
    BlockType.TITLE: (15, 118, 110),        # Teal profundo
    BlockType.SUBTITLE: (29, 78, 216),      # Azul real
    BlockType.PARAGRAPH: (71, 85, 105),     # Pizarra
    BlockType.LIST_ITEM: (180, 83, 9),      # Ámbar oscuro
    BlockType.KEY_VALUE: (109, 40, 217),    # Violeta editorial
    BlockType.TABLE: (4, 120, 87),          # Esmeralda
    BlockType.QUOTE_OR_NOTE: (190, 18, 60), # Carmesí
}


class OCRWorkerThread(QThread):
    """
    Hilo de ejecución en segundo plano que realiza el preprocesamiento de imagen,
    la extracción OCR mediante RapidOCR y la reconstrucción estructural coherente.
    """

    finished_success = Signal(object)  # Emite StructuredDocument
    failed_error = Signal(str)

    def __init__(
        self,
        ocr_engine: OCREngine,
        structure_engine: StructureEngine,
        image_path: str,
        enhance_contrast: bool = True,
        denoise: bool = False,
        detect_tables: bool = True,
        detect_lists: bool = True,
        detect_kv: bool = True,
    ) -> None:
        super().__init__()
        self.ocr_engine = ocr_engine
        self.structure_engine = structure_engine
        self.image_path = image_path
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
        self.detect_tables = detect_tables
        self.detect_lists = detect_lists
        self.detect_kv = detect_kv

    def run(self) -> None:
        try:
            self.structure_engine.detect_tables = self.detect_tables
            self.structure_engine.detect_lists = self.detect_lists
            self.structure_engine.detect_key_value_pairs = self.detect_kv

            regions, img_size, elapsed_ms = self.ocr_engine.extract_regions(
                self.image_path,
                enhance_contrast=self.enhance_contrast,
                denoise=self.denoise,
            )
            structured_doc = self.structure_engine.build_structured_document(
                raw_regions=regions,
                source_path=self.image_path,
                image_size=img_size,
                processing_time_ms=elapsed_ms,
            )
            self.finished_success.emit(structured_doc)
        except Exception as exc:
            err_msg = f"{exc}\n\n{traceback.format_exc()}"
            self.failed_error.emit(err_msg)


class AnnotatedImageCanvas(QWidget):
    """
    Lienzo inteligente para visualizar la imagen cargada con soporte de arrastrar y soltar
    (Drag & Drop) y superposición opcional de los bloques estructurales detectados.
    """

    image_dropped = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(360, 420)
        self._pixmap: Optional[QPixmap] = None
        self._structured_doc: Optional[StructuredDocument] = None
        self._show_overlays: bool = True

    def set_image_and_document(
        self,
        image_path: str,
        doc: Optional[StructuredDocument] = None,
    ) -> None:
        """Actualiza el pixmap mostrado y el documento estructurado asociado."""
        if os.path.exists(image_path):
            self._pixmap = QPixmap(image_path)
        else:
            self._pixmap = None
        self._structured_doc = doc
        self.update()

    def set_show_overlays(self, show: bool) -> None:
        """Activa o desactiva el dibujo de las cajas estructurales sobre la imagen."""
        self._show_overlays = show
        self.update()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            local_path = urls[0].toLocalFile()
            if local_path and os.path.isfile(local_path):
                self.image_dropped.emit(local_path)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self.rect().adjusted(8, 8, -8, -8)

        # Fondo del área del visor
        painter.setPen(QPen(QColor("#CBD5E1"), 1.5, Qt.PenStyle.DashLine))
        painter.setBrush(QColor("#F1EFEA"))
        painter.drawRoundedRect(rect, 10, 10)

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#475569"))
            font = QFont("Segoe UI", 11, QFont.Weight.Medium)
            painter.setFont(font)
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter,
                "Arrastra y suelta una imagen aquí\no usa el botón 'Cargar Imagen'\n\n"
                "(Formatos soportados: PNG, JPG, JPEG, BMP, WEBP, TIFF)",
            )
            return

        # Escalar proporcionalmente la imagen al contenedor
        scaled_pix = self._pixmap.scaled(
            rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        offset_x = rect.x() + (rect.width() - scaled_pix.width()) // 2
        offset_y = rect.y() + (rect.height() - scaled_pix.height()) // 2
        painter.drawPixmap(offset_x, offset_y, scaled_pix)

        # Dibujar cajas delimitadoras por bloque si están disponibles y habilitadas
        if self._show_overlays and self._structured_doc and self._structured_doc.blocks:
            orig_w, orig_h = self._structured_doc.image_size
            if orig_w > 0 and orig_h > 0:
                scale_x = scaled_pix.width() / float(orig_w)
                scale_y = scaled_pix.height() / float(orig_h)

                label_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
                painter.setFont(label_font)

                for block in self._structured_doc.blocks:
                    bx0, by0, bx1, by1 = block.bbox
                    rx = int(offset_x + bx0 * scale_x)
                    ry = int(offset_y + by0 * scale_y)
                    rw = max(8, int((bx1 - bx0) * scale_x))
                    rh = max(8, int((by1 - by0) * scale_y))

                    r, g, b = BLOCK_OVERLAY_COLORS.get(block.block_type, (71, 85, 105))
                    border_col = QColor(r, g, b, 230)
                    fill_col = QColor(r, g, b, 34)

                    painter.setPen(QPen(border_col, 2.0))
                    painter.setBrush(fill_col)
                    painter.drawRoundedRect(rx, ry, rw, rh, 4, 4)

                    # Etiqueta con número de bloque en la esquina superior izquierda
                    tag_text = f"#{block.block_id} {block.block_type.value[:4]}"
                    tag_w = min(rw, 64)
                    tag_h = 15
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(r, g, b, 225))
                    painter.drawRoundedRect(rx, max(offset_y, ry - tag_h), tag_w, tag_h, 3, 3)
                    painter.setPen(QColor("#FFFFFF"))
                    painter.drawText(
                        rx + 4,
                        max(offset_y, ry - tag_h) + 11,
                        tag_text,
                    )


class MainWindow(QMainWindow):
    """
    Ventana principal de DocuStruct Studio.
    Integra controles de carga, preprocesamiento, visor de imagen con bloques superpuestos,
    vistas de texto organizado (Vista Enriquecida, Markdown, Texto Plano y Analítica) y
    exportación directa a PDF maquetado, Markdown y TXT.
    """

    def __init__(self, base_dir: str) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.assets_dir = os.path.join(self.base_dir, "assets")
        self.outputs_dir = os.path.join(self.base_dir, "outputs")
        os.makedirs(self.assets_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Generar ícono y documento de muestra
        self.icon_png, self.icon_ico = ensure_app_icon(self.assets_dir)
        self.sample_img_path = generate_sample_document_image(
            os.path.join(self.assets_dir, "sample_document.png")
        )

        # Motores de procesamiento
        self.ocr_engine = OCREngine(confidence_threshold=0.35)
        self.structure_engine = StructureEngine()
        self.current_image_path: Optional[str] = None
        self.current_document: Optional[StructuredDocument] = None
        self.worker_thread: Optional[OCRWorkerThread] = None

        self._setup_window()
        self._apply_editorial_stylesheet()
        self._build_ui()

        # Cargar el documento de demostración e iniciar su análisis automáticamente
        # para que el usuario vea resultados reales desde el primer instante
        self.load_image_file(self.sample_img_path, auto_extract=True)

    def _setup_window(self) -> None:
        self.setWindowTitle("DocuStruct Studio — Extracción Óptica y Organización Estructural de Texto")
        self.resize(1340, 840)
        self.setMinimumSize(1020, 680)
        if os.path.exists(self.icon_ico):
            self.setWindowIcon(QIcon(self.icon_ico))
        elif os.path.exists(self.icon_png):
            self.setWindowIcon(QIcon(self.icon_png))

    def _apply_editorial_stylesheet(self) -> None:
        """
        Aplica la hoja de estilos QSS con paleta clara, cálida y editorial
        ("Warm Paper & Deep Teal Studio").
        """
        self.setStyleSheet(
            """
            QMainWindow, QWidget#CentralContainer {
                background-color: #F7F5F0;
                color: #0F172A;
                font-family: "Segoe UI", Inter, sans-serif;
            }
            QFrame#HeaderBanner {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
            }
            QFrame#CardPanel {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
            }
            QFrame#KpiBox {
                background-color: #F8FAFC;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
            }
            QLabel#AppTitle {
                color: #0F172A;
                font-size: 19px;
                font-weight: 800;
            }
            QLabel#AppSubtitle {
                color: #475569;
                font-size: 12px;
            }
            QLabel#SectionHeading {
                color: #0F766E;
                font-size: 13px;
                font-weight: 700;
            }
            QLabel#KpiValue {
                color: #0F766E;
                font-size: 18px;
                font-weight: 800;
            }
            QLabel#KpiLabel {
                color: #64748B;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton {
                background-color: #F1F5F9;
                color: #1E293B;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 7px 14px;
                font-size: 12.5px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #E2E8F0;
                border-color: #94A3B8;
            }
            QPushButton#PrimaryButton {
                background-color: #0F766E;
                color: #FFFFFF;
                border: 1px solid #0D9488;
                font-weight: 700;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #115E59;
            }
            QPushButton#PrimaryButton:disabled {
                background-color: #94A3B8;
                border-color: #94A3B8;
            }
            QPushButton#AccentButton {
                background-color: #D97706;
                color: #FFFFFF;
                border: 1px solid #B45309;
                font-weight: 700;
            }
            QPushButton#AccentButton:hover {
                background-color: #B45309;
            }
            QCheckBox {
                color: #334155;
                font-size: 12px;
                spacing: 6px;
            }
            QTabWidget::pane {
                border: 1px solid #CBD5E1;
                background-color: #FFFFFF;
                border-radius: 8px;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #F1EFEA;
                color: #475569;
                border: 1px solid #CBD5E1;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 16px;
                margin-right: 4px;
                font-size: 12.5px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #0F766E;
                border-bottom: 2px solid #0F766E;
                font-weight: 700;
            }
            QTextBrowser, QPlainTextEdit {
                background-color: #FFFFFF;
                color: #0F172A;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                selection-background-color: #CCFBF1;
                selection-color: #0F172A;
            }
            QTableWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F8FAFC;
                gridline-color: #E2E8F0;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #F1F5F9;
                color: #0F172A;
                font-weight: 700;
                padding: 6px;
                border: 1px solid #CBD5E1;
            }
            QProgressBar {
                border: 1px solid #CBD5E1;
                border-radius: 5px;
                background-color: #F1F5F9;
                text-align: center;
                height: 14px;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #0F766E;
                border-radius: 4px;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("CentralContainer")
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 10)
        root_layout.setSpacing(12)

        # --- 1. Barra Superior de Identidad y Acciones Principales ---
        header_frame = QFrame()
        header_frame.setObjectName("HeaderBanner")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(14)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        lbl_title = QLabel("DocuStruct Studio")
        lbl_title.setObjectName("AppTitle")
        lbl_sub = QLabel(
            "Extracción OCR inteligente y reconstrucción coherente de jerarquía, párrafos, listas y tablas"
        )
        lbl_sub.setObjectName("AppSubtitle")
        title_col.addWidget(lbl_title)
        title_col.addWidget(lbl_sub)
        header_layout.addLayout(title_col, stretch=1)

        self.btn_load_image = QPushButton("Cargar Imagen...")
        self.btn_load_image.setToolTip("Seleccionar una imagen del equipo (PNG, JPG, WEBP, BMP, TIFF)")
        self.btn_load_image.clicked.connect(self._on_browse_image)

        self.btn_load_sample = QPushButton("Cargar Demo")
        self.btn_load_sample.setToolTip("Cargar el documento de auditoría de ejemplo incluido")
        self.btn_load_sample.clicked.connect(self._on_load_sample)

        self.btn_process = QPushButton("Extraer y Organizar Texto")
        self.btn_process.setObjectName("PrimaryButton")
        self.btn_process.setToolTip("Ejecutar OCR local y reconstruir la estructura lógica del documento")
        self.btn_process.clicked.connect(self.run_extraction_pipeline)

        self.btn_export_pdf = QPushButton("Exportar Reporte PDF")
        self.btn_export_pdf.setObjectName("AccentButton")
        self.btn_export_pdf.setToolTip("Generar un reporte PDF maquetado con métricas, gráficos y texto organizado")
        self.btn_export_pdf.clicked.connect(self._on_export_pdf)

        header_layout.addWidget(self.btn_load_image)
        header_layout.addWidget(self.btn_load_sample)
        header_layout.addWidget(self.btn_process)
        header_layout.addWidget(self.btn_export_pdf)

        root_layout.addWidget(header_frame)

        # --- 2. Divisor Central: Panel Izquierdo (Imagen + Ajustes) y Panel Derecho (Resultados) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Panel Izquierdo: Visor de Imagen y Controles de Preprocesamiento
        left_card = QFrame()
        left_card.setObjectName("CardPanel")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 12, 14, 12)
        left_layout.setSpacing(10)

        left_header_row = QHBoxLayout()
        lbl_left_sec = QLabel("1. Documento de Entrada y Mapa Estructural")
        lbl_left_sec.setObjectName("SectionHeading")
        self.chk_show_overlays = QCheckBox("Mostrar bloques detectados")
        self.chk_show_overlays.setChecked(True)
        self.chk_show_overlays.toggled.connect(self._on_toggle_overlays)
        left_header_row.addWidget(lbl_left_sec)
        left_header_row.addStretch()
        left_header_row.addWidget(self.chk_show_overlays)
        left_layout.addLayout(left_header_row)

        self.image_canvas = AnnotatedImageCanvas()
        self.image_canvas.image_dropped.connect(lambda p: self.load_image_file(p, auto_extract=True))
        left_layout.addWidget(self.image_canvas, stretch=1)

        # Opciones de preprocesamiento y estructuración
        opts_frame = QFrame()
        opts_frame.setObjectName("KpiBox")
        opts_layout = QGridLayout(opts_frame)
        opts_layout.setContentsMargins(10, 8, 10, 8)
        opts_layout.setSpacing(8)

        self.chk_contrast = QCheckBox("Mejorar contraste (CLAHE para fotos)")
        self.chk_contrast.setChecked(False)
        self.chk_denoise = QCheckBox("Reducción de ruido bilateral")
        self.chk_denoise.setChecked(False)
        self.chk_tables = QCheckBox("Reconstruir tablas multi-columna")
        self.chk_tables.setChecked(True)
        self.chk_kv = QCheckBox("Detectar metadatos (Clave: Valor)")
        self.chk_kv.setChecked(True)

        opts_layout.addWidget(self.chk_contrast, 0, 0)
        opts_layout.addWidget(self.chk_denoise, 0, 1)
        opts_layout.addWidget(self.chk_tables, 1, 0)
        opts_layout.addWidget(self.chk_kv, 1, 1)
        left_layout.addWidget(opts_frame)

        self.lbl_file_info = QLabel("Archivo: Ninguno seleccionado")
        self.lbl_file_info.setStyleSheet("color: #64748B; font-size: 11.5px;")
        left_layout.addWidget(self.lbl_file_info)

        # Panel Derecho: Tarjetas de KPIs, Pestañas de Texto Organizado y Botones de Exportación
        right_card = QFrame()
        right_card.setObjectName("CardPanel")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(14, 12, 14, 12)
        right_layout.setSpacing(10)

        # Fila de KPIs rápidos
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)
        self.kpi_blocks = self._create_kpi_widget("Bloques Coherentes", "0")
        self.kpi_words = self._create_kpi_widget("Palabras Extraídas", "0")
        self.kpi_conf = self._create_kpi_widget("Confianza Media OCR", "0.0%")
        self.kpi_time = self._create_kpi_widget("Tiempo de Proceso", "0 ms")
        kpi_row.addWidget(self.kpi_blocks[0])
        kpi_row.addWidget(self.kpi_words[0])
        kpi_row.addWidget(self.kpi_conf[0])
        kpi_row.addWidget(self.kpi_time[0])
        right_layout.addLayout(kpi_row)

        # Pestañas de organización del texto
        self.tabs = QTabWidget()

        # Pestaña 1: Vista Estructurada Enriquecida (HTML con etiquetas de tipo de bloque y tablas renderizadas)
        self.rich_browser = QTextBrowser()
        self.rich_browser.setOpenExternalLinks(False)
        self.tabs.addTab(self.rich_browser, "Vista Estructurada (Jerárquica)")

        # Pestaña 2: Editor Markdown (.md) editable
        self.markdown_editor = QPlainTextEdit()
        self.markdown_editor.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.markdown_editor, "Markdown Coherente (.md)")

        # Pestaña 3: Texto Plano Limpio (.txt) editable
        self.plain_editor = QPlainTextEdit()
        self.plain_editor.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.plain_editor, "Texto Plano Organizado (.txt)")

        # Pestaña 4: Desglose de Bloques y Analítica Estadística
        analytics_tab = QWidget()
        analytics_layout = QVBoxLayout(analytics_tab)
        analytics_layout.setContentsMargins(8, 8, 8, 8)
        analytics_layout.setSpacing(8)

        self.blocks_table = QTableWidget(0, 6)
        self.blocks_table.setHorizontalHeaderLabels(
            ["# ID", "Categoría Estructural", "Palabras", "Caracteres", "Confianza", "Contenido del Bloque"]
        )
        self.blocks_table.setAlternatingRowColors(True)
        self.blocks_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.blocks_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.blocks_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.blocks_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.blocks_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.blocks_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        analytics_layout.addWidget(self.blocks_table, stretch=1)

        self.lbl_chart_preview = QLabel("El gráfico estadístico se actualizará tras procesar una imagen.")
        self.lbl_chart_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_chart_preview.setMinimumHeight(185)
        self.lbl_chart_preview.setStyleSheet(
            "background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 4px;"
        )
        analytics_layout.addWidget(self.lbl_chart_preview)

        self.tabs.addTab(analytics_tab, "Inspector de Bloques y Estadísticas")
        right_layout.addWidget(self.tabs, stretch=1)

        # Barra de acciones de copia y exportación rápida
        bottom_actions = QHBoxLayout()
        bottom_actions.setSpacing(8)

        self.btn_copy_md = QPushButton("Copiar Markdown")
        self.btn_copy_md.clicked.connect(self._on_copy_markdown)

        self.btn_copy_txt = QPushButton("Copiar Texto Plano")
        self.btn_copy_txt.clicked.connect(self._on_copy_plain_text)

        self.btn_save_md = QPushButton("Guardar .MD")
        self.btn_save_md.clicked.connect(self._on_save_markdown)

        self.btn_save_txt = QPushButton("Guardar .TXT")
        self.btn_save_txt.clicked.connect(self._on_save_txt)

        bottom_actions.addWidget(self.btn_copy_md)
        bottom_actions.addWidget(self.btn_copy_txt)
        bottom_actions.addStretch()
        bottom_actions.addWidget(self.btn_save_md)
        bottom_actions.addWidget(self.btn_save_txt)
        right_layout.addLayout(bottom_actions)

        splitter.addWidget(left_card)
        splitter.addWidget(right_card)
        splitter.setStretchFactor(0, 44)
        splitter.setStretchFactor(1, 56)
        root_layout.addWidget(splitter, stretch=1)

        # --- 3. Barra de Estado y Progreso ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(220)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Listo. Cargue una imagen o presione 'Extraer y Organizar Texto'.")

    def _create_kpi_widget(self, title: str, initial_val: str) -> Tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setObjectName("KpiBox")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(2)

        lbl_val = QLabel(initial_val)
        lbl_val.setObjectName("KpiValue")
        lbl_title = QLabel(title)
        lbl_title.setObjectName("KpiLabel")

        lay.addWidget(lbl_val)
        lay.addWidget(lbl_title)
        return frame, lbl_val

    def load_image_file(self, file_path: str, auto_extract: bool = True) -> None:
        """Carga una imagen en el visor y opcionalmente dispara la extracción automática."""
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Archivo no encontrado", f"No existe el archivo:\n{file_path}")
            return

        self.current_image_path = os.path.abspath(file_path)
        self.image_canvas.set_image_and_document(self.current_image_path, None)
        self.lbl_file_info.setText(f"Archivo activo: {os.path.basename(self.current_image_path)}")
        self.status_bar.showMessage(f"Imagen cargada: {self.current_image_path}")

        if auto_extract:
            self.run_extraction_pipeline()

    def _on_browse_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen con texto para extraer y organizar",
            self.base_dir,
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;Todos los archivos (*.*)",
        )
        if file_path:
            self.load_image_file(file_path, auto_extract=True)

    def _on_load_sample(self) -> None:
        self.sample_img_path = generate_sample_document_image(
            os.path.join(self.assets_dir, "sample_document.png")
        )
        self.load_image_file(self.sample_img_path, auto_extract=True)

    def _on_toggle_overlays(self, checked: bool) -> None:
        self.image_canvas.set_show_overlays(checked)

    def run_extraction_pipeline(self) -> None:
        """Lanza el hilo trabajador de OCR y estructuración coherente."""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            QMessageBox.information(
                self,
                "Seleccione una imagen",
                "Por favor cargue primero una imagen antes de ejecutar la extracción.",
            )
            return

        if self.worker_thread and self.worker_thread.isRunning():
            return

        self.btn_process.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Analizando imagen con OCR y reconstruyendo jerarquía documental...")

        self.worker_thread = OCRWorkerThread(
            ocr_engine=self.ocr_engine,
            structure_engine=self.structure_engine,
            image_path=self.current_image_path,
            enhance_contrast=self.chk_contrast.isChecked(),
            denoise=self.chk_denoise.isChecked(),
            detect_tables=self.chk_tables.isChecked(),
            detect_lists=True,
            detect_kv=self.chk_kv.isChecked(),
        )
        self.worker_thread.finished_success.connect(self._on_extraction_finished)
        self.worker_thread.failed_error.connect(self._on_extraction_failed)
        self.worker_thread.start()

    def _on_extraction_finished(self, doc: StructuredDocument) -> None:
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.current_document = doc

        # Actualizar visor de imagen con las cajas de los bloques detectados
        self.image_canvas.set_image_and_document(self.current_image_path or "", doc)

        # Actualizar KPIs
        an = doc.analytics
        self.kpi_blocks[1].setText(str(an.total_blocks))
        self.kpi_words[1].setText(str(an.total_words))
        self.kpi_conf[1].setText(f"{an.mean_confidence * 100:.1f}%")
        self.kpi_time[1].setText(f"{an.processing_time_ms:.0f} ms")

        # Actualizar las 3 vistas textuales coherentes
        self.rich_browser.setHtml(StructureEngine.format_as_rich_html(doc))
        self.markdown_editor.setPlainText(StructureEngine.format_as_markdown(doc))
        self.plain_editor.setPlainText(StructureEngine.format_as_plain_text(doc))

        # Poblar tabla de inspección de bloques
        self.blocks_table.setRowCount(len(doc.blocks))
        for r_idx, b in enumerate(doc.blocks):
            w_count = len([w for w in b.text.split() if w.strip()])
            label_es = BLOCK_LABELS_ES.get(b.block_type.value, b.block_type.value)
            preview = b.text.replace("\n", " | ")

            items = [
                QTableWidgetItem(str(b.block_id)),
                QTableWidgetItem(label_es),
                QTableWidgetItem(str(w_count)),
                QTableWidgetItem(str(len(b.text))),
                QTableWidgetItem(f"{b.confidence * 100:.1f}%"),
                QTableWidgetItem(preview),
            ]
            for c_idx, item in enumerate(items):
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.blocks_table.setItem(r_idx, c_idx, item)

        # Generar gráfico estadístico y reporte automático en `outputs/`
        try:
            chart_path = os.path.join(self.outputs_dir, "ultimo_analisis_bloques.png")
            generated_chart = ReportExporter.generate_analytics_chart(doc, chart_path)
            if generated_chart and os.path.exists(generated_chart):
                chart_pix = QPixmap(generated_chart)
                scaled_chart = chart_pix.scaled(
                    QSize(620, 195),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.lbl_chart_preview.setPixmap(scaled_chart)

            # Guardar siempre una copia actualizada del reporte PDF en `outputs/reporte_extraccion_estructurada.pdf`
            auto_pdf_path = os.path.join(self.outputs_dir, "reporte_extraccion_estructurada.pdf")
            ReportExporter.export_pdf_report(doc, auto_pdf_path, include_analytics=True)
        except Exception:
            pass

        self.status_bar.showMessage(
            f"Extracción completada: {an.total_blocks} bloques organizados ({an.total_words} palabras) "
            f"en {an.processing_time_ms:.0f} ms."
        )

    def _on_extraction_failed(self, error_details: str) -> None:
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Error durante la extracción OCR.")
        QMessageBox.critical(
            self,
            "Error en el procesamiento OCR",
            f"Ocurrió un problema al procesar la imagen:\n\n{error_details}",
        )

    def _on_copy_markdown(self) -> None:
        text = self.markdown_editor.toPlainText().strip()
        if not text:
            return
        QGuiApplication.clipboard().setText(text, QClipboard.Mode.Clipboard)
        self.status_bar.showMessage("Texto en formato Markdown copiado al portapapeles.", 4000)

    def _on_copy_plain_text(self) -> None:
        text = self.plain_editor.toPlainText().strip()
        if not text:
            return
        QGuiApplication.clipboard().setText(text, QClipboard.Mode.Clipboard)
        self.status_bar.showMessage("Texto plano organizado copiado al portapapeles.", 4000)

    def _on_save_markdown(self) -> None:
        if not self.current_document:
            return
        default_path = os.path.join(self.outputs_dir, "documento_estructurado.md")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Documento en Markdown",
            default_path,
            "Markdown (*.md)",
        )
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(self.markdown_editor.toPlainText())
            self.status_bar.showMessage(f"Archivo Markdown guardado en: {out_path}", 5000)

    def _on_save_txt(self) -> None:
        if not self.current_document:
            return
        default_path = os.path.join(self.outputs_dir, "documento_estructurado.txt")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar Documento en Texto Plano",
            default_path,
            "Texto Plano (*.txt)",
        )
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(self.plain_editor.toPlainText())
            self.status_bar.showMessage(f"Archivo de texto guardado en: {out_path}", 5000)

    def _on_export_pdf(self) -> None:
        if not self.current_document:
            QMessageBox.information(
                self,
                "Sin documento procesado",
                "Primero extraiga el texto de una imagen antes de exportar el reporte PDF.",
            )
            return

        default_pdf = os.path.join(self.outputs_dir, "reporte_extraccion_estructurada.pdf")
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Reporte PDF Maquetado",
            default_pdf,
            "Documento PDF (*.pdf)",
        )
        if out_path:
            try:
                ReportExporter.export_pdf_report(
                    self.current_document,
                    out_path,
                    include_analytics=True,
                )
                self.status_bar.showMessage(f"Reporte PDF exportado exitosamente en: {out_path}", 6000)
                QMessageBox.information(
                    self,
                    "Reporte PDF Generado",
                    f"El reporte PDF maquetado se ha guardado correctamente en:\n\n{out_path}",
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Error al exportar PDF",
                    f"No se pudo generar el archivo PDF:\n{exc}",
                )
