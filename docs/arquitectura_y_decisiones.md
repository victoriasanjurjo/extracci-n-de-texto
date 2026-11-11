# Documentación Técnica y Decisiones de Arquitectura — DocuStruct Studio

## 1. Propósito y Alcance del Sistema

**DocuStruct Studio** es una aplicación de escritorio escrita en Python 3.12 con **PySide6** diseñada para resolver un problema clásico de los sistemas OCR convencionales: la pérdida de estructura lógica y coherencia al extraer texto de imágenes.

Cuando un OCR estándar procesa una imagen, entrega una lista plana de fragmentos aislados. **DocuStruct Studio** añade una capa de **Reconstrucción Geométrica y Semántica (`StructureEngine`)** que analiza las coordenadas espaciales $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$, las alturas tipográficas relativas, las brechas horizontales/verticales y los patrones sintácticos para organizar el contenido en una jerarquía coherente:

1. **Títulos Principales (`TITLE` / H1)**
2. **Subtítulos y Secciones (`SUBTITLE` / H2-H3)**
3. **Pares Clave-Valor de Metadatos (`KEY_VALUE`)**
4. **Párrafos Cohesivos Multi-línea (`PARAGRAPH`)** con recomposición de palabras cortadas por guion (`-\n`).
5. **Listas de Viñetas y Numeradas (`LIST_ITEM`)**
6. **Tablas Multi-Columna (`TABLE`)** alineadas espacialmente en filas y columnas.
7. **Notas y Bloques Destacados (`QUOTE_OR_NOTE`)**

---

## 2. Justificación de Dependencias y Decisiones Técnicas

### 2.1. Motor OCR Local: `rapidocr_onnxruntime` + `opencv-python` + `Pillow`
- **Por qué se eligió:** A diferencia de `pytesseract` (que falla si el usuario no instala manualmente el binario de Tesseract en el sistema operativo y configura el `PATH`) o `easyocr` (que arrastra ~2 GB de dependencias de PyTorch), **`rapidocr_onnxruntime`** ejecuta los modelos PP-OCRv4 directamente sobre **ONNX Runtime** en CPU de forma 100% autónoma, rápida (~350–800 ms por página) y sin conexión a internet.
- **Preprocesamiento Fotométrico (`OCREngine.preprocess_image`):**
  - Se aplica ecualización adaptativa de histograma **CLAHE** en el canal $L$ del espacio de color $LAB$ para mejorar el contraste local en fotografías de documentos con iluminación irregular.
  - Se ofrece filtrado bilateral (`cv2.bilateralFilter`) para atenuar el ruido granulado sin difuminar los bordes de los caracteres.
  - La lectura desde disco se realiza con `np.fromfile` + `cv2.imdecode` para garantizar compatibilidad total con rutas que contengan tildes, eñes o espacios en Windows.

### 2.2. Algoritmo de Reconstrucción Estructural (`src/structure_engine.py`)
El motor `StructureEngine` opera en 5 etapas deterministas y auditables:

1. **Clustering Horizontal de Líneas (`_group_regions_into_lines`):**
   - Ordena las cajas detectadas por su centro vertical $y_c$.
   - Dos regiones se asignan a la misma línea si su solapamiento vertical supera el $42\%$ de la menor altura o si $|y_{c,1} - y_{c,2}| \le 0.55 \cdot \min(h_1, h_2)$.
   - Dentro de cada línea, fusiona fragmentos adyacentes cuya separación horizontal sea menor a $\max(2.4 \cdot \bar{w}_{\text{char}}, 1.15 \cdot \bar{h})$, pero preserva las celdas separadas cuando existe una brecha de columna tabular.
2. **Reconstrucción de Tablas (`_build_table_matrix`):**
   - Cuando dos o más líneas consecutivas presentan múltiples celdas separadas por brechas horizontales, proyecta las coordenadas $x_{\min}$ de todas las celdas y las agrupa en anclajes de columna para reconstruir una matriz bidimensional exacta.
3. **Referencia Tipográfica Robusta:**
   - Estima la altura tipográfica del cuerpo del texto usando el percentil 45 de las alturas de línea ($h_{\text{body}} = P_{45}(H)$), evitando que documentos con varios encabezados distorsionen la línea base.
4. **Segmentación y Clasificación Jerárquica (`_segment_lines_into_blocks` y `_inspect_single_line`):**
   - Evalúa razones de altura ($h / h_{\text{body}} \ge 1.26$ para H1 y $\ge 1.11$ para H2), numeración de secciones, viñetas, pares `Clave: Valor` y saltos de interlineado ($> 1.40 \cdot \bar{h}$) para agrupar líneas continuas en párrafos cohesivos.
5. **Generación Multiformato:**
   - Produce salidas sincronizadas en **HTML Enriquecido** (con insignias de confianza y tipo de bloque), **Markdown GFM** (`.md`), **Texto Plano Organizado** (`.txt`) y **Reporte PDF Maquetado** (`.pdf`).

### 2.3. Análisis Estadístico y Reporte PDF (`pandas`, `numpy`, `matplotlib`, `seaborn`, `reportlab`)
- **`pandas` y `numpy`:** Estructuran las métricas de cada bloque (conteo de palabras, caracteres, coordenadas y estadísticas de confianza media, mediana, mínima y máxima).
- **`matplotlib` y `seaborn`:** Renderizan un gráfico analítico dual en alta definición (`outputs/ultimo_analisis_bloques.png`) con la composición estructural del documento y la curva de confianza OCR por bloque.
- **`reportlab`:** Maqueta el archivo `outputs/reporte_extraccion_estructurada.pdf` con cabecera formal, resumen ejecutivo, tabla de KPIs, gráfico estadístico embebido y la transcripción íntegra estructurada del documento.

### 2.4. Diseño de Interfaz Gráfica (`PySide6`) y Empaquetado (`PyInstaller`)
- **Estética Visual ("Warm Paper & Deep Teal Studio"):**
  Se diseñó una paleta clara, editorial y descansada para la vista (`#F7F5F0` fondo marfil cálido, `#FFFFFF` paneles elevados, `#0F766E` acento principal verde azulado profundo, `#D97706` acento secundario ámbar terracota y `#0F172A` texto pizarra de alto contraste).
- **Concurrencia (`QThread`):** Todo el trabajo pesado de OCR y análisis ocurre en `OCRWorkerThread`, manteniendo la ventana 100% responsiva.
- **Compilación Autónoma (`PyInstaller`):** Se empaqueta con `--windowed` e ícono personalizado (`assets/app_icon.ico`), recolectando los modelos ONNX de `rapidocr_onnxruntime` para que el ejecutable funcione directamente.

