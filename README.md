# DocuStruct Studio — Extracción Óptica y Organización Estructural Coherente de Texto

**DocuStruct Studio** es una aplicación de escritorio desarrollada en **Python** con interfaz gráfica moderna en **PySide6** que permite cargar cualquier imagen de un documento (informes, facturas, apuntes, contratos, tablas o capturas) y **no sólo extraer su texto mediante OCR local, sino reconstruir de forma inteligente su estructura y jerarquía lógica**.

---

## 1. ¿Qué es y para qué sirve?

La mayoría de las herramientas de reconocimiento óptico de caracteres (OCR) devuelven líneas sueltas o texto desordenado que obliga al usuario a reconstruir manualmente los párrafos, títulos, listas y tablas.

**DocuStruct Studio** resuelve este problema combinando un motor OCR neuronal offline (**RapidOCR / PP-OCRv4 sobre ONNX Runtime**) con un **Motor de Reconstrucción Estructural (`StructureEngine`)** que analiza la geometría espacial de cada fragmento detectado para organizar el documento automáticamente en:

- **Títulos Principales (`H1`) y Subtítulos de Sección (`H2/H3`)** según su proporción tipográfica, aislamiento vertical y sintaxis.
- **Metadatos y Pares Clave-Valor (`Clave: Valor`)** habituales en cabeceras de informes, recibos y formularios.
- **Párrafos Cohesivos Multi-línea**, uniendo oraciones continuas y recomponiendo palabras cortadas con guion al final de línea.
- **Listas de Viñetas y Numeradas**, agrupadas semánticamente.
- **Tablas Multi-Columna**, detectando brechas horizontales y alineando cada celda en su columna correspondiente tanto en Markdown como en PDF y vista enriquecida.
- **Reportes Analíticos en PDF (`outputs/`)**, con resumen ejecutivo, métricas de calidad OCR, gráficos estadísticos (`matplotlib` + `seaborn`) y el documento íntegro maquetado con `reportlab`.

---

## 2. Arquitectura y Componentes del Sistema

El proyecto sigue una arquitectura modular limpia con separación estricta de responsabilidades:

1. **`src/models.py` — Modelos de Dominio Tipados:**
   Define las estructuras de datos inmutables/tipadas (`TextRegion`, `ReconstructedLine`, `DocumentBlock`, `BlockType`, `DocumentAnalytics` y `StructuredDocument`).
2. **`src/ocr_engine.py` — Motor OCR y Preprocesamiento Fotométrico (`OCREngine`):**
   Gestiona la carga segura de imágenes con soporte Unicode en Windows, aplica mejoras adaptativas de contraste (**CLAHE** en espacio LAB) y filtrado bilateral de ruido con **OpenCV**, y extrae polígonos de texto y niveles de confianza mediante **RapidOCR (ONNX Runtime)**.
3. **`src/structure_engine.py` — Motor de Reconstrucción Estructural (`StructureEngine`):**
   Implementa el algoritmo de 5 etapas: agrupación horizontal en líneas, detección de brechas tabulares, estimación robusta de altura tipográfica base, segmentación en bloques semánticos y formateo coherente a **Markdown (`.md`)**, **Texto Plano (`.txt`)** y **HTML Enriquecido**.
4. **`src/exporter.py` — Analítica de Datos y Exportador de Reportes (`ReportExporter`):**
   Construye un `pandas.DataFrame` con las métricas de cada bloque, genera gráficos estadísticos con `matplotlib` y `seaborn`, y maqueta reportes ejecutivos en **PDF** mediante `reportlab`.
5. **`src/gui.py` — Interfaz Gráfica Editorial en PySide6 (`MainWindow`, `AnnotatedImageCanvas`, `OCRWorkerThread`):**
   Presenta una interfaz clara y profesional (*Warm Paper & Deep Teal Studio*) con soporte *Drag & Drop*, superposición visual de bloques coloreados sobre la imagen original, tarjetas de KPIs en tiempo real, 4 pestañas de inspección/edición y procesamiento asíncrono en segundo plano (`QThread`).
6. **`src/icon_generator.py` — Generador de Identidad Visual y Documento Demo:**
   Genera programáticamente el ícono oficial de la aplicación (`assets/app_icon.ico` y `.png`) y un documento ejecutivo de prueba (`assets/sample_document.png`).

---

## 3. Árbol de Directorios del Proyecto

```text
extracción de texto/
├── docs/
│   └── arquitectura_y_decisiones.md      # Justificación técnica, algoritmos y trazabilidad
├── tests/
│   ├── conftest.py                       # Configuración de rutas para pytest
│   └── test_structure_and_ocr.py         # Pruebas unitarias y de integración extremo a extremo
├── config/
│   └── settings.json                     # Parámetros de configuración OCR y umbrales geométricos
├── assets/
│   ├── app_icon.ico                      # Ícono oficial multi-resolución para Windows y PyInstaller
│   ├── app_icon.png                      # Ícono de alta resolución (256x256)
│   └── sample_document.png               # Documento de demostración autogenerado
├── outputs/
│   ├── reporte_extraccion_estructurada.pdf               # Reporte ejecutivo PDF maquetado
│   ├── reporte_extraccion_estructurada_analytics_chart.png # Gráfico estadístico de bloques y confianza
│   ├── documento_estructurado.md                         # Salida estructurada en Markdown GFM
│   └── documento_estructurado.txt                        # Salida organizada en Texto Plano
├── src/
│   ├── __init__.py
│   ├── models.py                         # Dataclasses y enumeraciones de bloques
│   ├── ocr_engine.py                     # Preprocesamiento OpenCV + RapidOCR ONNX
│   ├── structure_engine.py               # Algoritmo de organización coherente y jerárquica
│   ├── exporter.py                       # Pandas/Seaborn analytics + ReportLab PDF
│   ├── icon_generator.py                 # Generación de íconos y documento de prueba
│   └── gui.py                            # Interfaz gráfica PySide6
├── .venv/                                # Entorno virtual aislado de Python 3.12
├── .gitignore                            # Exclusiones de control de versiones
├── LICENSE.txt                           # Licencia MIT completa y vigente
├── requirements.txt                      # Dependencias con versiones exactas fijadas
├── README.md                             # Guía general y documentación del proyecto
└── main.py                               # Punto de entrada principal (GUI y CLI --batch-demo)
```

---

## 4. Instrucciones Paso a Paso para Ejecutar el Proyecto

### 4.1. Requisitos Previos
- **Sistema Operativo:** Windows 10 / 11 (64 bits).
- **Python:** 3.12 (el proyecto incluye su entorno virtual `.venv` preconfigurado).

### 4.2. Activación del Entorno Virtual e Instalación de Dependencias

```powershell
# 1. Situarse en la carpeta del proyecto
cd "d:\Proyectos\extracción de texto"

# 2. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 3. (Opcional) Verificar o reinstalar dependencias exactas
pip install -r requirements.txt
```

### 4.3. Ejecución de la Aplicación Gráfica (GUI)

```powershell
.\.venv\Scripts\python.exe main.py
```

Al abrirse la ventana de **DocuStruct Studio**:
1. La aplicación carga y analiza automáticamente el documento de demostración (`assets/sample_document.png`), mostrando las cajas estructurales coloreadas a la izquierda y el texto organizado a la derecha.
2. Puedes **arrastrar y soltar cualquier imagen** sobre el lienzo izquierdo o hacer clic en **"Cargar Imagen..."**.
3. Navega entre las pestañas **"Vista Estructurada (Jerárquica)"**, **"Markdown Coherente (.md)"**, **"Texto Plano Organizado (.txt)"** e **"Inspector de Bloques y Estadísticas"**.
4. Haz clic en **"Exportar Reporte PDF"** para guardar un informe PDF maquetado con métricas, gráficos y el texto estructurado.

### 4.4. Ejecución en Modo Batch / Demostración por Consola

Para procesar el documento de muestra y regenerar todos los reportes en `outputs/` desde la terminal:

```powershell
.\.venv\Scripts\python.exe main.py --batch-demo
```

### 4.5. Ejecución de las Pruebas Automatizadas

```powershell
.\.venv\Scripts\pytest.exe -v
```

### 4.6. Ejecución del Binario Compilado (`.exe`)

El proyecto se compila automáticamente con **PyInstaller** generando el ejecutable autónomo en:

```text
dist\DocuStructStudio\DocuStructStudio.exe
```

---

## 5. Licencia

Distribuido bajo la licencia **MIT**. Consulta el archivo `LICENSE.txt` para más información.

