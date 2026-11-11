"""
Generador de recursos visuales (`assets/`):
1. Ícono representativo de la aplicación (`assets/app_icon.png` y `assets/app_icon.ico`),
   con estética editorial limpia (documento estilizado con capas de estructura jerárquica y lente de escaneo).
2. Imagen de documento de demostración (`assets/sample_document.png`), que incluye un título,
   subtítulos, metadatos clave-valor, párrafo multi-línea, lista de viñetas y una tabla financiera,
   permitiendo probar de inmediato todo el potencial de extracción y estructuración coherente.
"""

import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


def ensure_app_icon(assets_dir: str = "assets") -> Tuple[str, str]:
    """
    Genera el ícono oficial de DocuStruct Studio en formatos `.png` y `.ico`
    dentro de la carpeta `assets/`.

    Returns:
        Tupla `(ruta_png, ruta_ico)`.
    """
    os.makedirs(assets_dir, exist_ok=True)
    png_path = os.path.join(assets_dir, "app_icon.png")
    ico_path = os.path.join(assets_dir, "app_icon.ico")

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Fondo redondeado en verde azulado editorial profundo (#0F766E)
    draw.rounded_rectangle(
        [(12, 12), (size - 12, size - 12)],
        radius=48,
        fill=(15, 118, 110, 255),
        outline=(13, 148, 136, 255),
        width=4,
    )

    # Hoja de papel marfil cálido en el centro
    page_box = [(52, 40), (204, 216)]
    draw.rounded_rectangle(page_box, radius=14, fill=(252, 250, 245, 255))

    # Bloques estructurados representados dentro de la hoja
    # 1. Bloque de Título (Ámbar terracota #D97706)
    draw.rounded_rectangle([(70, 60), (168, 76)], radius=4, fill=(217, 119, 6, 255))

    # 2. Líneas de Párrafo (Azul pizarra #334155)
    draw.rounded_rectangle([(70, 92), (186, 102)], radius=3, fill=(51, 65, 85, 255))
    draw.rounded_rectangle([(70, 110), (180, 120)], radius=3, fill=(71, 85, 105, 255))
    draw.rounded_rectangle([(70, 128), (154, 138)], radius=3, fill=(71, 85, 105, 255))

    # 3. Cuadrícula de Tabla estructurada (Verde esmeralda/azul)
    for row_idx in range(2):
        for col_idx in range(2):
            x0 = 70 + col_idx * 58
            y0 = 152 + row_idx * 22
            draw.rounded_rectangle(
                [(x0, y0), (x0 + 50, y0 + 14)],
                radius=3,
                fill=(15, 118, 110, 220) if row_idx == 0 else (148, 163, 184, 255),
            )

    # Marco de escaneo inteligente (esquinas resaltadas en ámbar brillante)
    corner_color = (251, 191, 36, 255)
    w = 6
    # Esquina superior izquierda
    draw.line([(36, 56), (36, 36), (56, 36)], fill=corner_color, width=w)
    # Esquina superior derecha
    draw.line([(200, 36), (220, 36), (220, 56)], fill=corner_color, width=w)
    # Esquina inferior izquierda
    draw.line([(36, 200), (36, 220), (56, 220)], fill=corner_color, width=w)
    # Esquina inferior derecha
    draw.line([(200, 220), (220, 220), (220, 200)], fill=corner_color, width=w)

    img.save(png_path, format="PNG")
    img.save(
        ico_path,
        format="ICO",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
    )
    return png_path, ico_path


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Carga una fuente TrueType con excelente espaciado entre palabras (Verdana/Tahoma/Segoe UI)."""
    candidates = (
        ["verdanab.ttf", "tahomabd.ttf", "segoeuib.ttf", "arialbd.ttf"]
        if bold
        else ["verdana.ttf", "tahoma.ttf", "segoeui.ttf", "arial.ttf"]
    )
    win_fonts_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
    for name in candidates:
        full_path = os.path.join(win_fonts_dir, name)
        if os.path.exists(full_path):
            try:
                return ImageFont.truetype(full_path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def generate_sample_document_image(output_path: str = "assets/sample_document.png") -> str:
    """
    Genera una imagen de alta nitidez que simula un informe ejecutivo escaneado,
    con título, metadatos clave-valor, subtítulos, párrafo multi-línea, lista de viñetas
    y una tabla comparativa de varias columnas.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    width, height = 1150, 1480
    img = Image.new("RGB", (width, height), (252, 251, 247))
    draw = ImageDraw.Draw(img)

    # Borde sutil de página
    draw.rectangle([(35, 35), (width - 35, height - 35)], outline=(203, 213, 225), width=2)

    font_title = _load_font(38, bold=True)
    font_subtitle = _load_font(26, bold=True)
    font_body = _load_font(20, bold=False)
    font_bold = _load_font(20, bold=True)

    y = 75
    left_x = 80

    # 1. Título Principal (H1)
    draw.text(
        (left_x, y),
        "Informe de Operaciones y Auditoria 2026",
        fill=(15, 23, 42),
        font=font_title,
    )
    y += 72

    # 2. Pares Clave-Valor (Metadatos)
    kv_items = [
        ("Documento: INF-2026-089"),
        ("Fecha de emision: 11 de noviembre de 2026"),
        ("Departamento: Inteligencia Documental"),
        ("Estado de revision: Aprobado para publicacion"),
    ]
    for kv in kv_items:
        draw.text((left_x, y), kv, fill=(30, 41, 59), font=font_body)
        y += 38

    y += 28

    # 3. Subtítulo 1 (H2)
    draw.text(
        (left_x, y),
        "1. Resumen del Desempeno Trimestral",
        fill=(15, 118, 110),
        font=font_subtitle,
    )
    y += 48

    # 4. Párrafo continuo multi-línea (se unirá coherentemente en un único bloque PARAGRAPH)
    paragraph_lines = [
        "Durante el tercer trimestre del ejercicio fiscal se implemento la nueva arquitectura",
        "de digitalizacion automatica en las tres sedes regionales, lo cual permitio reducir",
        "los tiempos de clasificacion manual y elimino los errores de transcripcion en facturas",
        "y contratos logisticos de alta complejidad operativa.",
    ]
    for pline in paragraph_lines:
        draw.text((left_x, y), pline, fill=(30, 41, 59), font=font_body)
        y += 36

    y += 32

    # 5. Subtítulo 2 (H2)
    draw.text(
        (left_x, y),
        "2. Hallazgos Principales de la Auditoria",
        fill=(15, 118, 110),
        font=font_subtitle,
    )
    y += 48

    # 6. Lista de viñetas
    bullets = [
        "- Incremento del 42% en la velocidad de procesamiento de expedientes.",
        "- Estandarizacion de reportes ejecutivos en formatos Markdown y PDF.",
        "- Reduccion del 68% en incidencias de archivo y perdida de metadatos.",
    ]
    for b_text in bullets:
        draw.text((left_x + 15, y), b_text, fill=(30, 41, 59), font=font_body)
        y += 44

    y += 34

    # 7. Subtítulo 3 (H2)
    draw.text(
        (left_x, y),
        "3. Metricas por Unidad Operativa",
        fill=(15, 118, 110),
        font=font_subtitle,
    )
    y += 52

    # 8. Tabla de 3 columnas con brechas horizontales claras
    col_x = [left_x, 480, 810]
    table_rows = [
        ("Region Operativa", "Volumen Procesado", "Exactitud Media"),
        ("Sede Norte", "14520 documentos", "98.4%"),
        ("Sede Central", "28910 documentos", "99.1%"),
        ("Sede Sur", "11340 documentos", "97.8%"),
    ]
    for r_idx, row in enumerate(table_rows):
        row_font = font_bold if r_idx == 0 else font_body
        row_color = (15, 23, 42) if r_idx == 0 else (51, 65, 85)
        for c_idx, cell_val in enumerate(row):
            draw.text((col_x[c_idx], y), cell_val, fill=row_color, font=row_font)
        y += 40

    y += 38

    # 9. Nota final destacada
    draw.text(
        (left_x, y),
        "Nota: Todos los indicadores fueron verificados mediante muestreo estadistico interno.",
        fill=(159, 18, 57),
        font=font_bold,
    )

    img.save(output_path, format="PNG")
    return output_path
