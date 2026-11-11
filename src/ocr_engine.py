"""
Motor de Extracción Óptica de Caracteres (OCR) y Preprocesamiento de Imágenes.

Decisiones técnicas:
1. **Motor Principal (`rapidocr_onnxruntime`):** Se eligió RapidOCR basado en ONNX Runtime
   sobre los modelos PP-OCRv4 porque:
   - Es 100% local, offline y gratuito, sin depender de binarios externos del sistema
     (como Tesseract instalado en `C:\\Program Files`) ni de descargas de PyTorch de 2 GB.
   - Entrega polígonos precisos de 4 vértices y puntuaciones de confianza por cada línea/bloque
     detectado, lo cual es indispensable para nuestro algoritmo de reconstrucción geométrica.
2. **Preprocesamiento Adaptativo (OpenCV / Pillow):**
   - Antes de pasar la imagen al OCR, se ofrece un pipeline opcional de mejora de contraste
     mediante CLAHE (Contrast Limited Adaptive Histogram Equalization) en el canal de luminancia
     (espacio LAB) y reducción de ruido bilateral, mejorando drásticamente la lectura en fotos
     de documentos con sombras o iluminación desigual.
   - Se utiliza `np.fromfile` + `cv2.imdecode` en lugar de `cv2.imread` para soportar rutas con
     caracteres Unicode/acentos en Windows (ej. `d:\\Proyectos\\extracción de texto`).
"""

import logging
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.models import TextRegion

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Motor de reconocimiento óptico de caracteres con preprocesamiento de imagen
    y extracción geométrica de cajas delimitadoras.
    """

    def __init__(self, confidence_threshold: float = 0.35) -> None:
        """
        Inicializa el motor OCR ONNX Runtime de forma perezosa o inmediata.

        Args:
            confidence_threshold: Umbral mínimo de confianza [0.0, 1.0] para conservar
                una región detectada.
        """
        self.confidence_threshold = confidence_threshold
        self._ocr_instance = None

    def _ensure_engine(self) -> None:
        """Inicializa la instancia de RapidOCR sólo cuando se requiere."""
        if self._ocr_instance is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr_instance = RapidOCR()

    @staticmethod
    def load_image_unicode(image_path: str) -> np.ndarray:
        """
        Carga una imagen desde el disco soportando rutas con tildes, eñes o caracteres Unicode
        en sistemas Windows.

        Args:
            image_path: Ruta al archivo de imagen.

        Returns:
            Matriz BGR de OpenCV (`np.ndarray`).

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el archivo no es una imagen válida.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró la imagen en la ruta: {image_path}")

        raw_bytes = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback mediante Pillow para formatos especiales (ej. WebP animados o TIFF complejos)
            with Image.open(image_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        return img

    @staticmethod
    def preprocess_image(
        bgr_image: np.ndarray,
        enhance_contrast: bool = True,
        denoise: bool = False,
        upscale_small: bool = True,
    ) -> np.ndarray:
        """
        Aplica mejoras fotométricas suaves que preservan los bordes tipográficos
        para maximizar la precisión del detector y reconocedor OCR.

        Args:
            bgr_image: Imagen de entrada en formato BGR.
            enhance_contrast: Si es True, aplica ecualización adaptativa CLAHE sobre luminancia.
            denoise: Si es True, aplica filtrado bilateral que reduce ruido preservando bordes.
            upscale_small: Si es True y el ancho es menor a 900px, reescala con interpolación cúbica.

        Returns:
            Imagen preprocesada en formato BGR (`np.ndarray`).
        """
        processed = bgr_image.copy()
        height, width = processed.shape[:2]

        if upscale_small and max(height, width) < 900:
            scale = 1200.0 / float(max(height, width))
            new_w = int(round(width * scale))
            new_h = int(round(height * scale))
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        if denoise:
            processed = cv2.bilateralFilter(processed, d=5, sigmaColor=40, sigmaSpace=40)

        if enhance_contrast:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            merged = cv2.merge((cl, a_channel, b_channel))
            processed = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        return processed

    def extract_regions(
        self,
        image_path: str,
        enhance_contrast: bool = False,
        denoise: bool = False,
    ) -> Tuple[List[TextRegion], Tuple[int, int], float]:
        """
        Ejecuta la extracción OCR sobre una imagen y devuelve las regiones detectadas
        junto con las dimensiones de trabajo y el tiempo empleado.

        Args:
            image_path: Ruta al archivo de imagen.
            enhance_contrast: Activa mejora adaptativa de contraste CLAHE.
            denoise: Activa reducción de ruido bilateral.

        Returns:
            Tupla `(regiones, (ancho, alto), tiempo_ms)`.
        """
        start_time = time.perf_counter()
        self._ensure_engine()

        original_bgr = self.load_image_unicode(image_path)
        orig_h, orig_w = original_bgr.shape[:2]

        processed_bgr = self.preprocess_image(
            original_bgr,
            enhance_contrast=enhance_contrast,
            denoise=denoise,
            upscale_small=True,
        )
        proc_h, proc_w = processed_bgr.shape[:2]

        scale_x = float(orig_w) / float(proc_w)
        scale_y = float(orig_h) / float(proc_h)

        result, _ = self._ocr_instance(processed_bgr)
        regions: List[TextRegion] = []

        if result:
            for item in result:
                if len(item) < 3:
                    continue
                raw_poly, raw_text, raw_conf = item[0], item[1], item[2]
                text_clean = str(raw_text).strip()
                try:
                    conf = float(raw_conf)
                except (TypeError, ValueError):
                    conf = 0.0

                if not text_clean or conf < self.confidence_threshold:
                    continue

                polygon: List[Tuple[float, float]] = []
                xs: List[float] = []
                ys: List[float] = []
                for pt in raw_poly:
                    px = float(pt[0]) * scale_x
                    py = float(pt[1]) * scale_y
                    polygon.append((px, py))
                    xs.append(px)
                    ys.append(py)

                regions.append(
                    TextRegion(
                        text=text_clean,
                        confidence=min(1.0, max(0.0, conf)),
                        polygon=polygon,
                        x_min=min(xs),
                        y_min=min(ys),
                        x_max=max(xs),
                        y_max=max(ys),
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return regions, (orig_w, orig_h), elapsed_ms
