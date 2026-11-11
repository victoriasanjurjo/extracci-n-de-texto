"""Configuración de ruta de búsqueda para ejecución de pruebas con pytest."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

