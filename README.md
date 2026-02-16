

# Sistema OCR con Tesseract - Pipeline de Preprocesamiento

Sistema completo de OCR (Optical Character Recognition) con preprocesamiento avanzado de imágenes usando Tesseract y OpenCV. Optimizado para documentos en español con soporte para múltiples tipos de texto.

El proyecto fue desarrollado por:

Alejandro Rubiano
Juan Camilo San Miguel
Juan Sebastian Londoño

---

## ¿Qué hace este proyecto?

Extrae texto de imágenes de documentos (libros, facturas, manuscritos, etc.) usando:
- **Preprocesamiento inteligente** para mejorar la calidad de la imagen
- **Tesseract OCR** para reconocimiento de caracteres
- **Pipelines configurables** según el tipo de documento
- **Detección automática** de configuración óptima

---

## Características Principales

- **5 pipelines de preprocesamiento** para diferentes tipos de documentos
- **Configuración automática** de Tesseract en Windows/Linux/macOS
- **Soporte multiidioma** (español, inglés, y más)
- **Validación de imágenes** procesadas
- **Diagnóstico automático** para encontrar la mejor configuración
- **PSM dinámico** según el tipo de texto
- **Interfaz CLI** simple para uso rápido

---

## Requisitos

### Software
- Python 3.7+
- Tesseract OCR 4.0+

### Librerías Python
```bash
pip install opencv-python numpy pytesseract pillow matplotlib
```

### Tesseract + Idioma Español

**Windows:**
1. Descargar de: https://github.com/UB-Mannheim/tesseract/wiki
2. Durante instalación, marcar "Additional language data"
3. Seleccionar "Spanish" en la lista

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <tu-repo>
cd ocr-tesseract

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar Tesseract
tesseract --version
tesseract --list-langs  # Verificar que 'spa' esté en la lista
```

---

## Uso Rápido

### **Opción 1: Script de Inferencia (Más Simple)**

```bash
# Para páginas de libro
python inferencia.py imagen_libro.jpg libro

# Para documentos generales
python inferencia.py documento.jpg estandar

# Procesar toda una carpeta
python inferencia.py carpeta_imagenes/ libro
```

**Salida:**
- Texto en consola
- Archivo `.txt` con el resultado

---

### **Opción 2: Usando la Clase OCR (Más Control)**

```python
from ocr_pipeline import SistemaOCR

# Inicializar sistema
ocr = SistemaOCR(idioma='spa')

# Extraer texto
texto, resumen = ocr.leer_texto(
    'mi_imagen.jpg',
    preprocesar=True,
    pipeline='libro',
    tipo_texto='bloque',
    verbose=True
)

print(f"Texto extraído: {len(texto)} caracteres")
print(texto)
```

---

### **Opción 3: Solo Preprocesamiento**

```python
from utils import OCRPrepocesador
import cv2

# Cargar y preprocesar
prep = OCRPrepocesador()
imagen = cv2.imread('documento.jpg')
imagen_procesada = prep.preprocess(imagen, pipeline='libro')

# Guardar resultado
cv2.imwrite('procesada.jpg', imagen_procesada)
```

---

## Pipelines Disponibles

| Pipeline | Uso Recomendado | Características |
|----------|----------------|-----------------|
| `minimalista` | Imágenes limpias de alta calidad | Rápido, mínimo procesamiento |
| `estandar` | Documentos escaneados normales | Balanceado, uso general |
| `libro` | **Páginas de libro** ⭐ | Optimizado para libros, sin resize |
| `escritura` | Texto manuscrito | Preserva detalles finos |
| `agresivo` | Documentos degradados/antiguos | Máxima limpieza |

---

## Tipos de Texto (PSM)

| Tipo | PSM | Cuándo Usar |
|------|-----|-------------|
| `auto` | 3 | Detección automática (default) |
| `bloque` | 6 | Párrafos completos, libros |
| `linea` | 7 | Una sola línea de texto |
| `palabra` | 8 | Palabras aisladas |
| `disperso` | 11 | Texto irregular, formularios |

---

## Ejemplo Completo

```python
from ocr_pipeline import SistemaOCR

# 1. Inicializar
ocr = SistemaOCR(idioma='spa')

# 2. Opción A: Lectura simple
texto, _ = ocr.leer_texto('libro.jpg', pipeline='libro')
print(texto)

# 3. Opción B: Configuración detallada
texto, resumen = ocr.leer_texto(
    imagen_path='documento.jpg',
    preprocesar=True,
    pipeline='estandar',
    tipo_texto='bloque',
    verbose=True
)

print(f"Pipeline usado: {resumen}")
print(f"Caracteres: {len(texto)}")

# 4. Opción C: Diagnóstico automático
# Prueba todas las combinaciones y muestra la mejor
ocr.diagnosticar_imagen('mi_imagen.jpg')
```

**Salida del Diagnóstico:**
```
DIAGNÓSTICO: mi_imagen.jpg
============================================================

RESULTADOS (ordenados por caracteres detectados):
1. Pipeline: libro       | PSM: bloque    | Chars: 1354
   Preview: sabe dónde, comenzaron a difundirse ciertas declara-ciones...
2. Pipeline: estandar    | PSM: bloque    | Chars: 1280
   Preview: sabe dónde comenzaron a difundirse...
3. Pipeline: libro       | PSM: auto      | Chars: 1280
...

MEJOR CONFIGURACIÓN:
   Pipeline: libro
   PSM: bloque
   Caracteres: 1354
```

---

## Estructura del Proyecto

```
ocr-tesseract/
├── utils.py              # Preprocesador con 5 pipelines
├── ocr_pipeline.py       # Sistema OCR completo
├── inferencia.py         # Script CLI para uso rápido
├── README.md            # Este archivo

```

---

## Configuración Manual (Si es Necesario)

Si Tesseract no se detecta automáticamente:

**En `ocr_pipeline.py` o `inferencia.py`:**
```python
import pytesseract

# Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Linux/Mac (generalmente no necesario)
# Tesseract debe estar en PATH
```

---

## Solución de Problemas

### **Problema: "tesseract is not installed or it's not in your PATH"**

**Solución:**
1. Verificar instalación: `tesseract --version`
2. Si no está instalado, ver sección [Requisitos](#-requisitos)
3. Si está instalado, configurar ruta manualmente (ver arriba)

---

### **Problema: Texto vacío o 0 caracteres detectados**

**Causas comunes:**
1. Idioma español no instalado
2. Imagen demasiado pequeña después del resize
3. PSM incorrecto para el tipo de documento
4. Preprocesamiento demasiado agresivo

**Soluciones:**
```bash
# 1. Verificar idioma
tesseract --list-langs  # Debe aparecer 'spa'

# 2. Usar pipeline 'libro' para documentos (sin resize)
python inferencia.py imagen.jpg libro

# 3. Probar diagnóstico automático
python -c "from ocr_pipeline import SistemaOCR; ocr = SistemaOCR(); ocr.diagnosticar_imagen('imagen.jpg')"

# 4. Ver imagen procesada
# El script guarda 'imagen_procesada.jpg' en modo verbose
```

---

### **Problema: Resultados con muchos errores**

**Mejoras:**
1. Usar pipeline más agresivo: `agresivo`
2. Cambiar PSM: probar `disperso` (11) o `auto` (3)
3. Mejorar calidad de la imagen original
4. Probar con idioma combinado: `'spa+eng'`

---

## Ejemplo de Resultado

**Imagen original:** Página de libro de 1600x900px  
**Pipeline:** `libro`  
**PSM:** `6` (bloque)  
**Resultado:** 1,354 caracteres con ~95% de precisión

```
sabe dónde, comenzaron a difundirse ciertas declara-
ciones inquietantes, por no decir francamente amena-
zadoras, como por ejemplo, Quien no ponga la inmor-
tal bandera de la patria en la ventana de su casa no
merece estar vivo, Quienes no anden con la bandera
nacional bien a la vista es porque se han vendido a la
muerte, Únete, sé patriota, compra una bandera...
```

---

## Técnicas de Preprocesamiento

El sistema aplica las siguientes técnicas según el pipeline:

1. **Escala de grises** - Conversión de RGB a grayscale
2. **Reducción de ruido** - Gaussian, Median, Bilateral, NLM
3. **Mejora de contraste** - CLAHE, Histogram Equalization
4. **Binarización** - OTSU, Adaptive Threshold, Sauvola
5. **Operaciones morfológicas** - Opening, Closing, Erosion, Dilation
6. **Corrección de rotación** - Detección automática con Hough Transform
7. **Eliminación de bordes** - Recorte automático de márgenes
8. **Redimensionamiento** - Opcional según pipeline

---

## Casos de Uso

### **1. Digitalización de Libros**
```python
ocr = SistemaOCR(idioma='spa')
texto, _ = ocr.leer_texto('pagina.jpg', pipeline='libro', tipo_texto='bloque')
```

### **2. Extracción de Facturas**
```python
texto, _ = ocr.leer_texto('factura.jpg', pipeline='estandar', tipo_texto='disperso')
```

### **3. OCR de Manuscritos**
```python
texto, _ = ocr.leer_texto('manuscrito.jpg', pipeline='escritura', tipo_texto='linea')
```

### **4. Documentos Antiguos/Degradados**
```python
texto, _ = ocr.leer_texto('documento_antiguo.jpg', pipeline='agresivo', tipo_texto='auto')
```

---

## Fundamentos Académicos

Este proyecto implementa las mejores prácticas de OCR basadas en:

- **Preprocesamiento adaptativo** según características del documento
- **Binarización inteligente** para maximizar contraste texto-fondo
- **Page Segmentation Modes** apropiados según estructura del texto
- **Validación de resultados** en cada etapa del pipeline

**Referencias teóricas:**
- OTSU Thresholding (1979)
- CLAHE - Contrast Limited Adaptive Histogram Equalization
- Sauvola Binarization para documentos históricos
- Morphological Operations (Mathematical Morphology)

---

## 📈endimiento

| Tipo de Documento | Pipeline | Tiempo Promedio | Precisión |
|------------------|----------|-----------------|-----------|
| Libro moderno | `libro` | 2-3s | ~95% |
| Escaneo limpio | `estandar` | 1-2s | ~90% |
| Manuscrito | `escritura` | 3-4s | ~85% |
| Degradado | `agresivo` | 5-8s | ~80% |

*Tiempos en procesador i5, imagen de 1600x900px*

---

## Contribuciones

Este es un proyecto académico, pero las sugerencias son bienvenidas:
- Reportar bugs en Issues
- Proponer mejoras en Pull Requests
- Compartir casos de uso interesantes

---

## Licencia

Proyecto académico - Libre para uso educativo y de investigación

---


## Quick Start

```bash
# 1. Instalar
pip install opencv-python numpy pytesseract

# 2. Verificar Tesseract
tesseract --list-langs

# 3. Usar
python inferencia.py mi_imagen.jpg libro

# ¡Listo! 
```

---

**¿Dudas?** Revisa `TROUBLESHOOTING_GUIDE.md` para debugging avanzado.

---

