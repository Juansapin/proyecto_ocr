"""
Inferencia OCR con Tesseract
=========
Script para aplicar OCR sobre nuevas imágenes usando Tesseract.
Usa el preprocesador (utils.py) + Tesseract.

Autor: Juan S. Londoño

Uso:
    python inferencia.py imagen.jpg
    python inferencia.py imagen.jpg estandar
    python inferencia.py carpeta/
"""

import sys
import os
import cv2
import pytesseract
from utils import OCRPrepocesador

#  CONFIGURACIÓN DE TESSERACT 

def configurar_tesseract():
    """Configura Tesseract automáticamente."""
    if sys.platform == "win32":
        rutas_windows = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        for ruta in rutas_windows:
            if os.path.exists(ruta):
                pytesseract.pytesseract.tesseract_cmd = ruta
                print(f"✅ Tesseract encontrado: {ruta}")
                return True
        
        print("⚠️ Tesseract no encontrado automáticamente")
        print("   Configura manualmente la ruta si es necesario")
        return False
    return True

# Configurar Tesseract
configurar_tesseract()

# Verificar idiomas instalados
def verificar_idiomas():
    """Verifica los idiomas instalados en Tesseract."""
    try:
        langs = pytesseract.get_languages()
        print(f"📋 Idiomas disponibles: {', '.join(langs)}")
        
        if 'spa' not in langs:
            print("\n ADVERTENCIA: El idioma español NO está instalado")
            return False
        return True
    except:
        print("No se pudo verificar idiomas de Tesseract")
        return False


def leer_imagen(ruta_imagen: str, 
                pipeline: str = 'estandar',
                guardar_txt: bool = True):
    """
    Lee texto de una imagen usando Tesseract.
    
    Args:
        ruta_imagen: Ruta a la imagen
        pipeline: Pipeline de preprocesamiento
        guardar_txt: Si True, guarda resultado en .txt
    """
    print(f"\n📷 Procesando: {ruta_imagen}")
    print("-" * 60)
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    
    if imagen is None:
        raise ValueError(f"No se pudo cargar: {ruta_imagen}")
    
    # Inicializar preprocesador
    # Usar tamaño mayor para documentos (512 en vez de 128)
    preprocesador = OCRPrepocesador(tamano_estandar=512)
    
    # Preprocesar imagen
    print(f"🔧 Preprocesando con pipeline: {pipeline}")
    imagen_procesada = preprocesador.preprocess(imagen, pipeline=pipeline)
    
    # Validar imagen procesada
    if not preprocesador.validar_imagen(imagen_procesada):
        print("ADVERTENCIA: La imagen procesada parece vacía")
        print(" Prueba con otro pipeline")
    
    # Leer texto con Tesseract
    print(f"Leyendo texto (idioma: español)...")
    
    # Configuración de Tesseract Usar PSM apropiado según el tipo de documento
    if pipeline == 'libro':
        config = '--psm 6'  # Bloque uniforme para libros
    else:
        config = '--psm 3'  # Auto para otros documentos
    
    texto = pytesseract.image_to_string(
        imagen_procesada, 
        lang='spa',
        config=config
    )
    
    # Limpiar texto
    texto = texto.strip()
    
    # Mostrar resultado
    print(f"\nTexto detectado ({len(texto)} caracteres):")
    print("-" * 60)
    if len(texto) > 500:
        print(texto[:500] + "...")
        print(f"\n[... {len(texto) - 500} caracteres más ...]")
    elif len(texto) > 0:
        print(texto)
    else:
        print("NO SE DETECTÓ TEXTO")
        print("\nPosibles causas:")
        print("1. El idioma español no está instalado")
        print("2. El preprocesamiento eliminó el texto")
        print("3. La imagen no contiene texto legible")
        print("\nSoluciones:")
        print("- Verifica: tesseract --list-langs")
        print("- Prueba otro pipeline: minimalista, estandar, libro")
        print("- Guarda imagen procesada para verificar")
    
    print("-" * 60)
    
    # Guardar en .txt
    if guardar_txt and len(texto) > 0:
        # Obtener directorio actual en vez del de la imagen
        nombre_archivo = os.path.splitext(os.path.basename(ruta_imagen))[0]
        archivo_salida = f'{nombre_archivo}_ocr.txt'
        
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write("RESULTADO OCR - TESSERACT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Archivo:  {os.path.basename(ruta_imagen)}\n")
            f.write(f"Pipeline: {pipeline}\n")
            f.write(f"Caracteres detectados: {len(texto)}\n\n")
            f.write("-" * 60 + "\n")
            f.write("TEXTO:\n")
            f.write("-" * 60 + "\n\n")
            f.write(texto)
        
        print(f"\n💾 Resultado guardado en: {archivo_salida}")
    
    return texto


def leer_carpeta(ruta_carpeta: str,
                 pipeline: str = 'estandar',
                 guardar_txt: bool = True):
    """
    Lee texto de todas las imágenes en una carpeta.
    """
    print(f"\n📁 Procesando carpeta: {ruta_carpeta}")
    print("=" * 60)
    
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    imagenes = []
    for archivo in os.listdir(ruta_carpeta):
        ext = os.path.splitext(archivo)[1].lower()
        if ext in extensiones:
            imagenes.append(os.path.join(ruta_carpeta, archivo))
    
    if not imagenes:
        print("❌ No se encontraron imágenes")
        return
    
    print(f"✅ Encontradas {len(imagenes)} imágenes\n")
    
    resultados = []
    
    for i, ruta_imagen in enumerate(imagenes, 1):
        nombre = os.path.basename(ruta_imagen)
        print(f"[{i}/{len(imagenes)}] {nombre}")
        
        try:
            # Cargar y preprocesar
            imagen = cv2.imread(ruta_imagen)
            preprocesador = OCRPrepocesador(tamano_estandar=512)
            imagen_procesada = preprocesador.preprocess(imagen, pipeline=pipeline)
            
            # OCR
            config = '--psm 6' if pipeline == 'libro' else '--psm 3'
            texto = pytesseract.image_to_string(imagen_procesada, lang='spa', config=config)
            texto = texto.strip()
            
            print(f"   Detectado: {len(texto)} caracteres")
            
            resultados.append({
                'archivo': nombre,
                'texto': texto,
                'caracteres': len(texto)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            resultados.append({
                'archivo': nombre,
                'texto': f"ERROR: {e}",
                'caracteres': 0
            })
        
        print()
    
    # Guardar resultados
    if guardar_txt:
        archivo_salida = 'resultados_ocr.txt'
        
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS OCR - TESSERACT\n")
            f.write("=" * 60 + "\n\n")
            
            for resultado in resultados:
                f.write(f"Archivo: {resultado['archivo']}\n")
                f.write(f"Caracteres: {resultado['caracteres']}\n")
                f.write(f"Texto:\n\n")
                f.write(resultado['texto'])
                f.write("\n\n")
                f.write("-" * 60 + "\n\n")
        
        print(f"💾 Resultados guardados en: {archivo_salida}")
    
    return resultados


def main():
    """Función principal."""
    
    print("\n" + "=" * 60)
    print(" " * 20 + "OCR - INFERENCIA")
    print(" " * 25 + "Tesseract")
    print("=" * 60)
    
    # Verificar idiomas
    verificar_idiomas()
    
    if len(sys.argv) < 2:
        print("\n❌ Uso incorrecto")
        print("\nFormato:")
        print("  python inferencia.py imagen.jpg")
        print("  python inferencia.py imagen.jpg estandar")
        print("  python inferencia.py carpeta/")
        print("\nPipelines disponibles:")
        print("  - minimalista : Rápido para imágenes limpias")
        print("  - estandar    : Balanceado (recomendado)")
        print("  - libro       : Optimizado para páginas de libro (NUEVO)")
        print("  - escritura   : Para texto manuscrito")
        print("  - agresivo    : Para documentos degradados")
        print("\n💡 Para páginas de libro, usa el pipeline 'libro':")
        print("  python inferencia.py imagen.jpg libro")
        return
    
    ruta = sys.argv[1]
    pipeline = sys.argv[2] if len(sys.argv) > 2 else 'estandar'
    
    if not os.path.exists(ruta):
        print(f"\n❌ No existe: {ruta}")
        return
    
    if os.path.isfile(ruta):
        leer_imagen(ruta, pipeline=pipeline, guardar_txt=True)
    elif os.path.isdir(ruta):
        leer_carpeta(ruta, pipeline=pipeline, guardar_txt=True)
    else:
        print(f"\n❌ Ruta inválida: {ruta}")
    
    print("\n" + "=" * 60)
    print("✅ Proceso completado")
    print("=" * 60)


if __name__ == "__main__":
    main()