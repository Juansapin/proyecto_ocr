
# Pipeline OCR con Tesseract

import cv2
import numpy as np
import pytesseract
from utils import OCRPrepocesador
from typing import Tuple, Optional
import sys
import os

#  CONFIGURACIÓN DE TESSERACT 

def configurar_tesseract():
    """Configura la ruta de Tesseract automáticamente para Windows."""
    if sys.platform == "win32":
        # Rutas comunes en Windows
        rutas_windows = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        for ruta in rutas_windows:
            if os.path.exists(ruta):
                pytesseract.pytesseract.tesseract_cmd = ruta
                print(f"Tesseract encontrado en: {ruta}")
                return True
        
        print("Tesseract no encontrado en rutas comunes de Windows")
        print("Si Tesseract está instalado, configura la ruta manualmente:")
        print("pytesseract.pytesseract.tesseract_cmd = r'RUTA_A_TESSERACT\\tesseract.exe'")
        return False
    
    return True

# Ejecutar configuración al importar
configurar_tesseract()

class SistemaOCR:
    
    def __init__(self, idioma: str = 'spa'):
        """
        Args:
            idioma: Idioma para Tesseract ('spa', 'eng', 'spa+eng')
        """
        print("Iniciando Sistema OCR con Tesseract")
        
        # Inicializar preprocesador
        self.preprocesador = OCRPrepocesador(tamano_estandar=128) # Se puede ajustar el tamaño dependiendo de la calidad de la imagen
        self.idioma = idioma
        
        # Verificar que Tesseract funciona
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract versión: {version}")
        except Exception as e:
            print(f"Error al conectar con Tesseract: {e}")
            print("   Verifica que Tesseract esté instalado correctamente")
        
        # Verificar idiomas disponibles
        try:
            langs = pytesseract.get_languages()
            print(f"Idiomas disponibles: {', '.join(langs)}")
            
            if idioma not in langs and idioma != 'spa+eng':
                print(f"ADVERTENCIA: El idioma '{idioma}' no está instalado")
                print(f"   Idiomas disponibles: {', '.join(langs)}")
                if 'spa' not in langs:
                    print("\n   Para instalar español:")
                    print("   - Windows: Reinstalar Tesseract con 'Additional language data'")
                    print("   - Linux: sudo apt-get install tesseract-ocr-spa")
                    print("   - macOS: brew install tesseract-lang")
        except:
            pass
        
        print(f"Sistema OCR listo (idioma: {idioma})")
    
    def _seleccionar_psm(self, tipo_texto: str = 'auto') -> int:
        """
        Selecciona el PSM (Page Segmentation Mode) apropiado.
        
        Args:
            tipo_texto: Tipo de texto esperado
                - 'auto': Detección automática (PSM 3)
                - 'bloque': Bloque uniforme (PSM 6)
                - 'linea': Una sola línea (PSM 7)
                - 'palabra': Una sola palabra (PSM 8)
                - 'disperso': Texto disperso (PSM 11)
                
        Returns:
            Número PSM para Tesseract
        """
        psm_map = {
            'auto': 3,
            'bloque': 6,
            'linea': 7,
            'palabra': 8,
            'disperso': 11
        }
        
        return psm_map.get(tipo_texto, 3)
    
    def leer_texto(self, 
                   imagen_path: str, 
                   preprocesar: bool = True,
                   pipeline: str = 'estandar',
                   tipo_texto: str = 'auto',
                   verbose: bool = True) -> Tuple[str, str]:
        """
        Lee texto de una imagen.
        
        Args:
            imagen_path: Ruta a la imagen
            preprocesar: Si True, aplica preprocesamiento
            pipeline: Tipo de pipeline ('estandar', 'agresivo', 'minimalista', 'escritura', 'libro')
            tipo_texto: Tipo de texto esperado ('auto', 'bloque', 'linea', 'palabra', 'disperso')
            verbose: Si True, imprime información de debug
            
        Returns:
            Tupla de (texto_completo, resumen_pipeline)
        """
        # Cargar imagen
        imagen = cv2.imread(imagen_path)
        
        if imagen is None:
            raise ValueError(f"No se pudo cargar: {imagen_path}")
        
        if verbose:
            print(f"📷 Imagen cargada: {imagen.shape}")
        
        # Preprocesar si se solicita
        if preprocesar:
            imagen_procesada = self.preprocesador.preprocess(imagen, pipeline=pipeline)
            resumen = self.preprocesador.resumen_pipeline()
            
            # Validar imagen procesada
            if not self.preprocesador.validar_imagen(imagen_procesada):
                print("ADVERTENCIA: La imagen procesada parece vacía")
                print("Intenta con otro pipeline o sin preprocesamiento")
            
            if verbose:
                print(f"Pipeline: {resumen}")
                print(f"Imagen procesada: {imagen_procesada.shape}")
        else:
            # Si no preprocesa, solo convertir a gris
            if len(imagen.shape) == 3:
                imagen_procesada = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                imagen_procesada = imagen
            resumen = "Sin preprocesamiento"
        
        # Seleccionar PSM apropiado
        psm = self._seleccionar_psm(tipo_texto)
        config = f'--psm {psm}'
        
        if verbose:
            print(f"📖 Leyendo con PSM {psm} ({tipo_texto})")
        
        # Leer texto con Tesseract
        try:
            texto = pytesseract.image_to_string(
                imagen_procesada, 
                lang=self.idioma,
                config=config
            )
            
            # Limpiar texto
            texto = texto.strip()
            
            if verbose:
                print(f"✅ Texto detectado: {len(texto)} caracteres")
                if len(texto) == 0:
                    print("TEXTO VACÍO - Posibles causas:")
                    print("1. El preprocesamiento eliminó el texto")
                    print("2. El PSM no es apropiado para este tipo de imagen")
                    print("3. La imagen no contiene texto legible")
                    print("4. El idioma de Tesseract no coincide con el texto")
            
        except Exception as e:
            print(f"❌ Error en Tesseract: {e}")
            texto = ""
            
        return texto, resumen
    
    def leer_desde_pil(self,
                      pil_image,
                      preprocesar: bool = True,
                      pipeline: str = 'escritura',
                      tipo_texto: str = 'linea',
                      verbose: bool = False) -> Tuple[str, str]:
        """
        Lee texto desde imagen PIL (para datasets).
        
        Args:
            pil_image: Imagen PIL
            preprocesar: Si True, aplica preprocesamiento
            pipeline: Tipo de pipeline
            tipo_texto: Tipo de texto esperado
            verbose: Si True, imprime información
            
        Returns:
            Tupla de (texto_completo, resumen_pipeline)
        """
        # Convertir PIL a OpenCV
        imagen = self.preprocesador.from_pil(pil_image)
        
        # Preprocesar si se solicita
        if preprocesar:
            imagen_procesada = self.preprocesador.preprocess(imagen, pipeline=pipeline)
            resumen = self.preprocesador.resumen_pipeline()
        else:
            if len(imagen.shape) == 3:
                imagen_procesada = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                imagen_procesada = imagen
            resumen = "Sin preprocesamiento"
        
        # Seleccionar PSM
        psm = self._seleccionar_psm(tipo_texto)
        config = f'--psm {psm}'
        
        # Leer texto con Tesseract
        try:
            texto = pytesseract.image_to_string(
                imagen_procesada, 
                lang=self.idioma,
                config=config
            )
            texto = texto.strip()
        except Exception as e:
            if verbose:
                print(f"❌ Error: {e}")
            texto = ""
        
        return texto, resumen
    
    def diagnosticar_imagen(self, imagen_path: str):
        """
        Ejecuta diagnóstico completo de una imagen.
        Prueba diferentes combinaciones de pipeline y PSM.
        """
        print("\n" + "=" * 60)
        print(f"DIAGNÓSTICO: {os.path.basename(imagen_path)}")
        print("=" * 60)
        
        pipelines = ['minimalista', 'estandar', 'libro']
        psms = ['auto', 'bloque', 'disperso']
        
        resultados = []
        
        for pipeline in pipelines:
            for psm_tipo in psms:
                try:
                    texto, _ = self.leer_texto(
                        imagen_path,
                        preprocesar=True,
                        pipeline=pipeline,
                        tipo_texto=psm_tipo,
                        verbose=False
                    )
                    
                    resultados.append({
                        'pipeline': pipeline,
                        'psm': psm_tipo,
                        'caracteres': len(texto),
                        'texto': texto[:100] if texto else ""
                    })
                    
                except Exception as e:
                    resultados.append({
                        'pipeline': pipeline,
                        'psm': psm_tipo,
                        'caracteres': 0,
                        'texto': f"ERROR: {e}"
                    })
        
        # Ordenar por número de caracteres detectados
        resultados.sort(key=lambda x: x['caracteres'], reverse=True)
        
        print("\n📊 RESULTADOS (ordenados por caracteres detectados):")
        print("-" * 60)
        
        for i, r in enumerate(resultados[:5], 1):
            print(f"{i}. Pipeline: {r['pipeline']:12} | PSM: {r['psm']:10} | "
                  f"Chars: {r['caracteres']:4}")
            if r['caracteres'] > 0:
                print(f"   Preview: {r['texto'][:80]}...")
        
        print("\n" + "=" * 60)
        
        if resultados[0]['caracteres'] > 0:
            print(f"✅ MEJOR CONFIGURACIÓN:")
            print(f"   Pipeline: {resultados[0]['pipeline']}")
            print(f"   PSM: {resultados[0]['psm']}")
            print(f"   Caracteres: {resultados[0]['caracteres']}")
        else:
            print("Ninguna configuración detectó texto")
            print("   Verifica:")
            print("   1. La imagen contiene texto legible")
            print("   2. Tesseract tiene el idioma correcto instalado")
            print("   3. La calidad de la imagen es suficiente")


#  EJEMPLO DE USO 

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print(" " * 15 + "SISTEMA OCR - TESSERACT")
    print("=" * 60)
    
    # Crear sistema OCR
    ocr = SistemaOCR(idioma='spa')
    
    print("\n💡 Ejemplos de uso:")
    print("\n1. Leer una imagen (básico):")
    print("   texto, resumen = ocr.leer_texto('imagen.jpg')")
    
    print("\n2. Con configuración específica:")
    print("   texto, _ = ocr.leer_texto(")
    print("       'imagen.jpg',")
    print("       preprocesar=True,")
    print("       pipeline='estandar',")
    print("       tipo_texto='bloque'")
    print("   )")
    
    print("\n3. Diagnóstico completo:")
    print("   ocr.diagnosticar_imagen('imagen.jpg')")
    
    print("\n4. Desde imagen PIL:")
    print("   from PIL import Image")
    print("   img = Image.open('imagen.jpg')")
    print("   texto, _ = ocr.leer_desde_pil(img, pipeline='escritura')")
    
    print("\n📋 Pipelines disponibles:")
    print("   - minimalista: Rápido, para imágenes limpias")
    print("   - estandar: Balanceado, recomendado")
    print("   - libro: Optimizado para páginas de libro (NUEVO)")
    print("   - agresivo: Para documentos degradados")
    print("   - escritura: Para texto manuscrito")
    
    print("\n📋 Tipos de texto (PSM):")
    print("   - auto: Detección automática (default)")
    print("   - bloque: Bloque uniforme de texto")
    print("   - linea: Una sola línea")
    print("   - palabra: Una sola palabra")
    print("   - disperso: Texto disperso")
    
    print("\n" + "=" * 60)
    print("✅ Para usar: importa esta clase en tu script")
    print("=" * 60)