import argparse
import os
from pathlib import Path
from ocr_pipeline import OCRPipeline


def main():
    # 1. Configuración de argumentos de línea de comandos
    # Por qué: Permite que el usuario interactúe con el programa sin tocar el código.
    parser = argparse.ArgumentParser(
        description="Script de inferencia para el sistema OCR"
    )
    parser.add_argument("--imagen", type=str, help="Ruta de la imagen a procesar")
    parser.add_argument(
        "--carpeta", type=str, help="Ruta de la carpeta con múltiples imágenes"
    )

    args = parser.parse_args()

    # 2. Inicializar el Pipeline
    # Para qué: Centralizamos la lógica de EasyOCR en una sola instancia.
    pipeline = OCRPipeline()

    # 3. Lógica de procesamiento
    if args.imagen:
        procesar_archivo(args.imagen, pipeline)
    elif args.carpeta:
        # Procesar todos los archivos .jpg, .png o .jpeg en la carpeta
        files = [
            f
            for f in os.listdir(args.carpeta)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            print(f"⚠️ No se encontraron imágenes en: {args.carpeta}")
            return

        for file in files:
            ruta_completa = os.path.join(args.carpeta, file)
            procesar_archivo(ruta_completa, pipeline)
    else:
        print(
            "❌ Error: Debes proporcionar --imagen o --carpeta. Usa --help para más info."
        )


def procesar_archivo(path, pipeline):
    """Ejecuta el OCR y guarda el resultado."""
    print(f"--- Procesando: {os.path.basename(path)} ---")
    try:
        texto = pipeline.extract_text(path)

        # Mostrar en consola según pide el taller
        print(texto)

        # Guardar en .txt (Plus de profesionalismo)
        output_name = f"{Path(path).stem}_resultado.txt"
        with open(output_name, "w", encoding="utf-8") as f:
            f.write(texto)
        print(f"✅ Guardado en: {output_name}\n")

    except Exception as e:
        print(f"❌ Error al procesar {path}: {e}")


if __name__ == "__main__":
    main()
