import argparse
from utils import preprocess_image
from ocr_pipeline import BookOCR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta de la imagen")
    args = parser.parse_args()

    print(f"--- Procesando con EasyOCR: {args.input} ---")
    
    # 1. Preprocesar
    img_ready = preprocess_image(args.input)
    
    if img_ready is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # 2. OCR
    engine = BookOCR(lang='es')
    texto_final = engine.extract_text(img_ready)

    # 3. Guardar
    with open("resultado.txt", "w", encoding="utf-8") as f:
        f.write(texto_final)
    
    print("¡Listo! Revisa resultado.txt")

if __name__ == "__main__":
    main()