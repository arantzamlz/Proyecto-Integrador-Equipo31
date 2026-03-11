import os
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['FLAGS_onednn_kernel_use_datetime_cache'] = '0'

import re
import random
import pandas as pd
from PIL import Image
from paddleocr import PaddleOCR

# ── configuración ──────────────────────────────────────────────
IMAGE_FOLDER = r'C:\Users\faby9\Downloads\20260121 imagenes Cashcollection1\20260121 imagenes Cashcollection'
EXCEL_PATH   = r'C:\Users\faby9\Downloads\codigostickets2.xlsx'
OUTPUT_PATH  = r'C:\Users\faby9\Downloads\resultados_ocr.xlsx'
NUM_IMAGENES = 850
# ──────────────────────────────────────────────────────────────

ocr = PaddleOCR(use_textline_orientation=True, lang='es', enable_mkldnn=False)

df = pd.read_excel(EXCEL_PATH)
df.columns = df.columns.str.strip()
df['archivo'] = df['archivo'].str.strip()

imagenes_disponibles = [f for f in os.listdir(IMAGE_FOLDER)
                        if f in df['archivo'].values]
print(f"Imágenes disponibles en Excel: {len(imagenes_disponibles)}")
muestra = random.sample(imagenes_disponibles, min(NUM_IMAGENES, len(imagenes_disponibles)))

def convertir_jfif(ruta):
    if ruta.lower().endswith('.jfif'):
        nueva_ruta = ruta.rsplit('.', 1)[0] + '_temp.jpg'
        img = Image.open(ruta)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(nueva_ruta, 'JPEG')
        return nueva_ruta, True
    return ruta, False

def extraer_texto_predict(ocr_resultado):
    lineas = []
    try:
        for pagina in ocr_resultado:
            if pagina:
                for linea in pagina:
                    if linea and len(linea) >= 2:
                        lineas.append(linea[1][0])
    except Exception as e:
        print(f"    Debug extracción: {e}")
    return ' '.join(lineas)

def extraer_columnas(texto_ocr):
    resultado = {'COL_1': '', 'COL_2': '', 'COL_3': ''}
    patrones = {
        'COL_1': r'C[O0]L\s*1[\.\s]*(\d{5}\s*\d{5})',
        'COL_2': r'C[O0]L\s*2[\.\s]*(\d{5}\s*\d{5})',
        'COL_3': r'C[O0]L\s*3[\.\s]*(\d{5}\s*\d{5})',
    }
    for col, patron in patrones.items():
        match = re.search(patron, texto_ocr, re.IGNORECASE)
        if match:
            resultado[col] = re.sub(r'\s+', '', match.group(1))
    return resultado

def limpiar_gt(texto):
    numeros = re.sub(r'[^0-9]', '', str(texto))
    return numeros[-10:]

def cer(referencia, hipotesis):
    ref = list(referencia.replace(' ', ''))
    hip = list(hipotesis.replace(' ', ''))
    if not ref:
        return 0.0
    d = [[0] * (len(hip) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hip) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hip) + 1):
            cost = 0 if ref[i-1] == hip[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref)][len(hip)] / len(ref)

def wer(referencia, hipotesis):
    ref = referencia.split()
    hip = hipotesis.split()
    if not ref:
        return 0.0
    d = [[0] * (len(hip) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hip) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hip) + 1):
            cost = 0 if ref[i-1] == hip[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref)][len(hip)] / len(ref)

# ── procesar imágenes ──────────────────────────────────────────
resultados = []

for i, archivo in enumerate(muestra):
    print(f"[{i+1}/{len(muestra)}] Procesando: {archivo}")
    ruta_original = os.path.join(IMAGE_FOLDER, archivo)
    ruta, es_temp = convertir_jfif(ruta_original)

    try:
        ocr_resultado = ocr.ocr(ruta, cls=False)
        texto_completo = extraer_texto_predict(ocr_resultado)

        detectado = extraer_columnas(texto_completo)

        fila = df[df['archivo'] == archivo].iloc[0]
        gt = {
            'COL_1': limpiar_gt(fila['COL_1']),
            'COL_2': limpiar_gt(fila['COL_2']),
            'COL_3': limpiar_gt(fila['COL_3']),
        }

        for col in ['COL_1', 'COL_2', 'COL_3']:
            resultados.append({
                'archivo': archivo,
                'columna': col,
                'ground_truth': gt[col],
                'detectado': detectado[col],
                'CER': round(cer(gt[col], detectado[col]), 4),
                'WER': round(wer(gt[col], detectado[col]), 4),
            })

    except Exception as e:
        print(f"  ⚠️ Error en {archivo}: {e}")

    finally:
        if es_temp and os.path.exists(ruta):
            os.remove(ruta)

# ── resultados ─────────────────────────────────────────────────
df_resultados = pd.DataFrame(resultados)

print("\n===== RESULTADOS FINALES =====")
for col in ['COL_1', 'COL_2', 'COL_3']:
    sub = df_resultados[df_resultados['columna'] == col]
    print(f"{col} → CER promedio: {sub['CER'].mean():.4f} | WER promedio: {sub['WER'].mean():.4f}")

print(f"\nCER global: {df_resultados['CER'].mean():.4f}")
print(f"WER global: {df_resultados['WER'].mean():.4f}")

df_resultados.to_excel(OUTPUT_PATH, index=False)
print(f"\n✅ Resultados guardados en: {OUTPUT_PATH}")