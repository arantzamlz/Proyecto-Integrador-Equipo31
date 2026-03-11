import os
import cv2
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import easyocr
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import Dataset
from PIL import Image

# ==========================================
# 0. FORZAR DIRECTORIO DE TRABAJO
# ==========================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)
    print(f"📁 Directorio de trabajo establecido en: {BASE_DIR}")
except NameError:
    BASE_DIR = "/Users/danieben/Documents/AI Scripts"
    os.chdir(BASE_DIR)
    print(f"📁 Directorio de trabajo forzado a: {BASE_DIR}")

# ==========================================
# 1. CONFIGURACIÓN DE HARDWARE (M3 / MPS)
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ==========================================
# 2. FUNCIONES CORE Y MATCHING ROBUSTO
# ==========================================
def preprocess_crop_for_ocr(crop_img):
    if crop_img is None or crop_img.size == 0: return None
    crop_img = cv2.resize(crop_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    kernel_dots = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray_thick = cv2.erode(gray, kernel_dots, iterations=1)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    background = cv2.morphologyEx(gray_thick, cv2.MORPH_CLOSE, kernel_bg)
    normalized = cv2.divide(gray_thick, background, scale=255.0)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_target_pattern_strict(text_list):
    full_text = " ".join(text_list).upper()
    extracted = []
    pattern = r"C[O0]L\s*([123])[\.\,\s]*(\d{5})\s+(\d{5})"
    matches = re.findall(pattern, full_text)
    if matches:
        for match in matches:
            standardized = f"COL {match[0]}. {match[1]} {match[2]}"
            extracted.append(standardized)
    return extracted

def normalize_name(filename):
    base_name = os.path.splitext(str(filename))[0].lower()
    digits_only = "".join(re.findall(r'\d+', base_name))
    return base_name, digits_only

# ==========================================
# 3. CARGA DE GROUND TRUTH (POR POSICIÓN DE COLUMNA)
# ==========================================
ruta_ground_truth = "ground_truth.xlsx" 
gt_entries = []

if os.path.exists(ruta_ground_truth):
    df_gt = pd.read_excel(ruta_ground_truth).fillna("")
    print(f"📊 Columnas detectadas en el Excel: {df_gt.columns.tolist()}")

    for index, row in df_gt.iterrows():
        # Leemos por el índice de la columna, ignorando cómo se llamen los encabezados
        raw_name = str(row.iloc[0]) if len(row) > 0 else ""
        norm_name, digits = normalize_name(raw_name)
        
        gt_entries.append({
            "raw": raw_name,
            "norm": norm_name,
            "digits": digits,
            "COL_1": str(row.iloc[1]).strip() if len(row) > 1 else "",
            "COL_2": str(row.iloc[2]).strip() if len(row) > 2 else "",
            "COL_3": str(row.iloc[3]).strip() if len(row) > 3 else ""
        })
    print(f"✅ Ground truth cargado con {len(gt_entries)} registros.")
else:
    print("⚠️ No se encontró 'ground_truth.xlsx'.")

# ==========================================
# 4. CARGA DE MODELOS (DINO y EasyOCR)
# ==========================================
print("Cargando modelos base...")
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
reader = easyocr.Reader(['es', 'en'], gpu=True)

# ==========================================
# 5. GESTIÓN DE RUTAS (SOPORTE MULTI-FORMATO)
# ==========================================
base_extract_path = "/Users/danieben/Documents/AI Scripts/20260121 imagenes Cashcollection"
formatos_validos = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".tiff", ".tif")

image_paths = []
if os.path.exists(base_extract_path):
    for root, dirs, files in os.walk(base_extract_path):
        for file in files:
            if file.lower().endswith(formatos_validos):
                image_paths.append(os.path.join(root, file))
else:
    print(f"
          ERROR: No se encontró la carpeta de imágenes en la ruta: {base_extract_path}")
    print("Por favor, corrige la variable 'base_extract_path' en la línea 110 del código.")

print(f"Total de imágenes detectadas (múltiples formatos): {len(image_paths)}")

os.makedirs("dataset_ocr/images", exist_ok=True)
datos_entrenamiento = []
labels_doctr = {}
resultados_evaluacion = []

# ==========================================
# 6. EXTRACCIÓN Y EVALUACIÓN
# ==========================================
print("\n--- INICIANDO EXTRACCIÓN Y EVALUACIÓN ---")
for idx, path in enumerate(tqdm(image_paths, desc="Procesando imágenes")):
    img_cv = cv2.imread(path)
    if img_cv is None: 
        continue
    
    nombre_archivo = os.path.basename(path)
    file_norm, file_digits = normalize_name(nombre_archivo)
    
    matched_gt = None
    for gt in gt_entries:
        if gt['norm'] == file_norm:
            matched_gt = gt
            break
    if not matched_gt:
        for gt in gt_entries:
            if gt['norm'] in file_norm or file_norm in gt['norm']:
                matched_gt = gt
                break
    if not matched_gt and file_digits:
        for gt in gt_entries:
            if gt['digits'] == file_digits and gt['digits'] != "":
                matched_gt = gt
                break
                
    expected = matched_gt if matched_gt else {"COL_1": "", "COL_2": "", "COL_3": "", "raw": "No Encontrado"}
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    inputs = dino_processor(images=img_rgb, text="printed text. numbers.", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.25, target_sizes=[img_cv.shape[:2]]
    )[0]

    textos_extraidos_raw = []

    for box_idx, box in enumerate(results["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        y1, y2 = max(0, y1-2), min(img_rgb.shape[0], y2+2)
        x1, x2 = max(0, x1-2), min(img_rgb.shape[1], x2+2)

        recorte = img_rgb[y1:y2, x1:x2]
        if recorte.shape[0] < 5 or recorte.shape[1] < 5: 
            continue 
        
        img_prep = preprocess_crop_for_ocr(recorte)
        if img_prep is None: 
            continue

        ocr_result = reader.readtext(img_prep, detail=0)
        texto_detectado = " ".join(ocr_result).strip()

        if texto_detectado:
            textos_extraidos_raw.append(texto_detectado)
            
            filename = f"recorte_{file_digits if file_digits else idx}_{box_idx}.jpg"
            save_path = f"dataset_ocr/images/{filename}"
            Image.fromarray(recorte).save(save_path)
            datos_entrenamiento.append({"image_path": save_path, "text": texto_detectado})
            labels_doctr[filename] = texto_detectado

    codigos_formateados = extract_target_pattern_strict(textos_extraidos_raw)
    dict_extraccion = {"COL_1": "", "COL_2": "", "COL_3": ""}
    for codigo in codigos_formateados:
        if "COL 1" in codigo: dict_extraccion["COL_1"] = codigo
        elif "COL 2" in codigo: dict_extraccion["COL_2"] = codigo
        elif "COL 3" in codigo: dict_extraccion["COL_3"] = codigo

    match_c1 = (dict_extraccion["COL_1"] == expected["COL_1"]) if expected["COL_1"] else None
    match_c2 = (dict_extraccion["COL_2"] == expected["COL_2"]) if expected["COL_2"] else None
    match_c3 = (dict_extraccion["COL_3"] == expected["COL_3"]) if expected["COL_3"] else None
    
    aciertos = sum([1 for m in [match_c1, match_c2, match_c3] if m is True])
    esperados = sum([1 for v in [expected["COL_1"], expected["COL_2"], expected["COL_3"]] if v])
    
    resultados_evaluacion.append({
        "Archivo Físico": nombre_archivo,
        "Match Excel": expected["raw"],
        "COL_1 Esperado": expected["COL_1"], "COL_1 Extraído": dict_extraccion["COL_1"], "Match COL_1": match_c1,
        "COL_2 Esperado": expected["COL_2"], "COL_2 Extraído": dict_extraccion["COL_2"], "Match COL_2": match_c2,
        "COL_3 Esperado": expected["COL_3"], "COL_3 Extraído": dict_extraccion["COL_3"], "Match COL_3": match_c3,
        "Exactitud Ticket": f"{aciertos}/{esperados}" if esperados > 0 else "N/A"
    })

if len(resultados_evaluacion) > 0:
    # Exportar reporte CSV
    df_resultados = pd.DataFrame(resultados_evaluacion)
    df_resultados.to_csv("reporte_rendimiento_detallado.csv", index=False, encoding="utf-8-sig")
    print("\n✅ Reporte de métricas guardado como 'reporte_rendimiento_detallado.csv'.")

    # ==========================================
    # 7. GENERACIÓN DE GRÁFICOS DE RENDIMIENTO
    # ==========================================
    print("Generando gráficos de rendimiento pre-entrenamiento...")
    try:
        cols = ['Match COL_1', 'Match COL_2', 'Match COL_3']
        accuracies = []
        
        for col in cols:
            evaluados = df_resultados[df_resultados[col].notnull()]
            if len(evaluados) > 0:
                exitos = evaluados[col].sum()
                accuracies.append((exitos / len(evaluados)) * 100)
            else:
                accuracies.append(0)

        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")
        ax = sns.barplot(x=['COL 1', 'COL 2', 'COL 3'], y=accuracies, palette="viridis")
        plt.ylim(0, 110)
        plt.title('Precisión del Modelo Pre-Entrenamiento por Columna', fontsize=14, pad=15)
        plt.ylabel('Precisión (%)', fontsize=12)
        
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 8), 
                        textcoords='offset points', weight='bold')

        plt.savefig("grafico_rendimiento.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Gráfico generado exitosamente: 'grafico_rendimiento.png'")
    except Exception as e:
        print(f"⚠️ No se pudo generar el gráfico: {e}")
else:
    print("⚠️ No hay resultados de evaluación para generar gráficas.")

# ==========================================
# 8. FINE-TUNING DE TrOCR (50 ÉPOCAS)
# ==========================================
if not datos_entrenamiento:
    print("⚠️ No se extrajeron datos válidos para entrenar. Finalizando proceso temprano.")
    exit(0)

df_train = pd.DataFrame(datos_entrenamiento)
with open("dataset_ocr/labels.json", "w") as f:
    json.dump(labels_doctr, f)

print("\n--- INICIANDO ENTRENAMIENTO TrOCR (50 ÉPOCAS) ---")

class TrOCRDataset(Dataset):
    def __init__(self, df, processor, max_target_length=128):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.df['image_path'][idx]).convert("RGB")
        text = self.df['text'][idx]
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

processor_trocr = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model_trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
model_trocr.config.decoder_start_token_id = processor_trocr.tokenizer.cls_token_id
model_trocr.config.pad_token_id = processor_trocr.tokenizer.pad_token_id
model_trocr.config.vocab_size = model_trocr.config.decoder.vocab_size

train_dataset = TrOCRDataset(df_train, processor_trocr)

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=False,
    output_dir="./trocr_bimbo_finetuned",
    num_train_epochs=50,
    save_strategy="epoch",
    logging_steps=10,
    use_mps_device=True if torch.backends.mps.is_available() else False
)

trainer = Seq2SeqTrainer(
    model=model_trocr,
    tokenizer=processor_trocr.feature_extractor,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
print("✅ Pipeline y Fine-Tuning completados exitosamente.")