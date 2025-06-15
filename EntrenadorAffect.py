import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# 🔧 Elegir el set a procesar:
# ====> OPCIÓN 1: TRAIN SET
image_folder = "train_set/images"
annotations_folder = "train_set/annotations"
output_csv = "affectnet_train_angles.csv"

# ====> OPCIÓN 2: VALIDATION SET
#image_folder = "val_set/images"
#annotations_folder = "val_set/annotations"
#output_csv = "affectnet_val_angles.csv"

# ===> Diccionario para traducir clase → emoción
emotion_names = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt",
    8: "None",
    9: "Uncertain",
    10: "Non-Face"
}

# ===> Índices para cálculo de ángulos con MediaPipe
index_angulos = [4,291,0,0,291,17,61,0,291,291,4,278,61,4,291,291,280,
                 278,291,426,278,278,280,374,278,4,351,336,4,107,4,351,
                 374,351,374,280,374,359,386,276,334,336,351,336,334,
                 359,276,334,386,359,276,374,351,336,280,291,426,426,
                 291,4,426,278,280]

# Normalización de ángulos
def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return [(v - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0 for v in values]

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
sin_malla = 0
procesadas = 0

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Emotion", "Imagen", "Valence", "Arousal"] + [f"Angulo{i}" for i in range(len(index_angulos)//3)])

        for img_file in os.listdir(image_folder):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            try:
                base_name = os.path.splitext(img_file)[0]
                img_index = int(base_name)
                img_path = os.path.join(image_folder, img_file)

                # Cargar etiquetas
                clase = int(np.load(os.path.join(annotations_folder, f"{img_index}_exp.npy")))
                valence = float(np.load(os.path.join(annotations_folder, f"{img_index}_val.npy")))
                arousal = float(np.load(os.path.join(annotations_folder, f"{img_index}_aro.npy")))
                emotion_name = emotion_names.get(clase, "Unknown")
            except Exception as e:
                print(f"❌ Error al cargar anotaciones para {img_file}: {e}")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Imagen no cargada: {img_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(image_rgb)

            if result.multi_face_landmarks is None:
                sin_malla += 1
                continue

            # Extraer puntos para ángulos
            puntos_3d = []
            for idx in index_angulos:
                lm = result.multi_face_landmarks[0].landmark[idx]
                puntos_3d.append(np.array([lm.x, lm.y, lm.z]))

            # Calcular ángulos
            angles = []
            for i in range(0, len(puntos_3d) - 2, 3):
                A = puntos_3d[i]
                B = puntos_3d[i + 1]
                C = puntos_3d[i + 2]
                AB = B - A
                BC = C - B
                cos_theta = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                angles.append(angle)

            angles_norm = normalize(angles)
            writer.writerow([emotion_name, img_file, valence, arousal] + angles_norm)
            procesadas += 1

print("✅ Procesamiento COMPLETO.")
print(f"🧠 Imágenes procesadas con éxito: {procesadas}")
print(f"❌ Imágenes sin malla facial detectada: {sin_malla}")