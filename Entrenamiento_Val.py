import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# ---------------------- CONFIGURACIÓN DE MEDIAPIPE ------------------------
mp_face_mesh = mp.solutions.face_mesh
index_angulos = [4, 291, 0, 0, 291, 17, 61, 0, 291, 291, 4, 278, 61, 4, 291, 291, 280,
                 278, 291, 426, 278, 278, 280, 374, 278, 4, 351, 336, 4, 107, 4, 351,
                 374, 351, 374, 280, 374, 359, 386, 276, 334, 336, 351, 336, 334,
                 359, 276, 334, 386, 359, 276, 374, 351, 336, 280, 291, 426, 426,
                 291, 4, 426, 278, 280]
numeros = list(range(0, 63, 3))  # Para calcular 21 ángulos

# ---------------------- FUNCIÓN DE EXTRACCIÓN DE ÁNGULOS ------------------------
def extraer_angulos(frame):
    angulos = []
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            puntos = [face_landmarks.landmark[i] for i in index_angulos]
            coords = np.array([[p.x, p.y, p.z] for p in puntos])
            for i in numeros:
                AB = coords[i + 1] - coords[i]
                BC = coords[i + 2] - coords[i + 1]
                producto_punto = np.dot(AB, BC)
                magnitud_AB = np.linalg.norm(AB)
                magnitud_BC = np.linalg.norm(BC)
                if magnitud_AB > 0 and magnitud_BC > 0:
                    cos_theta = np.clip(producto_punto / (magnitud_AB * magnitud_BC), -1.0, 1.0)
                    angulo_radianes = np.arccos(cos_theta)
                    angulo_grados = np.degrees(angulo_radianes)
                    angulos.append(angulo_grados)
    return angulos

# ---------------------- ENTRENAMIENTO Y VALIDACIÓN ------------------------
# Leer CSV
train_data = pd.read_csv('affectnet_train_angles.csv')
val_data = pd.read_csv('affectnet_val_angles.csv')

train_data = train_data.drop(columns=['Imagen', 'Valence', 'Arousal'])
val_data = val_data.drop(columns=['Imagen', 'Valence', 'Arousal'])

# Separar características y etiquetas
X_train = train_data.iloc[:, 1:22]
y_train = train_data['Emotion']
X_val = val_data.iloc[:, 1:22]
y_val = val_data['Emotion']

# Crear modelo MLP
modelo_mlp = MLPClassifier(hidden_layer_sizes=(600,), activation='relu', solver='adam',
                           max_iter=300, random_state=42)
modelo_mlp.fit(X_train, y_train)

# Validación
y_pred = modelo_mlp.predict(X_val)
print("REPORTE DE CLASIFICACIÓN EN VALIDACIÓN:")
print(classification_report(y_val, y_pred))
print("MATRIZ DE CONFUSIÓN:")
print(confusion_matrix(y_val, y_pred))

# Guardar el modelo
model_path = os.path.join(os.getcwd(), 'Modelo_MLP_entrenado.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(modelo_mlp, file)

# ---------------------- VISUALIZACIÓN EN TIEMPO REAL ------------------------
# Cargar modelo (por si quieres ejecutar solo esta parte después)
# with open('Modelo_MLP_entrenado.pkl', 'rb') as file:
#     modelo_mlp = pickle.load(file)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    caracteristicas = extraer_angulos(frame)

    if len(caracteristicas) == 21:
        try:
            caracteristicas = np.array(caracteristicas).reshape(1, -1)
            emocion_predicha = modelo_mlp.predict(caracteristicas)
            texto = f'Emocion: {emocion_predicha[0]}'
            color = (0, 255, 0)
        except Exception as e:
            print(f"Error en la predicción: {e}")
            texto = "Error en la predicción"
            color = (0, 0, 255)
    else:
        texto = "No se pudieron extraer ángulos"
        color = (0, 0, 255)

    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Detección de Emoción', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
