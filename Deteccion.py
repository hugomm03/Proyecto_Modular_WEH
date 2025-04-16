from ultralytics import YOLO
import cv2
import os

# Obtener la ruta del script actual
act_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo entrenado
modelo_path = os.path.join(act_dir, "runs/detect/train/weights/best.pt") # NOTA: Editar el path según donde se almacene el archivo "best.pt"
modelo = YOLO(modelo_path)  # Usar el modelo entrenado

# Iniciar la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Verificar si la cámara se abre correctamente
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo recibir el frame de la cámara.")
        break

    # Realizar la predicción en cada frame
    resultados = modelo(frame)

    # Dibujar los resultados sobre el frame (aunque no haya detección, el frame se mostrará)
    frame_anotado = resultados[0].plot() if len(resultados) > 0 else frame

    # Mostrar la imagen anotada, aunque no haya detecciones
    cv2.imshow("Predicción en vivo YOLOv8", frame_anotado)

    # Salir si presionas 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar la ventana
cap.release()
cv2.destroyAllWindows()




