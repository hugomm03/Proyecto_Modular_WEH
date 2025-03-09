from ultralytics import YOLO
import cv2

# Cargar el modelo YOLOv8
modelo = YOLO("yolov8n.pt")  # Puedes usar "yolov8n.pt" 

print(modelo.names)
'''
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

#Ancho es el primero, alto el segundo (tamaños)
cap.set(3, 640)
cap.set(4, 480)  

while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        break 

    # Realizar detección con YOLO
    resultado = modelo(frame)

    # Dibujar los resultados en el frame
    frame_anotado = resultado[0].plot()  # YOLO nos va a devolver una lista, pero por frame tomamos el primero 

    # Mostrar la imagen con las detecciones
    cv2.imshow("YOLOv8 Live Detection", frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''