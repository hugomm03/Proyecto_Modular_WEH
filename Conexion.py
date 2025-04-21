import cv2
import socket
import struct
import pickle
from ultralytics import YOLO

# Cargar el modelo YOLOv8
modelo = YOLO("yolov8n.pt")  # Puedes usar "yolov8n.pt" 

HOST = "0.0.0.0"
PORT = 5000
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server_socket.bind((HOST,PORT))
server_socket.listen(1)

print("Esperando conexión...")
conn,addr = server_socket.accept()
print(f"Conectando a {addr}")

data = b""
payload_size = struct.calcsize(">L")

while True:
	while len(data) < payload_size:
		data += conn.recv(4096)
	packed_msg_size = data[:payload_size]
	data = data[payload_size:]
	msg_size = struct.unpack(">L",packed_msg_size)[0]
	
	while len(data) < msg_size:
		data += conn.recv(4096)
	
	frame_data = data[:msg_size]
	data = data[msg_size:]
	
	frame = pickle.loads(frame_data)
	frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
	
	#resultado = modelo(frame)
	#frame_anotado = resultado[0].plot()  # YOLO nos va a devolver una lista, pero por frame tomamos el primero 
	cv2.imshow("YOLOv8 Live Detection", frame)
	
	#cv2.imshow("Streaming", frame)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
    
conn.close()
server_socket.close()
cv2.destroyAllWindows()
