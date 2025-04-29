import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
import json
import cv2
from aiohttp import web

pcs = set()
'''
async def index(request):
	return web.Response(text="Servidor WebRTC en ejecución", content_type="text/html")
'''
async def offer(request):
	params = await request.json()
	offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
	
	pc = RTCPeerConnection()
	pcs.add(pc)
	
	@pc.on("track")
	
	def on_track(track):
		print("Recibiendo video en tiempo real...")
		if track.kind == "video":
			asyncio.create_task(display_stream(track))
	
	await pc.setRemoteDescription(offer)
	answer = await pc.createAnswer()
	await pc.setLocalDescription(answer)
	
	return web.json_response({"sdp":pc.localDescription.sdp, "type":pc.localDescription.type})


async def display_stream(track):

	while True:
		frame=await track.recv()
		img=frame.to_ndarray(format="bgr24")
		cv2.imshow("Video en tiempo real desde Pi Zero", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()
	

app = web.Application()
app.router.add_post("/offer",offer)
#app.router.add_get("/", index)
web.run_app(app,port=5000)
'''
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
	
	resultado = modelo(frame)
	frame_anotado = resultado[0].plot()  # YOLO nos va a devolver una lista, pero por frame tomamos el primero 
	#cv2.imshow("YOLOv8 Live Detection", frame_anotado)
	
	#cv2.imshow("Streaming", frame)
	
	#if cv2.waitKey(1) & 0xFF == ord('q'):
		#break
    
conn.close()
server_socket.close()
cv2.destroyAllWindows()
'''
