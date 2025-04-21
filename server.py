import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
import json
import cv2
import os
import pygame
import websockets
import numpy as np

pcs = set()

async def handler(websocket):
	print("Cliente Conectado")
	
	pc = RTCPeerConnection()
	pcs.add(pc)
	
	@pc.on("track")
	
	def on_track(track):
		print("Recibiendo video en tiempo real...", track.kind)
		if track.kind == "video":
			#asyncio.ensure_future(display_video(track))
			asyncio.create_task(display_video(track))
	
	data = await websocket.recv()
	offer = json.loads(data)
	await pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
	
	answer = await pc.createAnswer()
	await pc.setLocalDescription(answer)
	
	await websocket.send(json.dumps({
		"sdp": pc.localDescription.sdp,
		"type": pc.localDescription.type
		}))
	
	await asyncio.Future()
	
async def display_video(track):
	pygame.init()
	screen = None

	while True:
		try:
			print("Entre jejeje")
			frame = await track.recv()
			print("Lo recibi jejje")
			
			img = frame.to_ndarray(format="rgb24")
			
			if screen is None:
				h,w,_=img.shape
				screen = pygame.display.set_mode((w,h))
				pygame.display.set_caption("Video desde Pi Zero (pygame)")
			
			surface=pygame.surfarray.make_surface(np.rot90(img))
			screen.blit(surface, (0,0))
			pygame.display.flip()
			
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					return
			
		except Exception as e:
			print(f"Error procesando frame: {e}")
			break


async def main():
	print("Servidor WebSocket corriendo en el puerto 8080")
	async with websockets.serve(handler, "0.0.0.0", 8080):
		await asyncio.Future()
		
if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		print("Servidor Detenido")

