import asyncio
import json
import websockets

class WebSocketSignaling:
	def __init__(self,uri):
		self._uri = uri
		self._ws = None
		
	async def connect(self):
		self._ws = await websockets.connect(self._uri)
		
	async def send(self,message):
		await self._ws.send(json.dumps(message))
		
	async def receive(self):
		data = await self._ws.recv()
		return json.loads(data)
		
	async def close(self):
		if self._ws:
			await self._ws.close()
