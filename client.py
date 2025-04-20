import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer
from signaling import WebSocketSignaling
from picamera2 import Picamera2
import av
import numpy as np

SIGNALING_SERVER = "ws://10.42.0.1:8765"


class PiCameraTrack(VideoStreamTrack):
	def __init__(self):
		super().__init__()
		self.picam2 = Picamera2()

		self.picam2.configure(self.picam2.create_video_configuration(main={"size": (480,480)}))
		self.picam2.start()
		
	async def recv(self):
		frame = self.picam2.capture_array()
		print("Enviando Frame")
		video_frame = VideoFrame.from_ndarray(frame,format ="rgb24")
		video_frame.pts, video_frame.time_base = await self.next_timestamp()
		return video_frame

async def run():
	signaling = WebSocketSignaling(SIGNALING_SERVER)
	await signaling.connect()
	
	pc = RTCPeerConnection()
	
	#player = MediaPlayer("/dev/video0", format="v4l2", options={"video_size": "640x480"})
	#pc.addTrack(player.video)
	camera_track = PiCameraTrack()
	pc.addTrack(camera_track)
	
	offer= await pc.createOffer()
	await pc.setLocalDescription(offer)
	
	await signaling.send({
		"sdp": pc.localDescription.sdp,
		"type": pc.localDescription.type
		})
	
	response = await signaling.receive()
	await pc.setRemoteDescription(RTCSessionDescription(sdp=response["sdp"],type=response["type"]))
	
	print("Transmitiendo video en tiempo real desde Pi Zero...")
	await asyncio.Future()
	
asyncio.run(run())
	
