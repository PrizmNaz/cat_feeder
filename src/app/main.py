import time
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, Response

from .camera import CameraStream, get_device_from_env


@asynccontextmanager
async def lifespan(app:FastAPI):
    camera = CameraStream(device=get_device_from_env())
    camera.start()

    app.state.camera = camera
    try:
        yield
    finally:
        camera.stop()


app = FastAPI(title="Cat Vision =^.^=", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    camera:CameraStream = app.state.camera  
    return {
        "device": camera.device,
        "frames_ok": camera.stats.frames_ok,
        "frames_drop": camera.stats.frames_drop,
        "uptime_s": round(camera.stats.uptime_s(),3),
        "fps_est": round(camera.stats.fps_estimate(), 2),
        "last_frame_age_s": None if camera.stats.last_frame_ts is None else round(time.time()-camera.stats.last_frame_ts, 3)
        }

@app.get("/frame.jpg")
def frame_jpg():
    camera:CameraStream = app.state.camera
    frame = camera.get_last_frame()
    if frame is None:
        return Response(status_code=503, content=b"No frame yet")
    
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return Response(status_code=500, content=b"Jpg encode failed")
    
    return Response(status_code=200, content=jpg.tobytes(), media_type="image/jpeg")