import os
from dotenv import load_dotenv
from matplotlib.pyplot import pink
import numpy as np
import requests
from pathlib import Path
import fastapi
import uvicorn
from general import Pipeline
from engine import QdrantEngine, score
from pydantic import BaseModel, HttpUrl
from typing import Optional
from uuid import UUID

load_dotenv()

app = fastapi.FastAPI()

video_dir = Path(os.getenv("VIDEO_DIR"))
video_dir.mkdir(exist_ok=True)

print("Loading models...")

video_pipeline = Pipeline.from_yaml(os.getenv("VIDEO_CONFIG"))
audio_pipeline = Pipeline.from_yaml(os.getenv("AUDIO_CONFIG"))
text_pipeline = Pipeline.from_yaml(os.getenv("TEXT_CONFIG"))

audio_empty_vec = np.zeros((1, int(os.getenv("AUDIO_DIMS"))))
text_empty_vec = np.zeros((1, int(os.getenv("TEXT_DIMS"))))

qdrant_client = QdrantEngine(":memory:", int(os.getenv("VIDEO_DIMS")), int(os.getenv("AUDIO_DIMS")), int(os.getenv("TEXT_DIMS")))

print("Starting server...")

class VideoLinkRequest(BaseModel):
    link: HttpUrl

class VideoLinkResponse(BaseModel):
    is_duplicate: bool
    duplicate_for: Optional[UUID] = None


def download_file(url: str, filename: str):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


@app.post("/upload-video")
def upload_video(link: VideoLinkRequest):
    link = str(link.link) 
    name = link.split("/")[-1]
    video_path = video_dir / name
    if not video_path.exists():
        download_file(link, video_path)

    video_embedding = video_pipeline(video_path)

    try:
        audio_embedding = audio_pipeline(video_path)
        text_embedding = text_pipeline(video_path)
    except Exception as e:
        audio_embedding = audio_empty_vec
        text_embedding = text_empty_vec

    qdrant_client.add(name.split(".")[0], video_embedding, audio_embedding, text_embedding)
    return {"status": "ok"}


@app.post("/check-video-duplicate")
def check_video_duplicate(link: VideoLinkRequest):
    link = str(link.link)
    name = link.split("/")[-1]
    video_path = video_dir / name
    if not video_path.exists():
        download_file(link, video_path)
    video_embedding = video_pipeline(video_path)

    try:
        audio_embedding = audio_pipeline(video_path)
        text_embedding = text_pipeline(video_path)
        results = qdrant_client.search(image_embedding=video_embedding, audio_embedding=audio_embedding, text_embedding=text_embedding).points
    except Exception as e:
        results = qdrant_client.search(image_embedding=video_embedding).points

    if len(results) == 0:
        is_duplicate = False
        duplicate_for = None
        return VideoLinkResponse(is_duplicate=is_duplicate, duplicate_for=duplicate_for)
    
    is_duplicate = True
    duplicate_for = results[0].id
    return VideoLinkResponse(is_duplicate=is_duplicate, duplicate_for=duplicate_for)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)