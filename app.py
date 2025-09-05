import os
import urllib.request
import numpy as np
from PIL import Image
import gradio as gr
import cv2
import gdown

from gfpgan import GFPGANer

# ===================== Paths & Models =====================
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Google Drive file IDs
models_to_download = {
    "colorization_deploy_v2.prototxt": "1zCT7qsLjckfdvvFUoNbGc0sT28CXIOo2",
    "colorization_release_v2.caffemodel": "1MjholzNWvfLQK1kA_QT9pFhm5ktik4pM",
    "GFPGANv1.3.pth": "1Dp0tVXIsjiVaG3pHCfLvoOQaEruTbL2b",
    "haarcascade_frontalface_default.xml": "1G2YSvmUIi308YKXDfMB2EbUQOxZTLjiJ",
    "pts_in_hull.npy": "1ovn7oSLprM4oqbSbIoAFs8x_YTuHnPgN",
}

# Download missing models
for fname, file_id in models_to_download.items():
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        print(f"⬇ Downloading {fname} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
        print(f"✅ Saved to {path}")

# ===================== Default URLs (fallback) =====================
GFPGAN_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

COLOR_PROTO_URLS = [
    "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/colorization_deploy_v2.prototxt",
]
COLOR_MODEL_URLS = [
    "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
    "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel",
]
COLOR_PTS_URLS = [
    "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
    "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy",
]

HAAR_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
]

# ===================== Utilities =====================
def try_download(path: str, urls: list[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path
    last_err = None
    for url in urls:
        try:
            print(f"[download] {url} -> {path}")
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as e:
            last_err = e
            print(f"[warn] failed: {e}")
    if last_err:
        raise last_err
    raise RuntimeError(f"Could not download {path}")

def ensure_gfpgan():
    path = os.path.join(MODELS_DIR, "GFPGANv1.3.pth")
    if not os.path.exists(path):
        print(f"[download] {GFPGAN_MODEL_URL} -> {path}")
        urllib.request.urlretrieve(GFPGAN_MODEL_URL, path)
    return path

def ensure_color_models():
    proto = os.path.join(MODELS_DIR, "colorization_deploy_v2.prototxt")
    model = os.path.join(MODELS_DIR, "colorization_release_v2.caffemodel")
    pts   = os.path.join(MODELS_DIR, "pts_in_hull.npy")

    if not os.path.exists(proto):
        proto = try_download(proto, COLOR_PROTO_URLS)
    if not os.path.exists(model):
        model = try_download(model, COLOR_MODEL_URLS)
    if not os.path.exists(pts):
        pts = try_download(pts, COLOR_PTS_URLS)

    return proto, model, pts

def ensure_haar():
    xml = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
    if not os.path.exists(xml):
        xml = try_download(xml, HAAR_URLS)
    return xml
