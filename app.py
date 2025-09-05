import os
import numpy as np
from PIL import Image
import gradio as gr
import cv2
import gdown

from gfpgan import GFPGANer

# ===================== Paths & Weights =====================
MODELS_DIR = "models"

# Google Drive file IDs for models
DRIVE_FILES = {
    "GFPGANv1.3.pth": "1Dp0tVXIsjiVaG3pHCfLvoOQaEruTbL2b",
    "colorization_deploy_v2.prototxt": "1zCT7qsLjckfdvvFUoNbGc0sT28CXIOo2",
    "colorization_release_v2.caffemodel": "1MjholzNWvfLQK1kA_QT9pFhm5ktik4pM",
    "pts_in_hull.npy": "1ovn7oSLprM4oqbSbIoAFs8x_YTuHnPgN",
    "haarcascade_frontalface_default.xml": "1G2YSvmUIi308YKXDfMB2EbUQOxZTLjiJ",
}

def download_from_drive(filename: str, file_id: str):
    """Download a file from Google Drive if not exists."""
    path = os.path.join(MODELS_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[gdown] downloading {filename}...")
        gdown.download(url, path, quiet=False)
    return path

# ===================== Utilities =====================
def ensure_gfpgan():
    return download_from_drive("GFPGANv1.3.pth", DRIVE_FILES["GFPGANv1.3.pth"])

def ensure_color_models():
    proto = download_from_drive("colorization_deploy_v2.prototxt", DRIVE_FILES["colorization_deploy_v2.prototxt"])
    model = download_from_drive("colorization_release_v2.caffemodel", DRIVE_FILES["colorization_release_v2.caffemodel"])
    pts   = download_from_drive("pts_in_hull.npy", DRIVE_FILES["pts_in_hull.npy"])
    return proto, model, pts

def ensure_haar():
    return download_from_drive("haarcascade_frontalface_default.xml", DRIVE_FILES["haarcascade_frontalface_default.xml"])

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))[:, :, ::-1].copy()

def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr[:, :, ::-1].astype("uint8"))

# ===================== Auto decisions =====================
def robust_need_color(bgr: np.ndarray) -> bool:
    h, w = bgr.shape[:2]
    cut = int(0.08 * h)
    roi = bgr[:h - cut, :, :] if cut > 0 else bgr
    roi_u8 = roi if roi.dtype == np.uint8 else np.clip(roi, 0, 255).astype(np.uint8)

    b, g, r = cv2.split(roi_u8.astype(np.int16))
    diff = (np.abs(b - g) + np.abs(g - r) + np.abs(b - r)) / 3.0
    if np.mean(diff) < 6:
        return True

    hsv = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2HSV)
    s_med = float(np.median(hsv[:, :, 1]))
    if s_med < 18:
        return True

    lab = cv2.cvtColor(roi_u8, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    bch = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + bch * bch)
    if np.mean(chroma) < 4.5:
        return True

    return False

def estimate_damage(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = 640 / max(h, w)
    gray_s = cv2.resize(gray, (int(w*scale), int(h*scale))) if scale < 1.0 else gray
    k = max(5, int(0.01 * min(gray_s.shape)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    tophat = cv2.morphologyEx(gray_s, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray_s, cv2.MORPH_BLACKHAT, kernel)
    th1 = np.mean(tophat) + np.std(tophat)
    th2 = np.mean(blackhat) + np.std(blackhat)
    mask = ((tophat > th1) | (blackhat > th2)).astype(np.uint8)
    return float(mask.mean())

def auto_params(bgr: np.ndarray):
    dmg = estimate_damage(bgr)
    severe = dmg > 0.12
    moderate = dmg > 0.05
    sens = 0.55 if moderate else 0.45
    if severe:
        sens = 0.70
    short_side = min(bgr.shape[:2])
    radius = int(np.clip(short_side * (0.002 if not severe else 0.0035), 2, 6))
    order = "restore_inpaint_colorize" if severe else "colorize_restore_inpaint"
    use_inpaint = moderate or severe
    return order, sens, radius, use_inpaint, dmg

# ===================== Colorization =====================
_color_net = None
def get_colorizer():
    global _color_net
    if _color_net is not None:
        return _color_net
    proto, model, pts_path = ensure_color_models()
    net = cv2.dnn.readNetFromCaffe(proto, model)
    pts = np.load(pts_path)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    _color_net = net
    return _color_net

def colorize_bgr(bgr: np.ndarray) -> np.ndarray:
    net = get_colorizer()
    H, W = bgr.shape[:2]
    img_float = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]
    L_rs = cv2.resize(L, (224, 224), interpolation=cv2.INTER_CUBIC)
    blob = cv2.dnn.blobFromImage(L_rs - 50.0)
    net.setInput(blob)
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_us = cv2.resize(ab, (W, H), interpolation=cv2.INTER_CUBIC)
    lab_out = np.concatenate((L[..., np.newaxis], ab_us), axis=2).astype(np.float32)
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
    bgr_out = np.clip(bgr_out, 0, 1)
    return (bgr_out * 255.0).astype(np.uint8)

# ===================== Face-safe Inpainting =====================
def detect_faces(bgr: np.ndarray):
    xml = ensure_haar()
    face_cascade = cv2.CascadeClassifier(xml)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    return faces

def build_crack_mask(bgr: np.ndarray, sensitivity: float) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = int(5 + sensitivity * 15)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    th = int(15 + sensitivity * 100)
    mask = ((tophat > th) | (blackhat > th)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)
    mask = cv2.dilate(mask, k2, iterations=1)
    return mask

def inpaint_background_only(bgr: np.ndarray, sensitivity: float, radius: int) -> np.ndarray:
    mask = build_crack_mask(bgr, sensitivity)
    for (x, y, w, h) in detect_faces(bgr):
        pad = max(10, int(0.1 * w))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
        mask[y0:y1, x0:x1] = 0
    return cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)

# ===================== GFPGAN Restorer =====================
_restorer = None
_restorer_upscale = None
def get_restorer(upscale: int):
    global _restorer, _restorer_upscale
    if _restorer is not None and _restorer_upscale == upscale:
        return _restorer
    model_path = ensure_gfpgan()
    _restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    _restorer_upscale = upscale
    return _restorer

def auto_upscale(bgr: np.ndarray) -> int:
    short = min(bgr.shape[:2])
    if short < 480:
        return 3
    if short < 720:
        return 2
    return 1

def restore_faces(bgr_in: np.ndarray, upscale: int) -> np.ndarray:
    restorer = get_restorer(upscale)
    _, _, restored = restorer.enhance(
        bgr_in, has_aligned=False, only_center_face=False, paste_back=True
    )
    return restored if restored is not None else bgr_in

# ===================== End-to-end Auto Pipeline =====================
def enhance_auto(image: Image.Image):
    bgr = pil_to_bgr(image)
    order, sens, radius, use_inpaint, dmg = auto_params(bgr)
    need_color = robust_need_color(bgr)
    up = auto_upscale(bgr)

    try:
        if order == "restore_inpaint_colorize":
            bgr = restore_faces(bgr, up)
            if use_inpaint:
                bgr = inpaint_background_only(bgr, sens, radius)
            if need_color:
                bgr = colorize_bgr(bgr)
        else:
            if need_color:
                bgr = colorize_bgr(bgr)
            bgr = restore_faces(bgr, up)
            if use_inpaint:
                bgr = inpaint_background_only(bgr, sens, radius)
    except Exception as e:
        print(f"[pipeline] fallback (restore only): {e}")
        bgr = restore_faces(bgr, up)

    info = (
        f"Auto decisions → damage={dmg:.2f}, need_color={need_color}, "
        f"order={order}, inpaint={use_inpaint}(sens={sens:.2f}, r={radius}), upscale={up}"
    )
    return bgr_to_pil(bgr), info

# ===================== Gradio UI =====================
with gr.Blocks(title="TimeLens — Auto Enhance") as demo:
    gr.Markdown("## TimeLens — Auto Enhance\nUpload an image and click **Enhance**. Everything else is automatic.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input")
            btn = gr.Button("Enhance")
            log = gr.Textbox(label="What the app decided (auto)", interactive=False)
        with gr.Column():
            out = gr.Image(type="pil", label="Output", show_download_button=True)

    btn.click(enhance_auto, inputs=[inp], outputs=[out, log])

if __name__ == "__main__":
    demo.launch()
