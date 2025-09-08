import os
import numpy as np
from PIL import Image
import gradio as gr
import cv2
import gdown
from gfpgan import GFPGANer

# ===================== Paths & Weights =====================
MODELS_DIR = "models"
DRIVE_FILES = {
    "GFPGANv1.3.pth": "1Dp0tVXIsjiVaG3pHCfLvoOQaEruTbL2b",
    "colorization_deploy_v2.prototxt": "1zCT7qsLjckfdvvFUoNbGc0sT28CXIOo2",
    "colorization_release_v2.caffemodel": "1MjholzNWvfLQK1kA_QT9pFhm5ktik4pM",
    "pts_in_hull.npy": "1ovn7oSLprM4oqbSbIoAFs8x_YTuHnPgN",
}

def download_from_drive(filename: str, file_id: str):
    path = os.path.join(MODELS_DIR, filename)
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    return path

# ===================== Utils =====================
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))[:, :, ::-1].copy()

def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr[:, :, ::-1].astype("uint8"))

# ===================== Colorization =====================
_color_net = None
def get_colorizer():
    global _color_net
    if _color_net is not None:
        return _color_net
    proto = download_from_drive("colorization_deploy_v2.prototxt", DRIVE_FILES["colorization_deploy_v2.prototxt"])
    model = download_from_drive("colorization_release_v2.caffemodel", DRIVE_FILES["colorization_release_v2.caffemodel"])
    pts_path = download_from_drive("pts_in_hull.npy", DRIVE_FILES["pts_in_hull.npy"])
    net = cv2.dnn.readNetFromCaffe(proto, model)
    pts = np.load(pts_path)
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]
    _color_net = net
    return _color_net

def colorize_bgr(bgr: np.ndarray) -> np.ndarray:
    net = get_colorizer()
    H, W = bgr.shape[:2]
    img_float = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]
    L_rs = cv2.resize(L, (224, 224))
    blob = cv2.dnn.blobFromImage(L_rs - 50.0)
    net.setInput(blob)
    ab = net.forward()[0].transpose((1, 2, 0))
    ab_us = cv2.resize(ab, (W, H))
    lab_out = np.concatenate((L[..., None], ab_us), axis=2).astype(np.float32)
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
    return np.clip(bgr_out * 255.0, 0, 255).astype(np.uint8)

def need_colorization(bgr: np.ndarray) -> bool:
    b, g, r = cv2.split(bgr.astype(np.int16))
    diff = (np.abs(b-g) + np.abs(g-r) + np.abs(b-r)).mean()
    return diff < 8

# ===================== Scratch Fix =====================
def build_crack_mask(bgr: np.ndarray, sensitivity: float) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, int(100 * sensitivity))
    mask = cv2.dilate(edges, (3, 3), iterations=1)
    mask = cv2.medianBlur(mask, 3)
    return mask

def estimate_damage(mask: np.ndarray) -> float:
    return float(np.sum(mask > 0) / mask.size)

def remove_scratches_small(bgr: np.ndarray):
    mask = build_crack_mask(bgr, 0.6)
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA), estimate_damage(mask), 0.6, 3

def remove_scratches_big(bgr: np.ndarray):
    mask = build_crack_mask(bgr, 0.98)
    return cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA), estimate_damage(mask), 0.98, 5

# ===================== GFPGAN Restorer =====================
_restorer = None
_restorer_upscale = None
def get_restorer(upscale: int):
    global _restorer, _restorer_upscale
    if _restorer is not None and _restorer_upscale == upscale:
        return _restorer
    model_path = download_from_drive("GFPGANv1.3.pth", DRIVE_FILES["GFPGANv1.3.pth"])
    _restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    _restorer_upscale = upscale
    return _restorer

def restore_faces(bgr_in: np.ndarray, upscale: int) -> np.ndarray:
    restorer = get_restorer(upscale)
    _, _, restored = restorer.enhance(
        bgr_in, has_aligned=False, only_center_face=False, paste_back=True
    )
    return restored if restored is not None else bgr_in

# ===================== Final Pipeline =====================
def enhance_pipeline(image: Image.Image, scratch_mode: str):
    bgr = pil_to_bgr(image)

    damage, sens, r, inpaint = 0.0, 0.0, 0, False

    # Scratch removal
    if scratch_mode == "Small Scratches":
        bgr, damage, sens, r = remove_scratches_small(bgr)
        inpaint = True
    elif scratch_mode == "Big Scratches":
        bgr, damage, sens, r = remove_scratches_big(bgr)
        inpaint = True

    # Always: Colorize → Restore
    colored = need_colorization(bgr)
    upscale = 2

    if colored:
        bgr = colorize_bgr(bgr)
    bgr = restore_faces(bgr, upscale)

    info = (f"Auto decisions → damage={damage:.2f}, need_color={colored}, "
            f"order=Colorize_before_Restore, "
            f"inpaint={inpaint}(sens={sens:.2f}, r={r}), "
            f"upscale={upscale}, scratch_mode={scratch_mode}")

    return bgr_to_pil(bgr), info

# ===================== Gradio UI =====================
custom_css = """
body { background: #d7ccc8; }
.gradio-container { font-family: 'Merriweather', serif; }
#title { text-align: center; font-size: 40px; font-weight: bold; color: #3e2723; margin-bottom: 8px; }
#subtitle { text-align: center; font-size: 18px; color: #5d4037; margin-bottom: 25px; }
"""

with gr.Blocks(css=custom_css, title="TimeLens — Revive Memories") as demo:
    gr.HTML("<div id='title'>🕰️ TimeLens — Revive Memories</div>")
    gr.HTML("<div id='subtitle'>Elegant AI restoration for your cherished old photos</div>")

    with gr.Row():
        inp = gr.Image(type="pil", label="Input")
        out = gr.Image(type="pil", label="Final Result", show_download_button=True)

    with gr.Row():
        scratch_mode = gr.Radio(["No Scratches", "Small Scratches", "Big Scratches"],
                                value="No Scratches", label="Scratch Type")

    btn = gr.Button("Enhance ✨")
    log = gr.Textbox(label="Auto Decisions", interactive=False)

    btn.click(enhance_pipeline, inputs=[inp, scratch_mode], outputs=[out, log])

if __name__ == "__main__":
    demo.launch()
