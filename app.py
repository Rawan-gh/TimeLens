import os
import cv2
import numpy as np
from PIL import Image
import gradio as gr
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
    """Download a file from Google Drive if not already available locally."""
    path = os.path.join(MODELS_DIR, filename)
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[gdown] downloading {filename}...")
        gdown.download(url, path, quiet=False)
    return path

# ===================== Utils =====================
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV BGR numpy array."""
    return np.array(img.convert("RGB"))[:, :, ::-1].copy()

def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR numpy array back to a PIL image."""
    return Image.fromarray(arr[:, :, ::-1].astype("uint8"))

# ===================== Scratch Removal =====================
def build_crack_mask(bgr: np.ndarray, sensitivity: float) -> np.ndarray:
    """Generate a binary mask for scratches using edge detection."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, int(150 * sensitivity))
    mask = cv2.dilate(edges, (3, 3), iterations=2)
    mask = cv2.medianBlur(mask, 5)
    return mask

def remove_scratches(bgr: np.ndarray, mode: str) -> np.ndarray:
    """Remove scratches from the image depending on selected mode."""
    if mode == "No Scratches":
        return bgr
    elif mode == "Small Scratches":
        sensitivity = 0.8
    else:  # Big Scratches
        sensitivity = 0.95

    mask = build_crack_mask(bgr, sensitivity)
    scratch_ratio = np.sum(mask > 0) / mask.size

    # Ignore tiny scratch detections to avoid false positives
    if scratch_ratio < 0.001:
        return bgr

    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)

# ===================== GFPGAN Restorer =====================
_restorer = None
def get_restorer(upscale: int = 2):
    """Load GFPGAN model for face restoration."""
    global _restorer
    if _restorer is not None:
        return _restorer
    model_path = download_from_drive("GFPGANv1.3.pth", DRIVE_FILES["GFPGANv1.3.pth"])
    _restorer = GFPGANer(
        model_path=model_path, upscale=upscale,
        arch="clean", channel_multiplier=2, bg_upsampler=None
    )
    return _restorer

def restore_faces(bgr: np.ndarray) -> np.ndarray:
    """Apply GFPGAN to restore facial regions."""
    restorer = get_restorer(2)
    _, _, restored = restorer.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True)
    return restored if restored is not None else bgr

# ===================== Colorization =====================
_color_net = None
def get_colorizer():
    """Load pretrained colorization model from Zhang et al."""
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
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    _color_net = net
    return _color_net

def need_colorization(bgr: np.ndarray) -> bool:
    """Decide if the image is grayscale and requires colorization."""
    b, g, r = cv2.split(bgr.astype(np.int16))
    diff = (np.abs(b-g) + np.abs(g-r) + np.abs(b-r)).mean()
    return diff < 8

def colorize_bgr(bgr: np.ndarray) -> np.ndarray:
    """Apply automatic colorization to grayscale images."""
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

# ===================== Pipeline =====================
def enhance_pipeline(image: Image.Image, scratch_mode: str):
    """Pipeline: scratch removal -> face restoration -> optional colorization + auto decision log."""
    bgr = pil_to_bgr(image)

    # Step 1: Scratch removal
    no_scratches = remove_scratches(bgr, scratch_mode)

    # Step 2: Face restoration
    restored = restore_faces(no_scratches)

    # Step 3: Colorization (if needed)
    need_color = need_colorization(restored)
    if need_color:
        final = colorize_bgr(restored)
    else:
        final = restored

    # ===== Auto decision info =====
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    damage = float(np.mean(edges)) / 255.0

    if scratch_mode == "No Scratches":
        inpaint = False
        sens = 0.0
        r = 0
    elif scratch_mode == "Small Scratches":
        inpaint = True
        sens = 0.55
        r = 3
    else:  # Big Scratches
        inpaint = True
        sens = 0.70
        r = 5

    upscale = 1
    order = "restore_inpaint_colorize" if inpaint else "restore_colorize"

    decisions = (
        f"damage={damage:.2f}, need_color={need_color}, "
        f"order={order}, inpaint={inpaint}(sens={sens:.2f}, r={r}), upscale={upscale}"
    )

    return bgr_to_pil(bgr), bgr_to_pil(final), decisions

# ===================== UI =====================
custom_css = """
body {
    background: #d7ccc8;
}
.gradio-container {
    font-family: 'Merriweather', serif;
}
#title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #3e2723;
    margin-bottom: 12px;
}
#subtitle {
    text-align: center;
    font-size: 18px;
    color: #5d4037;
    margin-bottom: 35px;
}
.card {
    background: rgba(255, 255, 255, 0.92);
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #a1887f;
}
.gr-button {
    background: #6d4c41 !important;
    color: white !important;
    font-size: 18px !important;
    border-radius: 10px !important;
}
"""

with gr.Blocks(css=custom_css, title="TimeLens — Revive Memories") as demo:
    gr.HTML("<div id='title'>🕰️ TimeLens — Revive Memories</div>")
    gr.HTML("<div id='subtitle'>Elegant AI restoration for your cherished old photos</div>")
    
    with gr.Row():
        inp = gr.Image(type="pil", label="Input", elem_classes="card")
        out1 = gr.Image(type="pil", label="Final Result", show_download_button=True, elem_classes="card")

    with gr.Row():
        with gr.Column(elem_classes="card"):
            scratch_mode = gr.Radio(
                ["No Scratches", "Small Scratches", "Big Scratches"],
                value="Small Scratches",
                label="Scratch Type"
            )
            btn = gr.Button("Enhance ✨")
    
    out_decisions = gr.Textbox(label="Auto Decisions", interactive=False)

    btn.click(enhance_pipeline, inputs=[inp, scratch_mode], outputs=[inp, out1, out_decisions])

if __name__ == "__main__":
    demo.launch()
