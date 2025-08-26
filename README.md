# TimeLens — Simple (GFPGAN)

Super-simple GFPGAN app for macOS (works on CPU).

## Setup (Conda recommended)
```bash
conda create -n timelens python=3.10 -y
conda activate timelens
python -m pip install -U pip setuptools wheel

# Install PyTorch (CPU; for GPU follow pytorch.org)
pip install torch torchvision

# Install remaining deps
pip install -r requirements.txt
```

## Run the UI
```bash
python app.py
# Open the link shown by Gradio (usually http://127.0.0.1:7860/)
```

## CLI (optional)
```bash
python cli_restore.py -i path/to/input.jpg -o restored.jpg
```

Notes:
- First run will auto-download GFPGANv1.3 weights into ./models/.
- RealESRGAN is not used (keeps things light and simple).
