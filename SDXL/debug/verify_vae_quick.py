"""Quick VAE verification with CUDA for speed."""
import torch, numpy as np, gc, os, sys
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFUSERS = os.path.join(ROOT, "sdxl_npu", "diffusers_pipeline")
ONNX_PATH = os.path.join(ROOT, "sdxl_npu", "onnx_clip_vae", "vae_decoder.onnx")

def compare(name, a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    mae = np.mean(np.abs(a - b))
    rmse = np.sqrt(np.mean((a - b) ** 2))
    max_abs = np.max(np.abs(a - b))
    cos = np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12)
    print(f"  [{name}] mae={mae:.2e}, rmse={rmse:.2e}, max_abs={max_abs:.2e}, cos={cos:.10f}")
    if cos > 0.9999 and max_abs < 0.1:
        print(f"  [{name}] PASS")
    elif cos > 0.999:
        print(f"  [{name}] MARGINAL")
    else:
        print(f"  [{name}] FAIL")

print("Loading VAE (FP32 on CUDA)...")
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(
    DIFFUSERS + "/vae", local_files_only=True, torch_dtype=torch.float32
).eval().cuda()

torch.manual_seed(42)
latent = torch.randn(1, 4, 128, 128, dtype=torch.float32, device="cuda")

print("PyTorch inference...")
with torch.no_grad():
    pt_image = vae.decode(latent, return_dict=False)[0].cpu().numpy()
print(f"  PyTorch: {pt_image.shape}, range=[{pt_image.min():.4f}, {pt_image.max():.4f}]")

del vae; gc.collect(); torch.cuda.empty_cache()

print("ORT inference (CUDA)...")
import onnxruntime as ort
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
try:
    sess = ort.InferenceSession(ONNX_PATH, providers=providers)
except:
    print("  CUDA ORT not available, falling back to CPU...")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

out_names = [o.name for o in sess.get_outputs()]
print(f"  ONNX outputs: {out_names}")

latent_np = latent.cpu().numpy()
ort_image = sess.run(None, {"latent": latent_np})[0]
print(f"  ONNX: {ort_image.shape}, range=[{ort_image.min():.4f}, {ort_image.max():.4f}]")

print("Comparing PyTorch vs ONNX:")
compare("pt_fp32_vs_onnx", pt_image, ort_image)
print("\nDone!")
