#!/data/data/com.termux/files/usr/bin/python3
"""
SDXL Lightning — standalone phone generation.
Runs entirely on phone: tokenizer + scheduler + QNN NPU inference.
No PC, no torch, no diffusers required.

Usage (in Termux):
  python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "1girl, anime, cherry blossoms"
  python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "cat on windowsill" --seed 777
  python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py "dark castle" --cfg 2.0 --neg "blurry, bad"
"""
import argparse
import json
import importlib
import math
import os
import re
import struct
import subprocess
import sys
import threading
import time

import numpy as np

# ─── Paths ───
DR = "/data/local/tmp/sdxl_qnn"
CONTEXTS = {
    "clip_l": f"{DR}/context/clip_l.serialized.bin.bin",
    "clip_g": f"{DR}/context/clip_g.serialized.bin.bin",
    "encoder": f"{DR}/context/unet_encoder_fp16.serialized.bin.bin",
    "decoder": f"{DR}/context/unet_decoder_fp16.serialized.bin.bin",
    "vae":     f"{DR}/context/vae_decoder.serialized.bin.bin",
}
PREVIEW_PNG = f"{DR}/outputs/preview_current.png"
# TAESD runs on CPU via onnxruntime (not QNN) — avoids NPU context overhead and layout bugs.
# Export: python SDXL/export_taesd_to_onnx.py
# Push:   adb push sdxl_npu/taesd_decoder/taesd_decoder.onnx /data/local/tmp/sdxl_qnn/phone_gen/
# Install:pip install onnxruntime  (in Termux)
TAESD_ONNX = f"{DR}/phone_gen/taesd_decoder.onnx"
TOKENIZER_DIR = f"{DR}/phone_gen/tokenizer"
OUTPUT_DIR = f"{DR}/outputs"
WORK_DIR = f"{DR}/phone_gen/work"
QNN_NET_RUN = f"{DR}/bin/qnn-net-run"
QNN_LIB = f"{DR}/lib"

# ─── QNN runtime environment (cached, avoid os.environ.copy() per call) ───
_QNN_ENV: dict = {}
_PRINT_LOCK = threading.Lock()

def _log(line: str = "") -> None:
    with _PRINT_LOCK:
        print(line, flush=True)


def _get_qnn_env() -> dict:
    """Build QNN env dict once and cache it."""
    if not _QNN_ENV:
        _QNN_ENV.update(os.environ)
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        _QNN_ENV["LD_LIBRARY_PATH"] = (
            f"{QNN_LIB}:{DR}/bin:{DR}/model"
            + (f":{existing_ld}" if existing_ld else "")
        )
        _QNN_ENV["ADSP_LIBRARY_PATH"] = (
            f"{QNN_LIB};/vendor/lib64/rfs/dsp;"
            f"/vendor/lib/rfsa/adsp;/vendor/dsp"
        )
    return _QNN_ENV


# ─── TAESD CPU decoder (onnxruntime) ────────────────────────────────────────────────────────────────
# Runs in a background thread — overlaps with UNet NPU execution (free latency).
# TAESD is 5 MB; on Cortex-X925 inference ≈ 0.5–2 s, much less than UNet × 7–14 s/step.

_ort_session = None   # cached InferenceSession
_ort_avail: bool | None = None  # None=not-checked, True/False
_preview_thread: threading.Thread | None = None


def _get_ort_session():
    """Lazily create and cache an onnxruntime CPU session for TAESD."""
    global _ort_session, _ort_avail
    if _ort_avail is False:
        return None
    if _ort_session is not None:
        return _ort_session

    if not os.path.exists(TAESD_ONNX):
        _ort_avail = False
        _log(f"  [TAESD] ONNX not found: {TAESD_ONNX}")
        _log("  [TAESD] Export: python SDXL/export_taesd_to_onnx.py --validate")
        _log(f"  [TAESD] Push:   adb push sdxl_npu/taesd_decoder/taesd_decoder.onnx {TAESD_ONNX}")
        return None

    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError:
        _ort_avail = False
        _log("  [TAESD] onnxruntime not found — run in Termux: pip install onnxruntime")
        return None

    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4  # Oryon: use performance cores
        # We only support CPU; GPU would need QnnExecutionProvider which has the same issues
        _ort_session = ort.InferenceSession(
            TAESD_ONNX,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        _ort_avail = True
        _log("  [TAESD] onnxruntime session ready (CPU)")
    except Exception as e:
        _ort_avail = False
        _log(f"  [TAESD] ort session failed: {e}")
    return _ort_session


def _prime_ctx_bg(paths: list) -> list:
    """Read context binary files into OS page cache on background threads.

    Rationale from NPU architecture docs:
    - Context binary loading from UFS flash dominates per-step overhead.
    - qnn-net-run reads the binary with standard Linux file I/O → OS page cache.
    - If pages are already cached (RAM), the file-read cost drops from ~1-2s to ~30ms.
    - Running this in parallel with CLIP inference is free (CLIP runs on NPU, not CPU/IO).

    Note: CDSP initialization + graph parsing still happens per-call regardless of cache.
    The real silver bullet is a persistent daemon (see TODO below), but this is a free win.
    """
    def _read(path):
        try:
            with open(path, "rb") as f:
                while f.read(8 * 1024 * 1024):  # 8 MB chunks
                    pass
        except Exception:
            pass

    threads = [
        threading.Thread(target=_read, args=(p,), daemon=True)
        for p in paths if os.path.exists(p)
    ]
    for t in threads:
        t.start()
    return threads


# Debug: print SoC temperature in denoise loop to detect thermal throttling.
# Enable with: SDXL_SHOW_TEMP=1 python3 generate.py ...
SHOW_TEMP = os.environ.get("SDXL_SHOW_TEMP", "0") == "1"

def _phone_temp() -> tuple:
    """Read SoC/NPU temperature zone (milli-Celsius → Celsius). Returns (temp, name)."""
    keywords = ("cpu", "npu", "dsp", "soc", "thermal")
    for zone in range(20):
        try:
            tp = f"/sys/class/thermal/thermal_zone{zone}/type"
            with open(tp) as f:
                ztype = f.read().strip().lower()
            if any(k in ztype for k in keywords):
                with open(f"/sys/class/thermal/thermal_zone{zone}/temp") as f:
                    return int(f.read().strip()) / 1000.0, ztype
        except Exception:
            continue
    return None, None

# ─── CLIP BPE Tokenizer (pure Python, no dependencies) ───

def _bytes_to_unicode():
    """Same mapping as openai/clip."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class CLIPTokenizer:
    """Minimal CLIP BPE tokenizer — works without transformers/tokenizers."""

    def __init__(self, vocab_path, merges_path, pad_token_id=49407):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        # Skip header line
        merges = [tuple(line.split()) for line in lines[1:]]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"""
            r"""[a-zA-Z\u00C0-\u024F\u0400-\u04FF\u0500-\u052F]+|[0-9]|[^\s\w]+""",
            re.IGNORECASE,
        )
        self.bos_id = self.encoder.get("<|startoftext|>", 49406)
        self.eos_id = self.encoder.get("<|endoftext|>", 49407)
        self.pad_id = pad_token_id

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def _tokenize(self, text):
        text = text.lower().strip()
        tokens = []
        matches = self.pat.findall(text)
        for token in matches:
            if token in ("<|startoftext|>", "<|endoftext|>"):
                tokens.append(token)
                continue
            encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens = self._bpe(encoded).split(" ")
            tokens.extend(bpe_tokens)
        return tokens

    def encode(self, text, max_length=77):
        """Encode text to token IDs with BOS/EOS/padding."""
        bpe_tokens = self._tokenize(text)
        ids = [self.bos_id]
        for t in bpe_tokens:
            if t in self.encoder:
                ids.append(self.encoder[t])
            else:
                ids.append(self.eos_id)  # unk → eos
        ids.append(self.eos_id)
        # Truncate
        if len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_id]
        # Pad
        while len(ids) < max_length:
            ids.append(self.pad_id)
        return ids


# ─── Euler Discrete Scheduler (pure numpy) ───

class EulerDiscreteScheduler:
    """Minimal EulerDiscreteScheduler for SDXL Lightning."""

    def __init__(self, num_train_timesteps=1000, beta_start=0.00085,
                 beta_end=0.012, beta_schedule="scaled_linear",
                 prediction_type="epsilon", steps_offset=1):
        self.num_train = num_train_timesteps
        self.prediction_type = prediction_type
        self.steps_offset = steps_offset

        if beta_schedule == "scaled_linear":
            betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps,
                                dtype=np.float64) ** 2
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # Precompute sigmas for all timesteps
        self._all_sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        self.timesteps = np.array([], dtype=np.int64)
        self.sigmas = np.array([], dtype=np.float32)
        self.init_noise_sigma = 1.0

    def set_timesteps(self, num_steps):
        """Set timesteps with trailing spacing (used by Lightning)."""
        # Trailing spacing: np.round(np.arange(N, 0, -N/steps)) - 1
        step_ratio = self.num_train / num_steps
        timesteps = np.round(np.arange(self.num_train, 0, -step_ratio)).astype(np.int64) - 1

        sigmas = np.array([self._all_sigmas[t] for t in timesteps], dtype=np.float64)
        sigmas = np.append(sigmas, 0.0)

        self.timesteps = timesteps
        self.sigmas = sigmas.astype(np.float32)
        self.init_noise_sigma = float(max(self.sigmas))

    def scale_model_input(self, sample, step_index):
        """Scale input by sigma (Karras schedule)."""
        sigma = self.sigmas[step_index]
        return sample / ((sigma**2 + 1) ** 0.5)

    def step(self, model_output, step_index, sample):
        """Single Euler step."""
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        if self.prediction_type == "epsilon":
            pred_x0 = sample - sigma * model_output
        elif self.prediction_type == "v_prediction":
            pred_x0 = sigma * sample + (1 - sigma**2)**0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Euler method
        d = (sample - pred_x0) / sigma
        prev_sample = sample + (sigma_next - sigma) * d
        return prev_sample


# ─── QNN Net Run helper ───

def qnn_run(ctx_path, input_list_path, output_dir, native=False):
    """Run QNN context on NPU via qnn-net-run.

    Uses direct exec (no shell wrapper) — avoids fork→sh→exec overhead (~10-20ms/call).

    # ─── ARCHITECTURE NOTE: why each call takes ~7s ──────────────────────────────
    # From qnn-profile-viewer on-device: actual NPU execute time ≈ 7ms (FP16 ~few×).
    # The remaining ~7000ms per call is dominated by:
    #   1. CDSP runtime initialization — DSP firmware context setup (~1-3s)
    #   2. Context binary parsing & graph loading into DSP DDR (~1-2s, partially
    #      reducible by OS page cache priming via _prime_ctx_bg)
    #   3. Input/output buffer allocation (~100-500ms)
    # Solution: persistent inference daemon that loads context ONCE and loops.
    #   → Eliminates steps 1-3 for steps 2..N → estimated 1-3s/step instead of 7s.
    #   → Requires ~250 lines of C++ using QNN Runtime API + ndkbuild.
    #   → See: NPU/qnn_inf_server.cpp (TODO)
    # ─────────────────────────────────────────────────────────────────────────────
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        QNN_NET_RUN,
        "--retrieve_context", ctx_path,
        "--backend", f"{QNN_LIB}/libQnnHtp.so",
        "--input_list", input_list_path,
        "--output_dir", output_dir,
        "--perf_profile", "burst",
        "--log_level", "warn",
    ]
    if native:
        cmd.append("--use_native_output_files")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        env=_get_qnn_env(),
        capture_output=True, text=True, timeout=120
    )
    elapsed = (time.time() - t0) * 1000
    if result.returncode != 0:
        print(f"  [qnn-net-run ERROR] {result.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError(f"qnn-net-run failed: exit {result.returncode}")
    return elapsed


# ─── Main generation pipeline ───

DEFAULT_NEG = "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, blurry"

def generate(prompt, seed=42, steps=8, cfg_scale=3.5, neg_prompt=None,
             stretch=True, name=None, preview=False, progressive_cfg=False):
    """
    progressive_cfg: Run CFG only on first ceil(steps/2) steps, then uncond-only.
                     Cuts UNet time by ~40% with minimal quality loss on Lightning.
    preview:         Decode each step's latent with TAESD and save preview PNG.
                     Requires taesd_decoder context on phone.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    use_cfg = cfg_scale > 1.0
    if neg_prompt is None:
        neg_prompt = DEFAULT_NEG if use_cfg else ""

    # Warm OS page cache for UNet contexts while CLIP runs on NPU.
    # File-read portion of context loading: ~1-2s from flash → ~30ms from cache.
    # CDSP init overhead is separate (~3-5s/call) — not affected, but this is free.
    _ctx_prime_threads = _prime_ctx_bg([
        CONTEXTS["encoder"], CONTEXTS["decoder"],
    ])

    _log(f"Prompt: {prompt}")
    if use_cfg:
        _log(f"Neg:    {neg_prompt[:80]}{'...' if len(neg_prompt) > 80 else ''}")
    _log(f"Seed: {seed}, Steps: {steps}, CFG: {cfg_scale}")
    t_total = time.time()

    # ── Load tokenizers ──
    tok_l = CLIPTokenizer(
        f"{TOKENIZER_DIR}/vocab.json",
        f"{TOKENIZER_DIR}/merges.txt",
        pad_token_id=49407,  # CLIP-L: pad = <|endoftext|>
    )
    tok_g = CLIPTokenizer(
        f"{TOKENIZER_DIR}/vocab.json",
        f"{TOKENIZER_DIR}/merges.txt",
        pad_token_id=0,  # CLIP-G: pad = "!"
    )

    # ── Setup scheduler ──
    sched = EulerDiscreteScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        steps_offset=1,
    )
    sched.set_timesteps(steps)

    # ── 1. CLIP ──
    def run_clip(text, tag):
        ids_l = np.array(tok_l.encode(text, 77), dtype=np.float32).reshape(1, 77)
        ids_g = np.array(tok_g.encode(text, 77), dtype=np.float32).reshape(1, 77)

        cd = f"{WORK_DIR}/clip/{tag}"
        os.makedirs(cd, exist_ok=True)
        ids_l.tofile(f"{cd}/ids_l.raw")
        ids_g.tofile(f"{cd}/ids_g.raw")

        with open(f"{cd}/il_l.txt", "w") as f:
            f.write(f"{cd}/ids_l.raw\n")
        with open(f"{cd}/il_g.txt", "w") as f:
            f.write(f"{cd}/ids_g.raw\n")

        ms_l = qnn_run(CONTEXTS["clip_l"], f"{cd}/il_l.txt", f"{cd}/out_l")
        ms_g = qnn_run(CONTEXTS["clip_g"], f"{cd}/il_g.txt", f"{cd}/out_g")

        cl = np.fromfile(f"{cd}/out_l/Result_0/penultimate_hidden.raw", np.float32).reshape(1, 77, 768)
        cg = np.fromfile(f"{cd}/out_g/Result_0/penultimate_hidden.raw", np.float32).reshape(1, 77, 1280)
        te = np.fromfile(f"{cd}/out_g/Result_0/text_embeds.raw", np.float32).reshape(1, 1280)

        pe = np.concatenate([cl, cg], axis=-1)  # [1, 77, 2048]
        return pe, te, ms_l, ms_g

    pe_cond, te_cond, ms_l, ms_g = run_clip(prompt, "cond")
    _log(f"[CLIP cond] L={ms_l:.0f}ms G={ms_g:.0f}ms")
    ms_clip = ms_l + ms_g
    pe_uncond = None
    te_uncond = None

    if use_cfg:
        pe_uncond, te_uncond, ms_l2, ms_g2 = run_clip(neg_prompt, "uncond")
        _log(f"[CLIP uncond] L={ms_l2:.0f}ms G={ms_g2:.0f}ms")
        ms_clip += ms_l2 + ms_g2

    # Wait for context priming threads (they're likely done by now; minimal wait if not)
    for t in _ctx_prime_threads:
        t.join(timeout=0.5)

    # ── 2. Prepare UNet inputs ──
    def push_unet_data(pe, te, tag):
        """Save encoder_hidden_states, text_embeds, time_ids for a condition."""
        d = f"{WORK_DIR}/unet/{tag}"
        os.makedirs(d, exist_ok=True)

        # encoder_hidden_states [1,77,2048] — standard layout for AI Hub split
        enc = pe.astype(np.float32)
        enc.tofile(f"{d}/enc.raw")

        # text_embeds [1,1280]
        te.astype(np.float32).tofile(f"{d}/te.raw")

        # time_ids [1,6] — fixed 1024x1024
        tid = np.array([[1024, 1024, 0, 0, 1024, 1024]], dtype=np.float32)
        tid.tofile(f"{d}/tid.raw")

    push_unet_data(pe_cond, te_cond, "cond")
    if use_cfg:
        assert pe_uncond is not None and te_uncond is not None
        push_unet_data(pe_uncond, te_uncond, "uncond")

    # ── 3. Denoise loop ──
    rng = np.random.RandomState(seed)
    # Generate latents matching torch's randn with the same seed
    latents = rng.randn(1, 4, 128, 128).astype(np.float32)
    latents = latents * sched.init_noise_sigma
    total_unet_ms = 0

    # Pre-create a single reusable work dir (avoids mkdir overhead per step)
    _ensure_unet_workdirs(use_cfg)

    # Progressive CFG: apply guidance only on first half of steps.
    # Composition is determined early; last steps refine details (guidance not needed).
    cfg_cutoff = (steps // 2 + 1) if (use_cfg and progressive_cfg) else steps
    if progressive_cfg and use_cfg:
        _log(f"  [Progressive CFG] CFG on steps 1..{cfg_cutoff}, uncond-only after")

    for si in range(steps):
        t = sched.timesteps[si]
        lat_in = sched.scale_model_input(latents, si)

        # Progressive CFG: drop guidance on later steps
        step_uses_cfg = use_cfg and (si < cfg_cutoff)

        temp_str = ""
        if SHOW_TEMP:
            temp, zone = _phone_temp()
            if temp is not None:
                temp_str = f" [{temp:.1f}°C {zone}]"

        if step_uses_cfg:
            np_cond, np_uncond, ms = _run_unet_split_cfg(
                lat_in, t,
                f"{WORK_DIR}/unet/cond",
                f"{WORK_DIR}/unet/uncond",
            )
            noise_pred = np_uncond + cfg_scale * (np_cond - np_uncond)
        else:
            noise_pred, ms = _run_unet_split(lat_in, t, 0, "cond")

        total_unet_ms += ms
        cfg_str = " CFG" if step_uses_cfg else ""
        _log(
            f"  [UNet {si+1}/{steps}]{cfg_str}{temp_str} "
            f"{ms:.0f}ms [{noise_pred.min():.2f}..{noise_pred.max():.2f}]"
        )

        latents = sched.step(noise_pred, si, latents)

        # Launch TAESD preview in background (CPU, overlaps with next UNet step)
        if preview:
            _start_bg_preview(latents.copy(), si, steps)

    # Wait for final preview thread before proceeding to VAE
    if preview:
        _join_preview_thread()

    _log(f"  UNet total: {total_unet_ms:.0f}ms ({total_unet_ms/steps:.0f}ms/step)")

    # ── 4. VAE decode ──
    scaling_factor = 0.13025
    lat_sc = latents / scaling_factor

    vd = f"{WORK_DIR}/vae"
    os.makedirs(vd, exist_ok=True)
    # VAE expects NHWC
    lat_nhwc = np.transpose(lat_sc, (0, 2, 3, 1)).astype(np.float32)
    lat_nhwc.tofile(f"{vd}/lat.raw")
    with open(f"{vd}/il.txt", "w") as f:
        f.write(f"{vd}/lat.raw\n")

    ms_vae = qnn_run(CONTEXTS["vae"], f"{vd}/il.txt", f"{vd}/out", native=True)
    _log(f"[VAE] {ms_vae:.0f}ms")

    raw = np.fromfile(f"{vd}/out/Result_0/image_native.raw", np.float16).astype(np.float32)
    expected = 1024 * 1024 * 3
    if raw.size != expected:
        raise ValueError(f"VAE output: expected {expected} elements, got {raw.size}")
    img = raw.reshape(1024, 1024, 3)
    img = np.clip(img / 2 + 0.5, 0, 1)

    if stretch:
        lo, hi = np.percentile(img, [0.5, 99.5])
        if hi - lo > 0.05:
            img = np.clip((img - lo) / (hi - lo), 0, 1)

    img_u8 = (img * 255).astype(np.uint8)

    # ── 5. Save ──
    from PIL import Image
    tag = name or f"gen_s{seed}"
    out_path = f"{OUTPUT_DIR}/{tag}.png"
    Image.fromarray(img_u8).save(out_path)

    elapsed = time.time() - t_total
    _log(f"\n{'=' * 40}")
    _log(f"Saved: {out_path}")
    _log(f"CLIP: {ms_clip:.0f}ms | UNet: {total_unet_ms:.0f}ms | VAE: {ms_vae:.0f}ms")
    _log(f"Total: {elapsed:.1f}s")
    return out_path


def _preview_step(latents: np.ndarray, step_idx: int, total_steps: int) -> None:
    """Decode current denoised latents with TAESD on CPU (onnxruntime).

    Input:  [1,4,128,128] NCHW float32 — raw denoised latents (no scaling_factor division).
    Output: saves PREVIEW_PNG as RGB PNG [1024×1024].
    Called from background thread — must be thread-safe (only writes PREVIEW_PNG).
    """
    sess = _get_ort_session()
    if sess is None:
        return

    t0 = time.time()
    try:
        # TAESD expects [1,4,128,128] NCHW float32 — same layout as our latents
        result = sess.run(None, {"latents": latents.astype(np.float32)})
    except Exception as e:
        print(f"  [TAESD] inference error: {e}", flush=True)
        return

    # Output: [1,3,1024,1024] NCHW float32 in [-1, 1]
    out = result[0]  # [1,3,1024,1024]
    img = out[0].transpose(1, 2, 0)  # NCHW → HWC [1024,1024,3]
    img = np.clip(img / 2.0 + 0.5, 0.0, 1.0)
    img_u8 = (img * 255).astype(np.uint8)

    from PIL import Image
    tmp_path = PREVIEW_PNG + ".tmp"
    Image.fromarray(img_u8).save(tmp_path, format="PNG")
    os.replace(tmp_path, PREVIEW_PNG)
    try:
        os.chmod(PREVIEW_PNG, 0o644)
    except Exception:
        pass

    ms = (time.time() - t0) * 1000
    _log(f"  [PREVIEW step {step_idx+1}/{total_steps}] CPU {ms:.0f}ms")


def _start_bg_preview(latents_copy: np.ndarray, step_idx: int, total_steps: int) -> None:
    """Launch _preview_step in a background thread (overlaps with next UNet call)."""
    global _preview_thread
    if _preview_thread and _preview_thread.is_alive():
        _preview_thread.join(timeout=0.2)
        if _preview_thread.is_alive():
            _log(f"  [PREVIEW step {step_idx+1}/{total_steps}] skipped (previous decode still running)")
            return
    _preview_thread = threading.Thread(
        target=_preview_step,
        args=(latents_copy, step_idx, total_steps),
        daemon=True,
    )
    _preview_thread.start()


def _join_preview_thread(timeout: float = 10.0) -> None:
    """Wait for background preview thread to finish."""
    global _preview_thread
    if _preview_thread and _preview_thread.is_alive():
        _preview_thread.join(timeout=timeout)


# ─── UNet helpers (split encoder + decoder) ───────────────────────────────────

def _ensure_unet_workdirs(use_cfg):
    """Pre-create all work directories so per-step mkdir is skipped."""
    for tag in (["cond", "uncond"] if use_cfg else ["cond"]):
        os.makedirs(f"{WORK_DIR}/unet/{tag}/out_enc", exist_ok=True)
        os.makedirs(f"{WORK_DIR}/unet/{tag}/out_dec", exist_ok=True)
    if use_cfg:
        os.makedirs(f"{WORK_DIR}/unet/enc_batch", exist_ok=True)
        os.makedirs(f"{WORK_DIR}/unet/dec_batch", exist_ok=True)


def _enc_dec_inputs(base, smp_path, ts_path):
    """Return (enc_entries, dec_entries_builder) for one condition."""
    enc = [
        f"{base}/enc.raw",   # [0] encoder_hidden_states
        ts_path,              # [1] timestep
        f"{base}/tid.raw",   # [2] time_ids
        f"{base}/te.raw",    # [3] text_embeds
        smp_path,             # [4] sample
    ]
    return enc


def _dec_entries_from_enc_out(base, enc_out_dir):
    """Build decoder input list from encoder output directory."""
    dec = [
        f"{base}/enc.raw",               # [0] encoder_hidden_states
        f"{enc_out_dir}/output_0.raw",    # [1] mid_out
        f"{enc_out_dir}/output_9.raw",    # [2] skip_8
        f"{enc_out_dir}/output_10.raw",   # [3] temb
    ]
    for i in range(8, 0, -1):
        dec.append(f"{enc_out_dir}/output_{i}.raw")  # skip_7→skip_0
    return dec


def _read_noise_pred(out_dec_dir, result_idx=0):
    """Read decoder noise_pred output (auto-detect float32/float16)."""
    out_path = f"{out_dec_dir}/Result_{result_idx}/output_0.raw"
    raw_bytes = os.path.getsize(out_path)
    expected_f32 = 1 * 4 * 128 * 128 * 4
    expected_f16 = 1 * 4 * 128 * 128 * 2
    if raw_bytes == expected_f32:
        d = np.fromfile(out_path, np.float32)
    elif raw_bytes == expected_f16:
        d = np.fromfile(out_path, np.float16).astype(np.float32)
    else:
        raise ValueError(f"Decoder output: unexpected {raw_bytes} bytes at {out_path}")
    return d.reshape(1, 4, 128, 128)


def _run_unet_split(latent_np, timestep, step_idx, tag):
    """Run split UNet (encoder + decoder) on NPU — single condition."""
    # Reuse a fixed work dir (step_idx ignored — files overwritten each step)
    base = f"{WORK_DIR}/unet/{tag}"
    enc_out = f"{base}/out_enc"
    dec_out = f"{base}/out_dec"
    os.makedirs(enc_out, exist_ok=True)
    os.makedirs(dec_out, exist_ok=True)

    smp_path = f"{base}/smp.raw"
    ts_path = f"{base}/ts.raw"

    latent_np.astype(np.float32).tofile(smp_path)
    np.array([float(timestep)], dtype=np.float32).tofile(ts_path)

    enc_entries = _enc_dec_inputs(base, smp_path, ts_path)
    il_enc = f"{base}/il_enc.txt"
    with open(il_enc, "w") as f:
        f.write(" ".join(enc_entries) + "\n")

    ms_enc = qnn_run(CONTEXTS["encoder"], il_enc, enc_out)

    dec_entries = _dec_entries_from_enc_out(base, f"{enc_out}/Result_0")
    il_dec = f"{base}/il_dec.txt"
    with open(il_dec, "w") as f:
        f.write(" ".join(dec_entries) + "\n")

    ms_dec = qnn_run(CONTEXTS["decoder"], il_dec, dec_out)

    return _read_noise_pred(dec_out, 0), ms_enc + ms_dec


def _run_unet_split_cfg(latent_np, timestep, cond_base, uncond_base):
    """Optimized CFG: run cond+uncond as 2-line input_list in ONE subprocess each.

    Instead of 4 qnn-net-run calls, we make 2:
      - encoder: processes uncond then cond in one process
      - decoder: same
    Outputs land in Result_0/ (uncond) and Result_1/ (cond).
    """
    # Common files (same latent/timestep for both conditions)
    smp_path = f"{cond_base}/smp.raw"
    ts_path = f"{cond_base}/ts.raw"
    smp_np = latent_np.astype(np.float32)
    ts_np = np.array([float(timestep)], dtype=np.float32)
    smp_np.tofile(smp_path)
    ts_np.tofile(ts_path)
    # uncond uses the same values — write them there too
    smp_np.tofile(f"{uncond_base}/smp.raw")
    ts_np.tofile(f"{uncond_base}/ts.raw")

    enc_out_batch = f"{WORK_DIR}/unet/enc_batch"
    dec_out_batch = f"{WORK_DIR}/unet/dec_batch"
    os.makedirs(enc_out_batch, exist_ok=True)
    os.makedirs(dec_out_batch, exist_ok=True)

    # ── Batched Encoder: 2 inferences in one qnn-net-run ──
    # Line 0 → uncond (Result_0), Line 1 → cond (Result_1)
    enc_uncond = _enc_dec_inputs(uncond_base, f"{uncond_base}/smp.raw", ts_path)
    enc_cond = _enc_dec_inputs(cond_base, smp_path, ts_path)

    il_enc_batch = f"{cond_base}/il_enc_batch.txt"
    with open(il_enc_batch, "w") as f:
        f.write(" ".join(enc_uncond) + "\n")
        f.write(" ".join(enc_cond) + "\n")

    # Single shared output dir; Result_0 = uncond, Result_1 = cond
    ms_enc = qnn_run(CONTEXTS["encoder"], il_enc_batch, enc_out_batch)

    # ── Batched Decoder: 2 inferences in one qnn-net-run ──
    dec_uncond = _dec_entries_from_enc_out(uncond_base, f"{enc_out_batch}/Result_0")
    dec_cond = _dec_entries_from_enc_out(cond_base, f"{enc_out_batch}/Result_1")

    il_dec_batch = f"{cond_base}/il_dec_batch.txt"
    with open(il_dec_batch, "w") as f:
        f.write(" ".join(dec_uncond) + "\n")
        f.write(" ".join(dec_cond) + "\n")

    dec_out_batch = f"{WORK_DIR}/unet/dec_batch"
    ms_dec = qnn_run(CONTEXTS["decoder"], il_dec_batch, dec_out_batch)

    np_cond = _read_noise_pred(dec_out_batch, 1)
    np_uncond = _read_noise_pred(dec_out_batch, 0)

    return np_cond, np_uncond, ms_enc + ms_dec


# ─── CLI ───

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SDXL Lightning — Phone NPU (standalone)")
    ap.add_argument("prompt", type=str, help="Text prompt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=3.5,
                    help="CFG scale (1.0=off, 3.5=default for NPU quality)")
    ap.add_argument("--neg", type=str, default=None,
                    help="Negative prompt (auto-set if --cfg > 1.0)")
    ap.add_argument("--no-stretch", action="store_true",
                    help="Disable contrast stretching")
    ap.add_argument("--name", type=str, default=None,
                    help="Output filename (without .png)")
    ap.add_argument("--preview", action="store_true",
                    help="Decode each step with TAESD for live preview (~200-500ms extra/step)")
    ap.add_argument("--prog-cfg", action="store_true",
                    help="Progressive CFG: guidance on first half of steps only (~40%% faster)")
    a = ap.parse_args()

    generate(
        a.prompt, seed=a.seed, steps=a.steps,
        cfg_scale=a.cfg, neg_prompt=a.neg,
        stretch=not a.no_stretch, name=a.name,
        preview=a.preview, progressive_cfg=a.prog_cfg,
    )
