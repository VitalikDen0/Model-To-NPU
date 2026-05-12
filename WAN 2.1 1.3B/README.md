# WAN 2.1 1.3B

This folder is the starting workspace for **Wan 2.1 T2V 1.3B** experiments aimed at a future phone-oriented path.

> [!WARNING]
> WAN end-to-end flow in this repository is **NOT VERIFIED** (`НЕ ПРОВЕРЕН`) and may not work at all.
> Hot resolution switching (`WxH` bucket selection) is test-stage behavior.
> [!IMPORTANT]
> The repository still has only one **practically validated** phone pipeline today: `SDXL/`.
> Everything inside `WAN 2.1 1.3B/` is currently **research / selection / probing / preparation work**.

## Current working decision

### Start now

- **Primary practical start:** `IPostYellow/Wan2.1-T2V-1.3B-INT8-Diffusers`
  - public Diffusers layout;
  - about **15.4 GB** on the Hugging Face tree page;
  - no confirmed step reduction, but clearly better positioned than full-size baseline assets for constrained hardware;
  - best current **"start immediately"** choice.

### Stable fallback

- **Official baseline:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
  - official 1.3B path;
  - official notes indicate **480p is the recommended stable target**;
  - **720p is possible but less stable**.

### Few-step / Lightning-like research branch

- **Algorithmically closest match to the "Lightning / Distill / similar" request:** `NVlabs/rcm`
  - documents **1–4 step** inference capability;
  - conceptually much closer to a real low-step path than pure INT8 quantization.
- **But:** during this pass, a clean public **Wan 2.1 1.3B distilled student checkpoint** was **not confidently verified**.
- `worstcoder/Wan` currently looks like **converted official Wan checkpoints** for rCM/TurboDiffusion-style use, not a proven ready-made low-step 1.3B release by itself.

### Not the right main start for this 1.3B branch

- `lightx2v/Wan2.1-Distill-Models`
  - real and useful as a **4-step distillation reference**;
  - but the accessible public set is centered around **14B / I2V-heavy** artifacts, not a clean public 1.3B Lightning-style drop-in.

### Watchlist, but not trusted yet

- `maty0505/Wan1.3B-rCM-8step-iter20000`
  - interesting name;
  - current public tree looked effectively **empty / unverified** during this pass.

## Resolution strategy

- **Step 1:** prove a **480p** path first.
- **Step 2:** only then try **720p**.
- Reason: official Wan 2.1 1.3B guidance itself treats **480p** as the safer operating point, while **720p** is less stable even before phone-side conversion/runtime constraints are added.

## Connected phone snapshot from this session

- device: `CPH2653`
- Android: `15`
- platform: `sun`
- physical screen: `1440x3168`
- density: `640`

The screen itself is obviously high-resolution enough, but that does **not** change the current model-side conclusion: **480p-first remains the sane starting point**.

## Files in this folder

- `wan_tool.py` — main helper for:
  - listing candidate repos;
  - printing the current recommendation;
  - downloading selected Hugging Face repos;
  - probing the connected phone via ADB.
- `host_phone_wan_generate.py` — host-orchestrated WAN run path (phone QNN for transformer, host/phone VAE options), now with requested `--width/--height` manifest-bucket resolution selection.
- `run_end_to_end.ps1` — practical beta wrapper for end-to-end WAN host-phone flow.
- `download_wan_assets.py` — convenience wrapper around `wan_tool.py download`
- `phone_check.py` — convenience wrapper around `wan_tool.py phone-check`

## Suggested workflow

1. Inspect the current candidate matrix.
2. Probe the connected phone.
3. Start with the INT8 Diffusers repo at **480p**.
4. Keep the official Diffusers repo as the clean baseline.
5. Revisit `rCM` only after a verifiable public 1.3B few-step checkpoint is confirmed.
6. Treat **720p** as a second-phase experiment, not as the first milestone.

## Checked sources

- `https://huggingface.co/lightx2v/Wan2.1-Distill-Models`
- `https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B`
- `https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- `https://huggingface.co/IPostYellow/Wan2.1-T2V-1.3B-INT8-Diffusers`
- `https://github.com/NVlabs/rcm`
- `https://github.com/NVlabs/rcm/blob/main/Wan.md`
- `https://huggingface.co/worstcoder/Wan`
- `https://huggingface.co/maty0505/Wan1.3B-rCM-8step-iter20000`
- `https://github.com/ModelTC/LightX2V`

## Short reality check

A phone-usable Wan path is still going to be heavy.

What we have done here is the correct **first engineering step**:

- narrow the model family;
- separate it into its own top-level folder;
- document what is actually public and usable;
- avoid pretending that every repo with `rCM`, `distill`, or `lightning` in the name is immediately deployable.

In other words: less hype, more signal — a rare and beautiful creature.
