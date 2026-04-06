# SDXL UNet overhead review

Focused notes about where wall-clock time is currently lost in the phone-side SDXL runtime.

## Why split UNet is still required

The split is not an arbitrary design choice.

- full FP16 UNet is roughly **5 GB**;
- practical HTP allocation limit is roughly **3.5 GB**;
- current split artifacts fit individually:
  - encoder ≈ **2.52 GB**
  - decoder ≈ **2.69 GB**

So the current optimization problem is **not** “remove the split first”, but rather “make the split path cheaper to execute”.

## What the deep probe showed

Measured on the live phone runtime with `qnn-net-run` + `qnn-profile-viewer`.

### Single-call half-step cost

| Case | Wall time |
| --- | ---: |
| encoder single | **8312.0 ms** |
| decoder single | **8141.1 ms** |

### `mmap` effect on the same half-step

| Case | Baseline | With `--use_mmap` | Gain |
| --- | ---: | ---: | ---: |
| encoder | 8312.0 ms | **6115.0 ms** | **26.4%** |
| decoder | 8141.1 ms | **6011.4 ms** | **26.2%** |

### Repeat-in-one-process effect

| Case | Single-call wall | repeat×4 avg/inference | Gain |
| --- | ---: | ---: | ---: |
| encoder | 8312.0 ms | **3527.5 ms** | **57.6%** |
| decoder | 8141.1 ms | **4379.7 ms** | **46.2%** |

### Main conclusion

The largest remaining loss is **runtime lifecycle overhead**:

- process launch;
- context/binary load;
- graph init/deinit;
- extra file I/O around every call.

`mmap` is a real cheap win, but it is **not** the whole story.
The repeat×4 result strongly suggests that a future persistent runner / daemon is still the biggest next step.

## CFG clarification

The runtime no longer pays the fully naive `4 subprocesses per step` CFG path.

Current optimized CFG step is:

- **1 batched encoder call** for `uncond + cond`
- **1 batched decoder call** for `uncond + cond`

So CFG above `1.0` still roughly doubles the real denoising work, but it is no longer the older worst-case subprocess pattern.

## What changed in `v0.1.3`

Release `0.1.3` applies the first cheap runtime win everywhere it matters:

- `phone_generate.py` now defaults to **`SDXL_QNN_USE_MMAP=1`**;
- the rooted sample runtime mirrors the same default;
- the APK launch path explicitly exports `SDXL_QNN_USE_MMAP=1`;
- repo-visible debug tools were added:
  - `SDXL/debug/sdxl_speed_probe.py`
  - `SDXL/debug/sdxl_unet_overhead_probe.py`

You can still disable the default for testing:

```bash
export SDXL_QNN_USE_MMAP=0
```

## End-to-end control run after the change

Validated in this session on OnePlus 13 / Snapdragon 8 Elite.

Prompt:
`1girl, upper body, looking at viewer, masterpiece, best quality`

Config:

- 1024×1024
- 8 steps
- seed 777
- CFG = 1.0
- `mmap` ON

Result:

| Stage | Time |
| --- | ---: |
| CLIP | **1.993 s** |
| UNet total | **91.466 s** |
| VAE | **8.992 s** |
| Total | **104.4 s** |

Compared with the earlier public no-CFG baseline of about **126 s**, this control run is about **17.1% faster**.

## Reuse follow-up in `v0.2.3`

The later reuse-tuned phone runtime finally changed the *shape* of the first-step slowdown, not just the total number at the end.

- the first four guided UNet steps on the current fast CFG path now descend roughly as **12.2 → 10.4 → 9.9 → 9.8 s**;
- the first four `CFG=1.0` steps on the no-guidance path sit roughly around **7.4 → 7.4 → 6.2 → 6.5 s** with normal run-to-run jitter;
- the practical README-visible marker moved to **78.0 s total** once those reuse gains were combined with the rest of the tuned path.

That means the runtime is no longer stuck in the older “flat ~12 s guided plateau for the early steps” behavior. There is now clear warm-path decay even before the run reaches the cheaper tail steps.

## Practical recommendation

Short term:

- keep `mmap` enabled by default;
- use the new probes when changing runtime behavior;
- treat the first guided step as the next hot spot to squeeze harder;
- keep an eye on CLIP prompt overhead now that UNet and TAESD are cheaper;
- do not chase `8W8A` again before the runtime overhead path is squeezed harder.

Medium term:

- build a persistent encoder/decoder runner so contexts are loaded once and reused across steps.

## Next likely wins after `v0.2.3`

After the current reuse pass and the TAESD GPU move, the next practical speed targets look like this:

1. **parallel CLIP work as a low-risk experiment** — the cond/uncond text path is now a cleaner fraction of total wall time, so overlapping some of that prompt-side work is worth testing;
2. **revisit the daemon / persistent-runner path as the highest-reward project** — the repeat-in-one-process probe still says lifecycle overhead is the biggest remaining wall-clock tax even though the first daemon attempt regressed;
3. **small VAE/context warmup experiments** — probably not a giant win, but still worth checking once the bigger UNet/CLIP questions are exhausted.

The current guidance is: squeeze the first guided UNet step and CLIP overhead first, but do not forget that the daemon path is still the biggest theoretical payoff if its regression is understood.
