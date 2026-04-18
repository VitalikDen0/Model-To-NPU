# SDXL runtime architecture for `v0.2.4-beta`

This note captures the **current architecture shift** for the next validation wave.
The goal of `v0.2.4-beta` is **not** to claim a new best phone benchmark yet.
The goal is to make the runtime more explicit, more self-contained, and easier to optimize without guessing.

## Why this beta exists

After the phone reset, the old **`62.0 s`** marker stopped being reproducible.
That does **not** automatically mean the raw UNet math became slower.
The more likely suspects are:

- context lifecycle / QNN driver state differences after redeploy;
- missing or inactive backend-extension fast path on the rebuilt phone layout;
- orchestration overhead around the split UNet path;
- extra buffer churn, copies, and file I/O inside `phone_generate.py`;
- warmup / runtime residency behavior no longer matching the earlier “good” state.

So the immediate job of `v0.2.4-beta` is to reduce the amount of runtime behavior that is left to chance.

## What was added in this beta

### 1. Optional native C accelerator layer

New files:

- `phone_runtime_accel.py`
- `NPU/runtime_accel/sdxl_runtime_accel.c`
- `scripts/build_runtime_accel.py`
- `scripts/bench_runtime_accel.py`

The native layer is intentionally **small and surgical**.
It handles the hottest non-QNN math that is still happening on the Python side:

- model-input scaling per sigma;
- Euler step update;
- CFG + Euler step fusion;
- NCHW -> NHWC conversion with scaling for VAE input.

The Python runtime keeps a **NumPy fallback**, so the pipeline stays runnable even when the shared library is absent.

### 2. Tensor arena / memory ownership

The runtime now has a dedicated tensor arena for reusable buffers instead of recreating the same arrays step after step.
The arena owns:

- model-input scratch;
- ping-pong latent step buffers;
- reusable timestep tensor;
- reusable NHWC VAE input buffer.

That means the scheduler path is now much closer to a native runtime model:

- explicit scratch ownership;
- predictable buffer reuse;
- less accidental allocation churn;
- easier profiling when the phone comes back online.

### 3. Deployable native runtime hook

`deploy_to_phone.py` can now ship the Python wrapper and, when present, the Android shared library.
This gives us a clean on-device experiment path:

- rebuild `libsdxl_runtime_accel.so`;
- deploy it into `phone_gen/lib/`;
- rerun the exact same generation path;
- compare logs and timings.

### 4. APK-side wiring for the optional native layer

The APK launch script can now export the accelerator library path when the file exists in the deployed phone tree.
This keeps the UI path and direct-Termux/debug path aligned.

## What `stable-diffusion.cpp` suggests we should imitate architecturally

Without copying code, there are several patterns worth borrowing conceptually.

### Explicit async job ownership

The `examples/server/async_jobs.*` side is a useful reminder that generation should be modeled as a **job** with:

- queued state;
- running state;
- cancellation;
- completion/failure records;
- TTL/cleanup.

That pattern matters for the phone too.
A future APK-native runtime service should own generation requests in the same explicit way instead of treating generation as a one-shot shell process.

### Clear separation of parameter/cache/compute memory

`stable-diffusion.cpp` keeps cache and compute contexts explicit.
That is exactly the direction we want for Snapdragon/QNN too:

- persistent model/context residency;
- separate scratch arenas;
- minimal per-step rebuild work;
- explicit cleanup points.

### Sample-cache family (`ucache`, `spectrum`, etc.)

Their cache docs are a strong hint that the next big win may come not from shaving microseconds in Python, but from **skipping work safely** on later denoise steps.
For our SDXL split-UNet path, the most interesting candidates are:

- condition-level residual reuse on later steps;
- step forecasting for no-guidance tail steps;
- optional skip policies activated only after warmup.

That is future work, but now the runtime is finally being shaped so those experiments can be inserted deliberately.

## Immediate experiments to run once the phone is back

1. Build the Android accelerator library:
   - `python scripts/build_runtime_accel.py --target android-arm64`
2. Deploy it together with the updated runtime.
3. Compare four cases on the same prompt/seed:
   - native accel OFF / backend-ext OFF
   - native accel ON / backend-ext OFF
   - native accel OFF / backend-ext ON
   - native accel ON / backend-ext ON
4. Keep `preview` both OFF and ON as separate tracks.
5. Record whether the historical `62 s` class only returns when backend-ext is truly active, or whether buffer/orchestration work also matters more than expected.

## Termux exit strategy

The desired end state is still:

- user picks a model directory;
- APK does the rest;
- no manual Termux rituals.

There are two realistic paths.

### Path A — hidden bundled Python (short-term practical)

Keep Python, but make it invisible to the user:

- bundle the offline runtime inside APK assets;
- extract it into app-private storage;
- run only through the app;
- stop depending on a separately managed external Termux install.

This repo already has the first pieces of that path.

### Path B — native runtime service (long-term correct)

Move orchestration out of Python entirely:

- native job service inside the APK;
- persistent QNN context ownership in-process;
- shared-memory or direct-buffer tensor passing;
- no shell indirection;
- no `python3` discovery path at runtime.

That is the architecture most likely to make the old fast path reproducible and stable.

## Universal Snapdragon direction

The repository should not remain “just OnePlus 13 tuning forever”.
The next architecture step should introduce a small **Snapdragon runtime descriptor** layer, for example:

- SoC / HTP generation identifier;
- recommended backend-extension profile;
- hvx/vtcm suggestions;
- available runtime libraries;
- optional preview backend;
- known-safe model/context shapes.

That would let one runtime choose better defaults per Snapdragon family instead of hardcoding everything around a single test device.

## Practical conclusion

`v0.2.4-beta` is an **architecture-first beta**:

- the APK is moving toward self-contained runtime packaging;
- Python-side hot math now has a native C path;
- buffer lifetime is more explicit;
- deploy/build/bench tooling exists for controlled experiments;
- the repo is now better prepared for both **phone re-validation** and the later **universal Snapdragon runtime** direction.
