# D1D-Turbo

**Drum one-shots generated from pure noise in 3 steps, on a CPU.**

D1D-Turbo generates **kick, snare and hi-hat one-shots as raw waveform**: 16,384
samples at 32 kHz (0.512 s), mono. Each sound takes **3 network passes, about
0.35 s on a 6-core CPU**, with no GPU and no cloud. The model has 4.9 M parameters.

This repository holds the inference code. The weights, model card and audio
samples are on Hugging Face: **[huggingface.co/sixf0ur/D1D-Turbo](https://huggingface.co/sixf0ur/D1D-Turbo)**.
The write-up is at [sixf0ur.github.io](https://sixf0ur.github.io/2026/11/08/d1d-drum-generator.html).

| | needs | files |
|---|---|---|
| **C++ library** | a C++17 compiler; header-only, standard library only (no BLAS, no ONNX, no DLLs) | `cpp/d1d.hpp` + `model.bin` |
| **C++ command line** | same; built on the library | `cpp/drumgen.cpp` + `model.bin` |
| **Python** | `numpy`, `onnxruntime`, `soundfile` | `gen_onnx.py` + `model.onnx` |

## Get the model

Put the model files in the repository root:

```bash
curl -LO https://huggingface.co/sixf0ur/D1D-Turbo/resolve/main/model.bin      # C++
curl -LO https://huggingface.co/sixf0ur/D1D-Turbo/resolve/main/model.onnx     # Python
curl -LO https://huggingface.co/sixf0ur/D1D-Turbo/resolve/main/test_io.bin    # optional: build self-test
```

`curl` is included in Windows 10 and later. Alternatively:
`huggingface-cli download sixf0ur/D1D-Turbo model.bin model.onnx test_io.bin --local-dir .`

## C++ command line

```bash
# Linux / macOS / MinGW
sh cpp/build.sh
cpp/drumgen --model model.bin --n 4 --out out

# Windows (Visual Studio or Build Tools, "Desktop development with C++")
cpp\build.bat
cpp\drumgen.exe --model model.bin --n 4 --out out
```

| option | |
|---|---|
| `--cls kick\|snare\|hat\|all` | what to generate (default `all`) |
| `--n N` | samples per class (default 4) |
| `--steps S`, `--cfg W` | sampler (defaults stored in `model.bin`: 3, 1.0) |
| `--seed S` | base seed; sample k uses S+k, printed per file (default: time-based) |
| `--float` | write 32-bit float WAVs instead of 16-bit PCM |
| `--threads T` | default: all logical cores |
| `--test test_io.bin` | check this build against PyTorch reference outputs |
| `--bench`, `--profile` | time one network pass (with per-op breakdown) |

`drumgen --test test_io.bin` must print `PASS`. On the reference machine the
maximum error is 3.3e-6 per pass and 6.7e-6 after the 3-step sampler.

The build uses an AVX2/FMA kernel when available (`/arch:AVX2`,
`-march=native`). Otherwise, or with `build.bat portable` / `build.sh portable`,
it uses a portable path that gives the same results more slowly.

## C++ library (header-only)

`cpp/d1d.hpp` is the complete inference engine in one header: C++17, standard
library only, no other files to compile or link. `drumgen` is built on it.

```cpp
#include "d1d.hpp"

d1d::Model model = d1d::Model::load("model.bin");        // from a file ...
// d1d::Model model = d1d::Model::load(data, size);       // ... or from bytes in memory

d1d::GenParams p;
p.seed = 42;                                              // same seed -> same sound
std::vector<float> kick = model.generate(d1d::Drum::Kick, p);   // 16384 floats @ 32 kHz

d1d::save_wav("kick.wav", kick);                          // 16-bit PCM file
std::vector<uint8_t> wav = d1d::wav_bytes(kick);          // or the WAV file image in memory
```

```bash
g++ -O3 -march=native -std=c++17 -pthread my_app.cpp     # GCC / Clang
cl /O2 /arch:AVX2 /std:c++17 /EHsc my_app.cpp            # MSVC
```

`GenParams` fields:

| field | default | |
|---|---|---|
| `seed` | 0 | noise seed: same seed + same model = same sound, on every platform |
| `steps` | 0 | Euler steps, 0 = the model's default (3) |
| `cfg` | 0 | CFG weight, 0 = the model's default (1.0, guidance is baked in) |
| `normalize`, `peak` | true, 0.95 | peak-normalise the output (the model's level is arbitrary) |
| `progress` | none | `bool(int done, int total)`, called after every step; return false to cancel |

The rest of the API:
- **Loading:** `Model::load(path, threads = 0)` and
  `Model::load(data, size, threads = 0)`. `threads = 0` uses all cores and
  `threads = 1` runs on the calling thread only. The bytes are copied, so the
  buffer may be freed afterwards.
- **Generating:**
  - `generate(drum, params)` returns a `std::vector<float>`, which is empty if
    the call was cancelled.
  - `generate(drum, params, float* out)` writes into your own buffer of
    `d1d::kLength` floats and returns false if cancelled.
- **Model info:** `default_steps()`, `default_cfg()`, `gated()`, `threads()`.
- **Low level:**
  - `make_noise(seed, out)` gives the start noise for a seed.
  - `sample(cls, x, steps, cfg)` runs the Euler sampler in place on your own
    noise.
  - `forward(x, t, cls, v)` does one network pass.
- **Output:** `wav_bytes(x, sample_rate, float32)` builds a WAV file image in
  memory and `save_wav(path, x, sample_rate, float32)` writes it to disk, as
  16-bit PCM or 32-bit float.
- **Constants:** `d1d::kSampleRate` = 32000, `d1d::kLength` = 16384,
  `d1d::kOnset` = 256 (where the hit starts).
- **Threads and errors:** each `Model` owns its worker threads. Calls on one
  `Model` from several threads are safe but run one after another, so for
  parallel generation load one `Model` per thread. Errors throw
  `std::runtime_error`, for example on a truncated or foreign model file.
- **Embedding the model:** `model.bin` (19.5 MB) can come from anywhere: a file,
  an asset pack, a download, or bytes compiled into the program (C23/C++26
  `#embed`, `incbin`, a Windows resource). Big-endian hosts work too, because the
  file is parsed byte by byte.

`cpp/example.cpp` shows all of this: loading from a file and from memory,
progress, an in-memory WAV and determinism.

**Porting.** `d1d.hpp` is written to be ported easily: one file, no platform
code apart from one optional SIMD block, and threading confined to a single
helper. See the porting notes at the top of the file.

## Python (ONNX Runtime)

```bash
pip install -r requirements.txt
python gen_onnx.py --n 4 --out out              # 4 x kick, snare, hat -> out/*.wav
python gen_onnx.py --cls snare --n 8 --seed 1   # one class, reproducible
python gen_onnx.py --n 16 --bench
```

`gen_onnx.py` reads the sampler defaults (`steps=3`, `cfg_w=1.0`) from the ONNX
metadata. Python and C++ use different random number generators, so the same
seed gives a different (equally valid) sound in each.

## Speed

AMD Ryzen 5 5500 (6 cores / 12 threads, AVX2), batch 1:

| | per network pass | per sound (3 passes) |
|---|---|---|
| C++, 12 threads | 114 ms | **0.34 s** |
| C++, 1 thread | 481 ms | 1.4 s |
| Python, ONNX Runtime fp32 | ~118 ms | 0.35 to 0.38 s |

## How it works

- **Raw waveform, no VAE:** a 1D U-Net over the 16,384 samples (5 resolution
  levels, FiLM residual blocks, self-attention with positions at the lowest
  resolution), trained with rectified flow.
- **3 steps instead of 100:** reflow distillation of a 50-step teacher with
  classifier-free guidance, so the guidance is baked in.
- **Exact silence:** a learned output gate (225 parameters) produces digital
  silence before the hit and after the decay, with no post-processing.

Architecture, training and evaluation are described in the
[model card](https://huggingface.co/sixf0ur/D1D-Turbo).

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See
`License.txt`. For commercial licensing contact info@sixf0ur.com.
