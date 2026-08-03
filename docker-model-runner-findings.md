# Docker Model Runner on this Mac: is it GPU accelerated?

Investigation date: 2026-08-02 to 2026-08-03.

**Verdict: yes, Docker Model Runner runs local models on the Metal GPU on this machine.
Confidence: very high, and it rests on direct measurement rather than on documentation.**

Four independent signals agree, including a direct hardware measurement (100% GPU residency
and 15.3 W draw during generation, against ~1% and single-digit milliwatts at idle) and
generation throughput within 1.6% of a known-Metal endpoint running the same weights. No signal
pointed the other way.

Every number below was measured on this machine. Claims taken from documentation rather than
measured are labeled as such, inline, every time.

---

## 1. Environment

| Item | Value | How known |
|---|---|---|
| Architecture | `arm64` | measured (`uname -m`) |
| macOS | 26.5.2, build 25F84 | measured (`sw_vers`) |
| Chip | Apple M4 Pro, 14 cores (10 performance), 48 GB unified memory | measured (`sysctl`) |
| Docker Desktop | client 28.2.2, engine 29.2.1 | measured (`docker version`) |
| Model Runner | CLI plugin v1.1.5, server v1.1.1 | measured (`docker model version`) |
| Backend in use | `llama.cpp latest-metal`, image `sha256:aa3e239c…`, build `ac4cdde` | measured (`docker model status`) |
| Other backends | `mlx` not installed, `vllm` not installed | measured (`docker model status`) |
| Comparison runtime | Ollama 0.32.5 native, Ollama 0.20.7 in container | measured (`/api/version`) |

Versions were re-checked at the end of the investigation and were unchanged, so all
measurements come from one software state rather than straddling an update.

The client/server version skew (v1.1.5 against v1.1.1) was present throughout and is not
something this investigation controlled for.

---

## 2. What the docs claim, and what I measured

Sources consulted:

- <https://docs.docker.com/ai/model-runner/>
- <https://docs.docker.com/ai/model-runner/api-reference/>
- <https://www.docker.com/blog/docker-model-runner-vllm-metal-macos/>
- <https://github.com/docker/model-runner>

Release notes were planned as a fifth source and could not be obtained:
<https://docs.docker.com/model-runner/release-notes/> returns 404, and notes for v1.1.5
specifically were never located. Nothing below rests on them.

| Question | What the docs say | What I measured | Agree? |
|---|---|---|---|
| Host process or Linux VM on macOS? | "On macOS and Windows, the engines don't run inside a container", they run in a seatbelt sandbox | `com.docker.llama-server` runs from `~/.docker/bin/inference/`, a host path, listening on a UNIX socket under `~/Library/Containers/com.docker.docker/Data/inference-0.sock`. During generation it was the only busy process; `com.docker.virtualization` stayed at 0.0% CPU | Yes |
| GPU requirements | macOS: "Apple Silicon" is the only requirement listed | Apple M4 Pro, GPU used with no configuration | Yes |
| Model storage and pull | Pulled from Docker Hub, an OCI registry, or Hugging Face, "stored locally". **No disk path given anywhere in the docs** | `~/.docker/models/`, an OCI layout (`blobs/`, `manifests/`, `layout.json`, `models.json`). Weights land at `~/.docker/models/bundles/sha256/<digest>/model/model.gguf` | Docs incomplete |
| Endpoints | Host `http://localhost:12434`, containers `http://model-runner.docker.internal`, OpenAI routes under `/engines/v1`, plus Ollama-compatible `/api/*` and Anthropic-compatible `/anthropic/v1/*` | Paths correct once reachable. **But `localhost:12434` refused connections out of the box**: host-side TCP is off by default and has to be enabled | Docs misleading by omission |

### Where docs and observation disagree

**The documented host endpoint does not work until you enable it, and the docs I read never
say so.** `curl http://localhost:12434/` gave connection refused on both IPv4 and IPv6 with
Model Runner running and healthy. The fix is `docker desktop enable model-runner --tcp=12434`
(note the `=`; the space form is rejected). Anyone following the API reference literally will
conclude Model Runner is broken. This is the single biggest gap between the documentation and
the product.

**The docs never state where models land on disk.** Found empirically. A Model Runner GitHub
README fetch returned a path of `~/.cache/nim`, which is NVIDIA NIM's cache and not this
product; I discarded it rather than report it, and mention it only as a warning that
secondary sources on this topic are unreliable.

**Backend naming.** The docs list llama.cpp, vLLM, and Diffusers. This install reports
llama.cpp, mlx, and vllm. Planning on 2026-08-02 saw only llama.cpp and mlx, so the backend
list is not stable release to release.

---

## 3. Evidence for GPU acceleration, ranked by strength

Ranked strongest first, with the reason each sits where it does. The spike's rule was that at
least two independent signals are required and that the engine's self-report can never be
sufficient alone. Four were obtained.

### Signal 1: GPU hardware residency and power draw (strongest)

This observes the hardware directly rather than inferring from behavior. It is the only signal
here that cannot be explained by a fast CPU path. Sampled with
`sudo powermetrics --samplers gpu_power -i 500 -n 8`.

| State | GPU HW active residency | GPU power | Active frequency |
|---|---|---|---|
| Idle, nothing generating | 0.51% to 1.00% | 0 to 8 mW | 338 MHz (lowest bin) |
| During a 2000-token generation | **100.00%**, seven consecutive samples | **15231 to 15354 mW** | **1578 MHz, the top bin, 100% of the time** |
| Tail sample as generation ended | 24.96% | 3500 mW | mixed |

A CPU-only inference path cannot produce 100% GPU residency at the maximum clock for the
duration of a generation. This is conclusive on its own.

### Signal 2: Throughput parity with a known-GPU endpoint

Model Runner generates at 386.0 tok/s. The native Metal Ollama, independently confirmed
GPU-resident on this machine via `size_vram` of 911715860 bytes from its `/api/ps`, generates
at 380.1 tok/s on the same weights at the same quantization. The CPU-only container endpoint,
independently confirmed via `size_vram` of 0, manages 72.9 tok/s.

Model Runner sits within 1.6% of the GPU end and 5.3x above the CPU end. A CPU-only runtime
could not land there.

### Signal 3: Process accounting

`com.docker.llama-server` was launched with `--threads 7`. A CPU-only run with 7 threads would
show roughly 700% CPU on a multi-core machine. Sampled three times during a sustained
2000-token generation:

```
88785  16.0%  2127504 KB  /Users/alexschiltmans/.docker/bin/inference/com.docker.llama-server
88785   1.1%  2127584 KB  ...
88785   0.4%  2127584 KB  ...
```

CPU stayed between 0.4% and 16% while the work was being done, which means something other
than the CPU was doing it. Meanwhile `com.docker.virtualization`, the Linux VM, held at 0.0%
throughout, confirming the work is not happening inside the VM. This also settles the
host-process question: the binary lives under `~/.docker/bin/`, a host path, and serves over a
host UNIX socket.

### Signal 4: The engine's own reporting (weakest)

Ranked last because it is a statement about which binary shipped and which flags were passed,
not about which device executed anything. It corroborates and does not conclude.

- `docker model status`: backend `llama.cpp latest-metal`
- `docker model logs`: `installed llama-server with gpuSupport=true`
- launch arguments: `-ngl 999`, meaning offload all layers to GPU
- device enumeration: `MTL0 : Apple M4 Pro (38338 MiB, 38338 MiB free)`, alongside `BLAS : Accelerate` and `CPU : Apple M4 Pro`

Worth recording how nearly this signal misled me. My first pass grepped the logs for
`metal|gpu|device|backend`, which matched the `BLAS : Accelerate` line but not the `MTL0` line
above it, and the partial output read as though only a CPU BLAS device had been enumerated.
Reading the full load sequence rather than a filtered view corrected it. A keyword grep over
engine logs is not a reliable way to answer this question.

### Signals that disagreed

None. All four point the same way.

---

## 4. Benchmark

Identical prompt, `max_tokens` 256, `temperature` 0, `seed` 7, three interleaved rounds so
thermal drift could not land on one path. Interleaved rather than three trials per path back
to back.

Seed determinism was verified on Model Runner before benchmarking: two runs at seed 7 produced
byte-identical output (`sha256` prefix `0e886dbb63118742` both times).

**Weights are identical across the three paths**: LiquidAI LFM2 350M, Q8_0 quantization.
See the caveats section for how this was arranged and what remains imperfect.

### Steady state

| Path | Device | Prompt eval tok/s (mean) | Generation tok/s (mean) | Gen spread across 3 runs |
|---|---|---|---|---|
| **Model Runner** | Metal GPU | 1211.0 | **386.01** | 12.96 |
| Ollama native | Metal GPU | 2378.4 | **380.10** | 7.15 |
| Ollama container | CPU | 716.0 | **72.90** | 24.98 |

Per-round generation figures: Model Runner 377.73 / 390.70 / 389.61; Ollama native
382.92 / 375.77 / 381.61; Ollama container 59.85 / 74.03 / 84.83.

### Cold load, reported separately from steady state

Measured as a first 8-token request after unloading the model from every runtime.

| Path | Cold load |
|---|---|
| Model Runner | 0.64 s |
| Ollama native | 0.40 s |
| Ollama container | 0.62 s |

All three are fast enough not to matter for a 350M model. This would need re-measuring before
drawing any conclusion about larger models.

### Resident memory during a loaded run

| Path | RSS |
|---|---|
| Model Runner (`com.docker.llama-server`) | 2087.6 MB |
| Ollama native (`llama-server`) | 844.0 MB |
| Ollama container (whole container) | 603.9 MiB |

Model Runner's 2.5x memory use against native Ollama is the one clear cost found. The likely
cause is configuration rather than inefficiency: Model Runner loads with `n_ctx = 128000` and
four parallel slots, while this project configures Ollama with `OLLAMA_NUM_CTX` far lower. The
KV cache scales with both. **This explanation is a hypothesis, not measured**; I did not test
it by varying the context size.

### The prefill gap

Model Runner's prompt evaluation is roughly half the native Ollama figure (1211 against 2378
tok/s), consistently across all three rounds, despite matching it on generation. **I did not
determine why.** Plausible causes are batch size settings, the `--no-mmap` flag Model Runner
passes, or the four-slot configuration. This is stated as an open question rather than
explained, and it matters for a RAG workload, where prompts are long and answers are short.

---

## 5. Can this project use Docker Model Runner?

**Yes, with no code changes, and it works today.** This is measured, not predicted: the app's
existing client stack was pointed at Model Runner and ran.

Model Runner exposes an Ollama-compatible API on the same port as its OpenAI-compatible one,
so the project's existing `ollama` provider reaches it directly:

```
connection : {'status': 'connected', 'version': '0.1.0', 'model': 'huggingface.co/liquidai/lfm2-350m-gguf:Q8_0', 'url': 'http://localhost:12434'}
model      : available
placement  : {'placement': 'unknown', 'url': 'http://localhost:12434'}
sidebar    : - A local LLM via Ollama (**huggingface.co/liquidai/lfm2-350m-gguf:Q8_0**) at `http://localhost:12434`
```

Setting `OLLAMA_BASE_URL=http://localhost:12434` and `LLM_MODEL_NAME` to the Model Runner model
name is the whole change. `LLM_PROVIDER=openai-compat` with
`LLM_BASE_URL=http://localhost:12434/engines/v1` is the other supported route.

### Five things to know before adopting it

**Placement reporting goes dark.** Model Runner's `/api/ps` returns entries without a
`size_vram` field. The app's placement check therefore reports `unknown`, so the sidebar names
the endpoint but states no placement. This is the correct behavior and not a defect: a missing
`size_vram` is deliberately treated as not-determinable rather than as CPU, and this is the
first real backend to exercise that path.
The practical consequence is that on Model Runner you would be trusting the endpoint name
alone, without the GPU confirmation Ollama gives for free. Given this spike measured the answer
directly, that is an acceptable trade, but it is a loss of a safety net that was added
specifically because this project was once burned by an unnoticed CPU backend.

**Host TCP must be enabled**, and it is off by default. Any setup documentation would have to
carry `docker desktop enable model-runner --tcp=12434` or users hit connection refused on the
endpoint the official docs tell them to use.

**Model naming changes.** Models are referenced as `hf.co/LiquidAI/LFM2-350M-GGUF:Q8_0` rather
than `sam860/LFM2:350m`, and the `/api/tags` name comes back normalized as
`huggingface.co/liquidai/lfm2-350m-gguf:Q8_0`. Two things happen to it: `hf.co` expands to
`huggingface.co`, and the repository path is lowercased. **The quantization tag keeps its case**
(`:Q8_0`, not `:q8_0`). The project's model matching is exact-match with a `:latest` fallback,
so the configured name has to reproduce that form character for character. Read it off
`/api/tags`, which is where the form above was observed, rather than retyping it.

**A missing model suggests the wrong command.** When `LLM_MODEL_NAME` does not match anything
Model Runner serves, the app's availability check reports `not_found` and suggests
`ollama pull <name>`. Model Runner is reached through the `ollama` provider but is not Ollama,
so that command cannot succeed here. The working command is `docker model pull <name>`, with the
`hf.co/LiquidAI/...` spelling rather than the normalized one. The remedy text branches on
whether the endpoint looks like a native or a compose-networked Ollama, and Model Runner's
`localhost:12434` is neither, so it falls through to the native suggestion. Documented rather
than fixed: the fix belongs in a change of its own, not in the spike that found it.

**The prefill gap is the wrong shape for RAG.** This project sends long retrieved contexts and
gets short answers back, so prompt evaluation is where its time goes. Model Runner's generation
parity is reassuring but not the metric that matters most here, and it currently loses on the
one that does. That gap is unexplained and might be tunable.

### Recommendation

Model Runner is a viable GPU-backed backend for this project and there is no correctness reason
to avoid it. But **nothing here argues for switching**: native Ollama matches it on generation,
beats it 2x on prefill, uses a third of the memory, and reports GPU placement so the app can
show it. Model Runner's advantages are packaging and distribution, which this project does not
currently need.

The case for revisiting would be wanting OCI-registry model distribution, or wanting to drop
the native Ollama install. If that comes up, the prefill gap should be investigated first,
since it is the number this workload actually pays.

---

## 6. Caveats and confounders

**Quantization was a confounder and was removed, but not perfectly.** The default pull of
`hf.co/LiquidAI/LFM2-350M-GGUF` gives Q4_K_M, while the project's Ollama model
`sam860/LFM2:350m` is Q8_0. Benchmarking those against each other would have measured
quantization as much as device. I pulled `hf.co/LiquidAI/LFM2-350M-GGUF:Q8_0` instead, so both
paths run LFM2 350M at Q8_0 from the same upstream. What remains imperfect: the Ollama copy is
`sam860`'s repack rather than LiquidAI's own file, so the two are the same model and
quantization but not verified byte-identical GGUFs.

**Two Ollama versions, not one.** Native is 0.32.5, the container is 0.20.7. The CPU baseline
therefore differs from the native reference in both device and software version. This does not
affect the GPU verdict, which rests on signals 1 and 3, but the precise 5.3x CPU-to-GPU ratio
should be read as approximate.

**The CPU baseline is a different runtime from the subject.** Model Runner could not be forced
onto CPU: `docker model run --help` and `docker model --help` expose no device or backend
selector, and the backend is chosen by the runner. The comparison is therefore Model Runner
against containerized Ollama, which varies device and runtime together. This is why signal 1
matters: it settles the device question without needing the baseline at all.

**A 350M model is small.** Per-request overhead is a larger fraction of total time than it
would be for a 7B model, and memory pressure never became a factor on a 48 GB machine. None of
these figures should be extrapolated to larger models.

**Ollama's `size` field is not stable across versions.** The same model reports 911715860 bytes
native and 539311104 in the 0.20.7 container. Only `size_vram` was used for placement, so this
did not affect anything, but any future attempt to compute a GPU fraction as `size_vram / size`
would be built on sand.

**`docker model df` and the filesystem disagree.** `df` reports 1.24 GB of models; `du -sh
~/.docker/models/` reports 824 MB. Not investigated. Also, `docker model pull` printed
nonsensical progress, counting past its own total ("Downloaded 965.64MB of 484.27MB").
Cosmetic, but it does not inspire confidence in the accounting.

**Thermal state was not controlled** beyond interleaving runs. No sustained load was applied
before measuring, so these are warm-but-not-throttled figures.

---

## 7. What I could not verify

- **Why prefill is half the native rate.** Stated as an open question above. Not investigated.
- **Why Model Runner uses 2.5x the memory.** The `n_ctx = 128000` and four-slot hypothesis is
  untested.
- **The container-side endpoint.** `http://model-runner.docker.internal/engines/v1` is
  documented but was never exercised; all measurements went through the host endpoint.
- **The MLX and vLLM backends.** Both report not installed. Installing either would have
  changed the subject partway through, so they were left alone. Whether MLX would beat the
  llama.cpp Metal path on Apple Silicon is unanswered.
- **Behavior under concurrent load.** `docker model bench` measures concurrency scaling and was
  not run. Every measurement here is single-request.
- **Larger models.** Nothing above 354M parameters was tested.
- **Release notes for v1.1.5.** The documented release-notes URL 404s and no replacement was
  found, so the version-specific half of the documentation review did not happen.
- **What `docker model ls` prints.** Every model name recorded here came from `/api/tags`. The
  CLI's own listing was never captured, so the two are not known to agree.

---

## 8. Reproducing this

```bash
docker model status                                    # backend and health
docker desktop enable model-runner --tcp=12434         # host endpoint, off by default
docker model pull hf.co/LiquidAI/LFM2-350M-GGUF:Q8_0
docker model logs | tail -60                           # read whole load sequence, do not grep
sudo powermetrics --samplers gpu_power -i 500 -n 8     # during a sustained generation
ps -A -o pid,%cpu,rss,comm | grep llama-server         # expect low CPU, high RSS
```

## 9. State left on this machine

- **Model Runner host TCP is now enabled on port 12434.** It was off before. Revert with
  `docker desktop enable model-runner --no-tcp`.
- Two models added to `~/.docker/models/`, 824 MB on disk: the Q4_K_M default pull and the
  Q8_0 variant. Remove with
  `docker model rm hf.co/LiquidAI/LFM2-350M-GGUF hf.co/LiquidAI/LFM2-350M-GGUF:Q8_0`.
- `codebase-rag-ollama` was created as the CPU baseline and has been **stopped**, with
  `sam860/LFM2:350m` left in its `codebase_rag_ollama_data` volume. Remove entirely with
  `docker compose -f docker/compose-dev.yml --profile ollama down -v`.
- Qdrant and Langfuse were already running before this investigation and were left running.
- Nothing was installed outside Docker. The native Ollama and its models are untouched.
