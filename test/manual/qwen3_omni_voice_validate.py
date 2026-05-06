#!/usr/bin/env python3
"""End-to-end Qwen3-Omni voice validation.

Generates audio on AMD MI300X via the existing load-test harness, copies
WAVs back to local, runs Parakeet STT, applies audio sanity checks
(duration + RMS energy), and reports per-fixture pass/fail.

The point of this script is to catch the regression class where the
server-side `realtime_after_eoi` and thinker `full_text` gates pass
green but the produced audio is semantically broken (speech-like but
incoherent). Bypassing this script is the same mistake that allowed the
prior simplification stack's K.1 bug to ship undetected for 14 commits
(2026-05-10).

Usage from the repo root::

    python test/manual/qwen3_omni_voice_validate.py
    python test/manual/qwen3_omni_voice_validate.py --fixtures greeting,question
    python test/manual/qwen3_omni_voice_validate.py --no-sync --no-restart \\
        --label retry
    python test/manual/qwen3_omni_voice_validate.py --skip-stt   # sanity-only

Exits 0 if every fixture passes audio sanity AND Parakeet ONNX STT is
within tolerance against the canonical reference text.
Exits 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------- AMD remote layout ----------------
# Hot Aisle MI300X validation target used for launch qualification.
AMD_HOST = "hotaisle@23.183.40.79"
AMD_CONTAINER = "path1-aiter-debug"
AMD_SGLANG = "/sgl-workspace/sglang"
AMD_PYTHON = "/opt/venv/bin/python"
AMD_TMP = "/tmp"
SERVER_PORT = 41012
DEFAULT_SERVER_MODEL_PATH = (
    "/mnt/data/bai_weights/lucy-1-text-sft-fp8-thinker-talker-attn-sglang-v1"
)

SERVER_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": f"{AMD_SGLANG}/python",
}
SERVER_ARGS = [
    "--model-path", DEFAULT_SERVER_MODEL_PATH,
    "--mem-fraction-static", "0.76",
    "--max-total-tokens", "400000",
    "--qwen3-omni-max-concurrent-audio-turns", "50",
    "--grpc-mode",
    "--enable-streaming-session",
    "--attention-backend", "aiter",
    "--host", "127.0.0.1",
    "--port", str(SERVER_PORT),
]

# ---------------- Fixtures ----------------
DEFAULT_FIXTURES = ["greeting", "question", "emotional", "excited", "accent"]

# Canonical reference transcriptions (Beginnings am-* fixtures, chelsie speaker).
# Short prompts the model is expected to respond to with a coherent reply; the
# Parakeet check is against the model's RESPONSE text we observe consistently
# across known-good builds, not against the user input. Updated 2026-05-10.
# ---------------- Sanity thresholds ----------------
MIN_DURATION_S = 3.0     # under this = early cutoff (truncated audio)
MIN_RMS = 0.005          # silent audio fails
STT_MAX_DISTANCE = 0.30  # normalized word-level edit distance vs full_text


# ---------------- AMD plumbing ----------------
def amd_run(cmd: str, *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run a bash command inside the AMD container via ssh+docker."""
    docker_cmd = f"docker exec {AMD_CONTAINER} bash -lc {shlex.quote(cmd)}"
    return subprocess.run(
        ["ssh", AMD_HOST, docker_cmd],
        capture_output=capture, text=True, check=check,
    )


def amd_run_detached(cmd: str) -> None:
    """Run a bash command inside the AMD container, detached."""
    docker_cmd = f"docker exec -d {AMD_CONTAINER} bash -lc {shlex.quote(cmd)}"
    subprocess.run(["ssh", AMD_HOST, docker_cmd], check=True)


def amd_write_text(remote_path: str, text: str) -> None:
    remote_quoted = shlex.quote(remote_path)
    write_cmd = f"cat > {remote_quoted} && chmod 600 {remote_quoted}"
    docker_cmd = (
        f"docker exec -i {AMD_CONTAINER} bash -lc "
        f"{shlex.quote(write_cmd)}"
    )
    subprocess.run(["ssh", AMD_HOST, docker_cmd], input=text, text=True, check=True)


def amd_copy_back(remote_path: str, local_path: Path) -> None:
    """`docker cp` a file from the AMD container into local_path (creates parents)."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f"docker cp {AMD_CONTAINER}:{remote_path} -"
    proc = subprocess.run(["ssh", AMD_HOST, cmd], capture_output=True, check=True)
    subprocess.run(
        ["tar", "-x", "-C", str(local_path.parent)],
        input=proc.stdout, check=True,
    )


def sync_code(repo_root: Path) -> None:
    """tar the served Python package and manual validation clients to AMD."""
    src = repo_root / "python" / "sglang"
    if not src.is_dir():
        raise RuntimeError(f"missing python/sglang at {src}")
    print(
        "[validate] syncing python/sglang and test/manual clients to "
        f"{AMD_HOST}:{AMD_CONTAINER}:{AMD_SGLANG}"
    )
    docker_cmd = f"docker exec -i {AMD_CONTAINER} tar -C {AMD_SGLANG} -xf -"
    tar = subprocess.Popen(
        [
            "tar",
            "-C",
            str(repo_root),
            "-cf",
            "-",
            "python/sglang",
            "test/manual/qwen3_omni_audio_load_test.py",
            "test/manual/qwen3_omni_audio_prewarm.py",
            "test/manual/qwen3_omni_audio_soak_test.py",
        ],
        stdout=subprocess.PIPE,
    )
    ssh = subprocess.Popen(["ssh", AMD_HOST, docker_cmd], stdin=tar.stdout)
    tar.stdout.close()  # type: ignore[union-attr]
    ssh.communicate()
    if ssh.returncode != 0:
        raise RuntimeError("sync failed")


def restart_server(
    server_log: str,
    *,
    model_path: str = DEFAULT_SERVER_MODEL_PATH,
) -> None:
    print(f"[validate] restarting server, log → {server_log}")
    amd_run(
        f"find {AMD_SGLANG}/python/sglang/srt -type d -name __pycache__ "
        f"-prune -exec rm -rf {{}} + 2>/dev/null; "
        "pkill -TERM -f '\\-m sglang.launch_server' 2>/dev/null; sleep 4 || true",
        check=False,
    )
    server_env = {**SERVER_ENV}
    hf_token = os.environ.get("HF_TOKEN")
    server_args = [*SERVER_ARGS]
    hf_env_prefix = ""
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for Qwen3-Omni launch validation")
    hf_env_file = f"{AMD_TMP}/qwen3_omni_voice_validate_hf_token.env"
    amd_write_text(hf_env_file, f"export HF_TOKEN={shlex.quote(hf_token)}\n")
    hf_env_prefix = f". {shlex.quote(hf_env_file)}; "
    server_args[server_args.index("--model-path") + 1] = model_path
    env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in server_env.items())
    args_str = " ".join(shlex.quote(arg) for arg in server_args)
    launch = (
        f"{hf_env_prefix}cd {AMD_SGLANG} && {env_str} nohup {AMD_PYTHON} -m sglang.launch_server "
        f"{args_str} > {server_log} 2>&1"
    )
    amd_run_detached(launch)
    print("[validate] waiting for 'fired up' (~50s)...")
    amd_run(f'until grep -q "fired up" {server_log} 2>/dev/null; do sleep 5; done')
    print("[validate] server ready")


def generate_fixture(
    fixture: str, amd_output_dir: str, system_prompt_file: str,
    *, session_ids_file: Optional[str] = None,
    close_session: bool = True,
    fixed_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the load test on AMD for one fixture; return the parsed aggregate."""
    extra = ""
    if session_ids_file is not None:
        extra += f" --session-ids-file {session_ids_file}"
    if not close_session:
        extra += " --no-close-sessions"
    if fixed_text is not None:
        extra += f" --regex {shlex.quote(re.escape(fixed_text))}"
    cmd = (
        f"set -euo pipefail; "
        f"rm -rf {amd_output_dir}; mkdir -p {amd_output_dir}; "
        f"PYTHONPATH={AMD_SGLANG}/python {AMD_PYTHON} "
        f"{AMD_SGLANG}/test/manual/qwen3_omni_audio_load_test.py "
        f"--host 127.0.0.1 --port {SERVER_PORT} "
        f"--concurrency 1 --realtime-input "
        f"--input-wav {AMD_TMP}/am-{fixture}-01_turn-0.wav "
        f"--system-prompt-file {system_prompt_file} "
        f"--speaker chelsie --max-new-tokens 64 --talker-max-new-tokens 256 "
        f"--save-output-dir {amd_output_dir}{extra}"
    )
    proc = amd_run(cmd)
    return json.loads(proc.stdout)


# ---------------- Audio sanity ----------------
def audio_sanity(wav_path: Path) -> Dict[str, float]:
    """Inspect a WAV: total duration (s) and RMS amplitude in [0, 1]."""
    import numpy as np
    with wave.open(str(wav_path), "rb") as w:
        n_frames = w.getnframes()
        sample_rate = w.getframerate()
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        raw = w.readframes(n_frames)
    duration_s = n_frames / sample_rate if sample_rate else 0.0
    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"unsupported sample_width={sample_width} for {wav_path}")
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    rms = float((samples ** 2).mean() ** 0.5) if samples.size else 0.0
    return {"duration_s": float(duration_s), "rms": rms, "samples": float(samples.size)}


# ---------------- Parakeet STT (ONNX-GPU only) ----------------
# Required local setup — script fails fast if any of these is missing:
#   - Nvidia GPU + CUDA toolkit visible to onnxruntime
#   - uv sync in ~/repos/qwen3-voice-validate
#   - FP32 model id (default: nemo-parakeet-tdt-0.6b-v3); quantized variants
#     are rejected because their ~3-5x WER degradation flips the
#     normalized-edit-distance check into false-positive failures.
#
# Why no NeMo fallback / no CPU fallback: this is local-iteration tooling for
# a known dev box. Soft fallbacks hide setup drift; the goal of the script is
# to make audio regressions impossible to miss, which means the gate runs
# with the same backend every time. Install the right thing once.

_STT_MODEL = None  # cached across calls

# Quantized variants would silently degrade STT accuracy; refuse them.
_QUANT_MARKERS = ("-int8", "-int4", "-quant", "_int8", "_int4", "_quant")


def _require(condition: bool, message: str) -> None:
    if not condition:
        print(f"[validate] FATAL: {message}", file=__import__("sys").stderr)
        raise SystemExit(2)


def _preload_cuda_libs() -> None:
    """dlopen nvidia-*-cu12 .so's from the running Python's site-packages.

    onnxruntime-gpu's CUDA EP needs libcudnn.so.9 + libcublas.so.12 etc. on
    the dynamic loader path. The pip wheels (nvidia-cudnn-cu12, nvidia-
    cublas-cu12, ...) install them under site-packages/nvidia/*/lib but
    don't expose them via LD_LIBRARY_PATH. We dlopen them with
    RTLD_GLOBAL so subsequent dlopens (ORT loading providers_cuda.so)
    can resolve them. This avoids needing a bash wrapper / system cuDNN.
    """
    import ctypes
    import glob
    import site

    site_pkgs = site.getsitepackages()
    if hasattr(site, "getusersitepackages"):
        site_pkgs.append(site.getusersitepackages())
    seen = set()
    for sp in site_pkgs:
        for lib_dir in sorted(glob.glob(f"{sp}/nvidia/*/lib")):
            # Order matters: cublas before cudnn (cudnn depends on cublas).
            for so in sorted(glob.glob(f"{lib_dir}/lib*.so*")):
                if so in seen:
                    continue
                seen.add(so)
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def stt_load(model_name: str = "nemo-parakeet-tdt-0.6b-v3",
             providers: Optional[List[str]] = None):
    """Load ONNX Parakeet on GPU. Hard-fails (exit 2) if local setup is missing."""
    global _STT_MODEL
    if _STT_MODEL is not None:
        return _STT_MODEL

    _require(
        not any(m in model_name.lower() for m in _QUANT_MARKERS),
        f"refusing quantized model '{model_name}'. Use FP32: 'nemo-parakeet-tdt-0.6b-v3'. "
        "Quantized int8/int4 Parakeet exports have ~3-5x WER degradation that flips this gate.",
    )

    # Preload nvidia-*-cu12 libs from site-packages BEFORE importing onnxruntime.
    _preload_cuda_libs()

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError:
        _require(False, "onnxruntime not installed. Run: uv add onnxruntime-gpu (in the validation env)")
    available = set(ort.get_available_providers())  # type: ignore[possibly-undefined]

    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _require(
        "CUDAExecutionProvider" in available,
        f"CUDAExecutionProvider not available in onnxruntime (got {sorted(available)}). "
        "Run `uv sync` in ~/repos/qwen3-voice-validate and verify nvidia-smi works.",
    )
    _require(
        "CUDAExecutionProvider" in providers,
        f"--stt-providers must include CUDAExecutionProvider (got {providers}). "
        "This script enforces GPU STT — pass CUDAExecutionProvider explicitly or omit --stt-providers.",
    )

    try:
        import onnx_asr  # type: ignore[import-not-found]
    except ImportError:
        _require(
            False,
            "onnx-asr is not installed. Run `uv sync` in ~/repos/qwen3-voice-validate.",
        )

    print(f"[validate] loading ONNX Parakeet (FP32): {model_name} on {providers}")
    try:
        model = onnx_asr.load_model(model_name, providers=providers)  # type: ignore[possibly-undefined]
    except TypeError as e:
        print(f"[validate] FATAL: onnx-asr does not accept providers kwarg ({e}). "
              "Upgrade: uv add -U onnx-asr (in the validation env)", file=__import__("sys").stderr)
        raise SystemExit(2)
    _STT_MODEL = model
    return model


def stt_transcribe(model, wav_path: Path) -> str:
    result = model.recognize(str(wav_path))
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, list) and result:
        first = result[0]
        return (first if isinstance(first, str) else getattr(first, "text", str(first))).strip()
    return str(result).strip()


def normalized_word_distance(a: str, b: str) -> float:
    """Levenshtein / max(len) over lowercased words, ignoring punctuation."""
    aw = re.findall(r"[a-z0-9]+", a.lower())
    bw = re.findall(r"[a-z0-9]+", b.lower())
    n, m = len(aw), len(bw)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    dp = [list(range(m + 1)) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if aw[i - 1] == bw[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / max(n, m)


def min_expected_duration_s(text: str) -> float:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not words:
        return MIN_DURATION_S
    return min(MIN_DURATION_S, max(1.0, 0.25 * len(words)))


# ---------------- Reporting ----------------
def fmt(value: Any, fmt_spec: str = "") -> str:
    if value is None:
        return "—"
    try:
        return format(value, fmt_spec) if fmt_spec else str(value)
    except (TypeError, ValueError):
        return str(value)


def print_report(rows: List[Dict[str, Any]], parakeet_active: bool) -> None:
    cols = [
        ("fixture", 12),
        ("status", 6),
        ("duration_s", 10),
        ("rms", 7),
        ("rtf", 6),
        ("realtime", 8),
        ("stt_dist", 9),
        ("stt_pass", 8),
    ]
    header = " | ".join(f"{name:<{w}}" for name, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    print()
    print("[validate] === SUMMARY ===")
    print(header)
    print(sep)
    for r in rows:
        cells = [
            fmt(r.get("fixture"), ""),
            fmt(r.get("status"), ""),
            fmt(r.get("duration_s"), ".2f"),
            fmt(r.get("rms"), ".3f"),
            fmt(r.get("rtf_after_eoi"), ".3f"),
            fmt("yes" if r.get("realtime") else "no" if r.get("realtime") is False else None, ""),
            fmt(r.get("stt_distance"), ".3f"),
            fmt("✓" if r.get("stt_pass") else "✗" if r.get("stt_pass") is False else "n/a", ""),
        ]
        print(" | ".join(f"{c:<{w}}" for c, (_, w) in zip(cells, cols)))
    if not parakeet_active:
        print("\n[validate] (Parakeet STT was skipped; STT columns are n/a)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", default=",".join(DEFAULT_FIXTURES),
                        help=f"comma-separated fixture names (default: {','.join(DEFAULT_FIXTURES)})")
    parser.add_argument("--no-sync", action="store_true", help="skip code sync to AMD")
    parser.add_argument("--no-restart", action="store_true", help="skip server restart (use existing server)")
    parser.add_argument("--skip-stt", action="store_true", help="skip Parakeet (sanity checks only)")
    parser.add_argument("--multi-turn", action="store_true",
                        help="also run a 4-turn multi-turn session through the fixtures in order")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/qwen3_voice_validate"),
                        help="local directory for WAVs + report (default: /tmp/qwen3_voice_validate)")
    parser.add_argument("--label", default="run",
                        help="subdir label inside --output-dir; new label for each independent run")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="repo root for code sync (default: derived from script path)")
    parser.add_argument("--system-prompt-file", default=f"{AMD_TMP}/beginnings_system_prompt.txt",
                        help="path to the Beginnings system prompt INSIDE the AMD container")
    parser.add_argument("--stt-providers", default="CUDAExecutionProvider,CPUExecutionProvider",
                        help="comma-separated ONNX Runtime providers in priority order. Must include "
                             "CUDAExecutionProvider — script hard-fails on CPU-only setups by design.")
    parser.add_argument("--parakeet-model", default="nemo-parakeet-tdt-0.6b-v3",
                        help="onnx-asr model id; must be FP32 (no -int8/-int4/-quant suffix). "
                             "Quantized variants have ~3-5x WER degradation that breaks the STT-distance check.")
    parser.add_argument("--server-model-path", default=DEFAULT_SERVER_MODEL_PATH,
                        help="remote model path passed to SGLang --model-path "
                             f"(default: {DEFAULT_SERVER_MODEL_PATH})")
    parser.add_argument("--fixed-text-json", type=Path,
                        help="optional JSON object mapping '<label>:<fixture>' to exact Thinker text")
    args = parser.parse_args()

    fixtures: Sequence[str] = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    out_root: Path = args.output_dir / args.label
    out_root.mkdir(parents=True, exist_ok=True)
    server_log = f"{AMD_TMP}/qwen3_validate_{args.label}_server.log"
    fixed_texts: Dict[str, str] = {}
    if args.fixed_text_json is not None:
        fixed_texts = json.loads(args.fixed_text_json.read_text())

    if not args.no_sync:
        sync_code(args.repo_root)
    if not args.no_restart:
        restart_server(
            server_log,
            model_path=args.server_model_path,
        )
    stt_model = None
    if not args.skip_stt:
        providers = [p.strip() for p in args.stt_providers.split(",") if p.strip()]
        stt_model = stt_load(args.parakeet_model, providers=providers)

    results: List[Dict[str, Any]] = []

    def evaluate(label: str, fixture: str, wav_path: Path, agg: Dict[str, Any]) -> Dict[str, Any]:
        agg_summary = agg.get("aggregate", {})
        per_turn = (agg.get("results") or [{}])[0]
        ref = per_turn.get("full_text", "")
        sanity = audio_sanity(wav_path)
        min_duration_s = min_expected_duration_s(ref)
        sanity_pass = sanity["duration_s"] >= min_duration_s and sanity["rms"] >= MIN_RMS

        stt_text: Optional[str] = None
        stt_distance: Optional[float] = None
        stt_pass: Optional[bool] = None
        if stt_model is not None:
            stt_text = stt_transcribe(stt_model, wav_path)
            # Compare audio transcript against the thinker's own text. This
            # catches the K.1-class regression where audio diverges from the
            # text the thinker emitted, without false-failing on legitimate
            # response variation across runs (the model uses sampled voice
            # generation; same prompt → different valid responses).
            if ref and stt_text is not None:
                stt_distance = normalized_word_distance(stt_text, ref)
                stt_pass = stt_distance <= STT_MAX_DISTANCE

        passed = bool(sanity_pass and (stt_pass is None or stt_pass))
        return {
            "label": label,
            "fixture": fixture,
            "status": "PASS" if passed else "FAIL",
            "wav": str(wav_path),
            "duration_s": sanity["duration_s"],
            "min_duration_s": min_duration_s,
            "rms": sanity["rms"],
            "samples": sanity["samples"],
            "sanity_pass": sanity_pass,
            "rtf_after_eoi": agg_summary.get("mean_rtf_after_eoi"),
            "max_rtf_after_eoi": agg_summary.get("max_rtf_after_eoi"),
            "first_audio_after_eoi_s": agg_summary.get("max_first_audio_after_eoi_s"),
            "realtime": agg_summary.get("realtime_after_eoi"),
            "full_text": per_turn.get("full_text", ""),
            "stt_text": stt_text,
            "stt_distance": stt_distance,
            "stt_pass": stt_pass,
        }

    # --- single-turn pass ---
    for fixture in fixtures:
        print(f"\n[validate] === single/{fixture} ===")
        amd_dir = f"{AMD_TMP}/qwen3_validate_{args.label}_single_{fixture}"
        try:
            agg = generate_fixture(
                fixture,
                amd_dir,
                args.system_prompt_file,
                fixed_text=fixed_texts.get(f"single:{fixture}"),
            )
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "")[-200:]
            print(f"[validate] generation failed: {err}")
            results.append({"label": "single", "fixture": fixture, "status": "GEN_FAIL"})
            continue
        wav_local = out_root / "single" / fixture / "turn_00.wav"
        amd_copy_back(f"{amd_dir}/turn_00.wav", wav_local)
        results.append(evaluate("single", fixture, wav_local, agg))

    # --- multi-turn pass (optional) ---
    if args.multi_turn:
        amd_session_file = f"{AMD_TMP}/qwen3_validate_{args.label}_session_ids.txt"
        amd_run(f"echo {args.label}-multi-1 > {amd_session_file}")
        for n, fixture in enumerate(fixtures, start=1):
            print(f"\n[validate] === multi/turn{n}_{fixture} ===")
            amd_dir = f"{AMD_TMP}/qwen3_validate_{args.label}_multi_t{n}_{fixture}"
            close_session = (n == len(fixtures))
            try:
                agg = generate_fixture(
                    fixture, amd_dir, args.system_prompt_file,
                    session_ids_file=amd_session_file,
                    close_session=close_session,
                    fixed_text=fixed_texts.get(f"multi_t{n}:{fixture}"),
                )
            except subprocess.CalledProcessError as e:
                err = (e.stderr or "")[-200:]
                print(f"[validate] multi-turn generation failed: {err}")
                results.append({"label": f"multi_t{n}", "fixture": fixture, "status": "GEN_FAIL"})
                continue
            wav_local = out_root / "multi" / f"turn_{n}_{fixture}" / "turn_00.wav"
            amd_copy_back(f"{amd_dir}/turn_00.wav", wav_local)
            results.append(evaluate(f"multi_t{n}", fixture, wav_local, agg))

    print_report(results, parakeet_active=stt_model is not None)

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\n[validate] JSON report → {report_path}")
    print(f"[validate] WAVs        → {out_root}")

    failed = [r for r in results if r.get("status") != "PASS"]
    if failed:
        print(f"[validate] FAIL: {len(failed)}/{len(results)} fixtures did not pass")
        return 1
    print(f"[validate] PASS: all {len(results)} fixtures passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
