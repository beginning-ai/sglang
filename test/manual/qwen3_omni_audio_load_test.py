#!/usr/bin/env python3
"""Concurrent Qwen3-Omni audio-turn smoke/load client for the gRPC API."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shlex
import subprocess
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import grpc
import numpy as np

from sglang.srt.grpc import qwen3_omni_audio_pb2 as audio_pb2
from sglang.srt.grpc import qwen3_omni_audio_pb2_grpc as audio_pb2_grpc

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
PCM16_BYTES_PER_SAMPLE = 2


@dataclass
class TurnResult:
    index: int
    request_id: str
    session_id: str
    ok: bool
    error: str | None
    text: str
    full_text: str
    prompt_tokens: int
    completion_tokens: int
    audio_bytes: int
    output_audio_s: float
    first_audio_after_eoi_s: float | None
    elapsed_after_eoi_s: float | None
    elapsed_total_s: float
    rtf_after_eoi: float | None
    realtime_after_eoi: bool | None
    audio_stream_rtf_after_first_audio: float | None
    realtime_audio_stream_after_first_audio: bool | None
    output_path: str | None
    stt_text: str | None
    stt_reference_text: str | None
    stt_normalized_distance: float | None
    stt_match: bool | None


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != PCM16_BYTES_PER_SAMPLE:
        raise ValueError(f"{path} must be PCM16, got {sample_width * 8}-bit")

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)[:, 0]
    return audio, sample_rate


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    if audio.size == 0:
        return audio

    duration_s = audio.shape[0] / src_rate
    dst_len = round(duration_s * dst_rate)
    src_x = np.arange(audio.shape[0], dtype=np.float64) / src_rate
    dst_x = np.arange(dst_len, dtype=np.float64) / dst_rate
    resampled = np.interp(dst_x, src_x, audio.astype(np.float64))
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


def _chunk_audio(audio: np.ndarray, chunk_ms: int) -> list[bytes]:
    chunk_samples = INPUT_SAMPLE_RATE * chunk_ms // 1000
    if chunk_samples <= 0:
        raise ValueError("--chunk-ms must produce at least one sample")
    return [
        audio[start : start + chunk_samples].tobytes()
        for start in range(0, audio.shape[0], chunk_samples)
    ]


def _write_wav(path: Path, pcm16_audio: bytes) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM16_BYTES_PER_SAMPLE)
        wav_file.setframerate(OUTPUT_SAMPLE_RATE)
        wav_file.writeframes(pcm16_audio)


def _normalize_transcript(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _normalized_distance(left: str, right: str) -> float:
    left_norm = _normalize_transcript(left)
    right_norm = _normalize_transcript(right)
    if not left_norm and not right_norm:
        return 0.0
    return _edit_distance(left_norm, right_norm) / max(len(left_norm), len(right_norm))


def _run_stt_command(command_template: str, audio_path: str) -> str:
    command = command_template.format(audio=audio_path)
    completed = subprocess.run(
        shlex.split(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


async def _run_turn(
    index: int,
    stub: audio_pb2_grpc.Qwen3OmniAudioStub,
    input_path: Path,
    args: argparse.Namespace,
) -> TurnResult:
    await asyncio.sleep(args.offset_s + args.stagger_s * index)

    request_id = (
        args.request_ids[index]
        if args.request_ids is not None
        else f"{args.request_prefix}-{index}-{uuid.uuid4().hex[:8]}"
    )
    session_id = (
        args.session_ids[index]
        if args.session_ids is not None
        else f"{args.session_prefix}-{index}-{uuid.uuid4().hex[:8]}"
    )
    audio, sample_rate = _load_wav(input_path)
    audio = _resample_linear(audio, sample_rate, INPUT_SAMPLE_RATE)
    chunks = _chunk_audio(audio, args.chunk_ms)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.talker_max_new_tokens is not None:
        sampling_params.setdefault("custom_params", {})[
            "talker_max_new_tokens"
        ] = args.talker_max_new_tokens
    if args.regex:
        sampling_params["regex"] = args.regex
    custom_params = sampling_params.setdefault("custom_params", {})
    if args.talker_temperature is not None:
        custom_params["talker_temperature"] = args.talker_temperature
    if args.talker_top_k is not None:
        custom_params["talker_top_k"] = args.talker_top_k
    if args.talker_top_p is not None:
        custom_params["talker_top_p"] = args.talker_top_p
    if args.talker_repetition_penalty is not None:
        custom_params["talker_repetition_penalty"] = args.talker_repetition_penalty

    start_time = time.perf_counter()
    eoi_time = None

    async def request_stream():
        nonlocal eoi_time
        yield audio_pb2.AudioTurnRequest(
            config=audio_pb2.AudioTurnConfig(
                request_id=request_id,
                session_id=session_id,
                sampling_params_json=json.dumps(sampling_params),
                system_prompt=args.system_prompt,
                speaker=args.speaker,
            )
        )
        for chunk in chunks:
            if args.realtime_input and args.sleep_before_chunk:
                await asyncio.sleep(
                    len(chunk) / PCM16_BYTES_PER_SAMPLE / INPUT_SAMPLE_RATE
                )
            yield audio_pb2.AudioTurnRequest(pcm16_audio=chunk)
            if args.realtime_input and not args.sleep_before_chunk:
                await asyncio.sleep(
                    len(chunk) / PCM16_BYTES_PER_SAMPLE / INPUT_SAMPLE_RATE
                )
        eoi_time = time.perf_counter()
        yield audio_pb2.AudioTurnRequest(end_of_input=True)

    output_audio = bytearray()
    text_parts: list[str] = []
    full_text = ""
    first_audio_time = None
    last_audio_time = None
    prompt_tokens = 0
    completion_tokens = 0
    error = None
    response_end_time = None
    complete_time = None

    try:
        async for event in stub.AudioTurn(request_stream(), timeout=args.rpc_timeout_s):
            if event.text_delta:
                text_parts.append(event.text_delta)
            elif event.pcm16_audio:
                last_audio_time = time.perf_counter()
                if first_audio_time is None:
                    first_audio_time = last_audio_time
                output_audio.extend(event.pcm16_audio)
            elif event.error.message:
                error = event.error.message
                break
            elif event.complete.turn_rid:
                if complete_time is None:
                    complete_time = time.perf_counter()
                full_text = event.complete.full_text
                prompt_tokens = event.complete.prompt_tokens
                completion_tokens = event.complete.completion_tokens
        response_end_time = time.perf_counter()
    except grpc.aio.AioRpcError as exc:
        response_end_time = time.perf_counter()
        error = f"{exc.code().name}: {exc.details()}"
    finally:
        if args.close_sessions:
            try:
                await stub.CloseAudioSession(
                    audio_pb2.CloseAudioSessionRequest(
                        session_id=session_id, reason="load-test cleanup"
                    ),
                    timeout=30,
                )
            except grpc.aio.AioRpcError as exc:
                if error is None:
                    error = f"close failed: {exc.code().name}: {exc.details()}"

    if response_end_time is None:
        response_end_time = time.perf_counter()
    metric_timestamps = [
        timestamp
        for timestamp in (complete_time, last_audio_time)
        if timestamp is not None
    ]
    metric_end_time = max(metric_timestamps) if metric_timestamps else response_end_time
    audio_bytes = len(output_audio)
    output_audio_s = audio_bytes / PCM16_BYTES_PER_SAMPLE / OUTPUT_SAMPLE_RATE
    elapsed_after_eoi = metric_end_time - eoi_time if eoi_time is not None else None
    first_audio_after_eoi = (
        first_audio_time - eoi_time
        if first_audio_time is not None and eoi_time is not None
        else None
    )
    rtf_after_eoi = (
        elapsed_after_eoi / output_audio_s
        if elapsed_after_eoi is not None and output_audio_s > 0
        else None
    )
    realtime_after_eoi = rtf_after_eoi <= 1.0 if rtf_after_eoi is not None else None
    audio_stream_elapsed = (
        last_audio_time - first_audio_time
        if last_audio_time is not None and first_audio_time is not None
        else None
    )
    audio_stream_rtf_after_first_audio = (
        audio_stream_elapsed / output_audio_s
        if audio_stream_elapsed is not None and output_audio_s > 0
        else None
    )
    realtime_audio_stream_after_first_audio = (
        audio_stream_rtf_after_first_audio <= 1.0
        if audio_stream_rtf_after_first_audio is not None
        else None
    )

    output_path = None
    if args.save_output_dir and output_audio:
        save_dir = Path(args.save_output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(save_dir / f"turn_{index:02d}.wav")
        _write_wav(Path(output_path), bytes(output_audio))

    stt_text = None
    stt_reference_text = full_text or "".join(text_parts)
    stt_normalized_distance = None
    stt_match = None
    if args.stt_command:
        if output_path is None:
            raise ValueError("--stt-command requires --save-output-dir")
        stt_text = _run_stt_command(args.stt_command, output_path)
        stt_normalized_distance = _normalized_distance(stt_text, stt_reference_text)
        stt_match = stt_normalized_distance <= args.stt_max_normalized_distance
        if not stt_match:
            error = (
                f"STT mismatch distance={stt_normalized_distance:.3f} "
                f"threshold={args.stt_max_normalized_distance:.3f}"
            )

    return TurnResult(
        index=index,
        request_id=request_id,
        session_id=session_id,
        ok=error is None and audio_bytes > 0,
        error=error,
        text="".join(text_parts),
        full_text=full_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        audio_bytes=audio_bytes,
        output_audio_s=output_audio_s,
        first_audio_after_eoi_s=first_audio_after_eoi,
        elapsed_after_eoi_s=elapsed_after_eoi,
        elapsed_total_s=metric_end_time - start_time,
        rtf_after_eoi=rtf_after_eoi,
        realtime_after_eoi=realtime_after_eoi,
        audio_stream_rtf_after_first_audio=audio_stream_rtf_after_first_audio,
        realtime_audio_stream_after_first_audio=realtime_audio_stream_after_first_audio,
        output_path=output_path,
        stt_text=stt_text,
        stt_reference_text=stt_reference_text,
        stt_normalized_distance=stt_normalized_distance,
        stt_match=stt_match,
    )


def _aggregate(results: list[TurnResult]) -> dict:
    ok_results = [result for result in results if result.ok]
    rtfs = [
        result.rtf_after_eoi
        for result in ok_results
        if result.rtf_after_eoi is not None
    ]
    first_audio = [
        result.first_audio_after_eoi_s
        for result in ok_results
        if result.first_audio_after_eoi_s is not None
    ]
    stream_rtfs = [
        result.audio_stream_rtf_after_first_audio
        for result in ok_results
        if result.audio_stream_rtf_after_first_audio is not None
    ]
    elapsed = [
        result.elapsed_after_eoi_s
        for result in ok_results
        if result.elapsed_after_eoi_s is not None
    ]
    stt_results = [
        result.stt_match for result in results if result.stt_match is not None
    ]
    realtime_results = [
        result.realtime_after_eoi
        for result in ok_results
        if result.realtime_after_eoi is not None
    ]
    realtime_stream_results = [
        result.realtime_audio_stream_after_first_audio
        for result in ok_results
        if result.realtime_audio_stream_after_first_audio is not None
    ]
    return {
        "ok": len(ok_results),
        "failed": len(results) - len(ok_results),
        "max_rtf_after_eoi": max(rtfs) if rtfs else None,
        "mean_rtf_after_eoi": sum(rtfs) / len(rtfs) if rtfs else None,
        "max_first_audio_after_eoi_s": max(first_audio) if first_audio else None,
        "max_audio_stream_rtf_after_first_audio": max(stream_rtfs)
        if stream_rtfs
        else None,
        "mean_audio_stream_rtf_after_first_audio": (
            sum(stream_rtfs) / len(stream_rtfs) if stream_rtfs else None
        ),
        "max_elapsed_after_eoi_s": max(elapsed) if elapsed else None,
        "total_audio_bytes": sum(result.audio_bytes for result in ok_results),
        "realtime_after_eoi": all(realtime_results) if realtime_results else None,
        "realtime_audio_stream_after_first_audio": (
            all(realtime_stream_results) if realtime_stream_results else None
        ),
        "stt_checked": len(stt_results),
        "stt_matched": sum(1 for matched in stt_results if matched),
    }


async def _main_async(args: argparse.Namespace) -> None:
    input_path = Path(args.input_wav)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    async with grpc.aio.insecure_channel(f"{args.host}:{args.port}") as channel:
        stub = audio_pb2_grpc.Qwen3OmniAudioStub(channel)
        results = await asyncio.gather(
            *[
                _run_turn(index, stub, input_path, args)
                for index in range(args.concurrency)
            ]
        )

    payload = {
        "config": {
            "host": args.host,
            "port": args.port,
            "concurrency": args.concurrency,
            "input_wav": str(input_path),
            "chunk_ms": args.chunk_ms,
            "realtime_input": args.realtime_input,
            "stagger_s": args.stagger_s,
            "max_new_tokens": args.max_new_tokens,
            "talker_max_new_tokens": args.talker_max_new_tokens,
            "talker_temperature": args.talker_temperature,
            "talker_top_k": args.talker_top_k,
            "talker_top_p": args.talker_top_p,
            "talker_repetition_penalty": args.talker_repetition_penalty,
            "speaker": args.speaker,
            "system_prompt_set": bool(args.system_prompt),
            "stt_command_set": bool(args.stt_command),
            "stt_max_normalized_distance": args.stt_max_normalized_distance,
        },
        "aggregate": _aggregate(results),
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["aggregate"]["failed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--input-wav", required=True)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--offset-s", type=float, default=0.0)
    parser.add_argument("--stagger-s", type=float, default=0.0)
    parser.add_argument("--realtime-input", action="store_true")
    parser.add_argument(
        "--sleep-before-chunk",
        action="store_true",
        help=(
            "With --realtime-input, wait for each chunk's duration before "
            "sending it. Use this for paced larger-chunk cadence sweeps, "
            "e.g. sending a 5s chunk at t=5s instead of t=0."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--talker-max-new-tokens", type=int)
    parser.add_argument("--talker-temperature", type=float)
    parser.add_argument("--talker-top-k", type=int)
    parser.add_argument("--talker-top-p", type=float)
    parser.add_argument("--talker-repetition-penalty", type=float)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--regex")
    parser.add_argument("--speaker", default="chelsie")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--system-prompt-file")
    parser.add_argument(
        "--stt-command",
        help="Command that prints the STT transcript for {audio}; use Parakeet here.",
    )
    # 0.30 matches qwen3_omni_voice_validate.STT_MAX_DISTANCE — empirical RNG
    # noise floor on talker sampling. Tighten with --stt-strict if needed.
    parser.add_argument("--stt-max-normalized-distance", type=float, default=0.30)
    parser.add_argument("--request-prefix", default="qwen3-omni-load")
    parser.add_argument("--session-prefix", default="qwen3-omni-load")
    parser.add_argument("--request-ids-file")
    parser.add_argument("--session-ids-file")
    parser.add_argument("--rpc-timeout-s", type=float, default=300.0)
    parser.add_argument("--save-output-dir")
    parser.add_argument(
        "--no-close-sessions", dest="close_sessions", action="store_false"
    )
    parser.set_defaults(close_sessions=True)
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.chunk_ms <= 0:
        raise ValueError("--chunk-ms must be positive")
    if args.stagger_s < 0:
        raise ValueError("--stagger-s must be non-negative")
    if args.talker_max_new_tokens is None:
        raise ValueError(
            "Qwen3-Omni load tests require --talker-max-new-tokens; "
            "Thinker --max-new-tokens does not bound Talker continuation"
        )
    if args.talker_max_new_tokens <= 0:
        raise ValueError("--talker-max-new-tokens must be positive")
    if args.system_prompt_file:
        if args.system_prompt:
            raise ValueError("Use either --system-prompt or --system-prompt-file")
        args.system_prompt = Path(args.system_prompt_file).read_text()
    args.request_ids = None
    if args.request_ids_file:
        args.request_ids = [
            line.strip()
            for line in Path(args.request_ids_file).read_text().splitlines()
            if line.strip()
        ]
        if len(args.request_ids) != args.concurrency:
            raise ValueError("--request-ids-file must contain exactly --concurrency IDs")
    args.session_ids = None
    if args.session_ids_file:
        args.session_ids = [
            line.strip()
            for line in Path(args.session_ids_file).read_text().splitlines()
            if line.strip()
        ]
        if len(args.session_ids) != args.concurrency:
            raise ValueError("--session-ids-file must contain exactly --concurrency IDs")
    if not 0 <= args.stt_max_normalized_distance <= 1:
        raise ValueError("--stt-max-normalized-distance must be in [0, 1]")
    if args.stt_command:
        if not args.system_prompt:
            raise ValueError("--stt-command requires Beginnings --system-prompt")
        if args.speaker != "chelsie":
            raise ValueError("--stt-command correctness gate requires --speaker chelsie")

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
