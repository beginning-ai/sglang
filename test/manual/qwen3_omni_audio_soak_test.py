#!/usr/bin/env python3
"""Long-lived Qwen3-Omni realtime audio soak client for the gRPC API."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib import request as urllib_request

import grpc
import numpy as np

from qwen3_omni_audio_load_test import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    PCM16_BYTES_PER_SAMPLE,
    _chunk_audio,
    _load_wav,
    _resample_linear,
    _write_wav,
)
from sglang.srt.grpc import qwen3_omni_audio_pb2 as audio_pb2
from sglang.srt.grpc import qwen3_omni_audio_pb2_grpc as audio_pb2_grpc

RUNTIME_SNAPSHOT_RE = re.compile(
    r"Qwen3-Omni runtime snapshot: event=(?P<event>\S+) "
    r"stats=(?P<stats>\{.*\})"
)


@dataclass(frozen=True)
class SoakFixture:
    name: str
    path: Path
    edge_case: bool = False


@dataclass
class SoakTurnResult:
    session_index: int
    session_generation: int
    turn_index: int
    fixture: str
    edge_case: bool
    request_id: str
    turn_rid: str | None
    session_id: str
    parent_rid: str | None
    request_start_wall_s: float
    eoi_wall_s: float | None
    first_audio_wall_s: float | None
    last_audio_wall_s: float | None
    complete_wall_s: float | None
    response_end_wall_s: float
    ok: bool
    error: str | None
    text: str
    full_text: str
    prompt_tokens: int
    completion_tokens: int
    input_audio_s: float
    input_chunk_count: int
    audio_bytes: int
    output_audio_s: float
    first_audio_after_eoi_s: float | None
    last_audio_after_eoi_s: float | None
    complete_after_eoi_s: float | None
    response_end_after_eoi_s: float | None
    audio_stream_elapsed_s: float | None
    complete_tail_after_last_audio_s: float | None
    elapsed_after_eoi_s: float | None
    elapsed_total_s: float
    rtf_after_eoi: float | None
    realtime_after_eoi: bool | None
    audio_stream_rtf_after_first_audio: float | None
    realtime_audio_stream_after_first_audio: bool | None
    output_path: str | None


@dataclass
class SoakSafeTurn:
    turn_rid: str
    fixture: SoakFixture
    assistant_text: str


@dataclass
class SoakRollbackResult:
    session_index: int
    session_generation: int
    rollback_index: int
    request_id: str
    session_id: str
    previous_parent_rid: str | None
    latest_turn_rid: str | None
    prefix_turn_count: int
    prompt_tokens: int
    elapsed_s: float
    ok: bool
    error: str | None


def _parse_fixture(raw: str, *, edge_case: bool = False) -> SoakFixture:
    name, sep, path = raw.partition(":")
    if not sep or not name or not path:
        raise ValueError("--fixture entries must be name:/path/to.wav")
    fixture = SoakFixture(name=name, path=Path(path), edge_case=edge_case)
    if not fixture.path.is_file():
        raise FileNotFoundError(fixture.path)
    return fixture


def _write_input_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM16_BYTES_PER_SAMPLE)
        wav_file.setframerate(INPUT_SAMPLE_RATE)
        wav_file.writeframes(audio.astype(np.int16).tobytes())


def _build_edge_fixtures(fixtures: list[SoakFixture], output_dir: Path) -> list[SoakFixture]:
    base_audio, sample_rate = _load_wav(fixtures[0].path)
    base_audio = _resample_linear(base_audio, sample_rate, INPUT_SAMPLE_RATE)
    edge_dir = output_dir / "_edge_inputs"

    short = base_audio[: int(0.25 * INPUT_SAMPLE_RATE)]
    short_path = edge_dir / "short_250ms.wav"
    _write_input_wav(short_path, short)

    rng = np.random.default_rng(0)
    noise = rng.integers(-128, 128, size=INPUT_SAMPLE_RATE, dtype=np.int16)
    noise_path = edge_dir / "low_noise_1s.wav"
    _write_input_wav(noise_path, noise)

    truncated = base_audio[: int(0.70 * INPUT_SAMPLE_RATE)]
    truncated_path = edge_dir / "truncated_700ms.wav"
    _write_input_wav(truncated_path, truncated)

    return [
        SoakFixture("edge_short", short_path, edge_case=True),
        SoakFixture("edge_noise", noise_path, edge_case=True),
        SoakFixture("edge_truncated", truncated_path, edge_case=True),
    ]


def _sampling_params(args: argparse.Namespace) -> dict:
    params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "custom_params": {
            "talker_max_new_tokens": args.talker_max_new_tokens,
        },
    }
    return params


def _internal_state_url(args: argparse.Namespace) -> str:
    if args.internal_state_url:
        return args.internal_state_url
    return f"http://{args.host}:{args.port + 1}/internal_state"


def _sample_internal_state(url: str, timeout_s: float) -> dict:
    with urllib_request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read())


def _sum_field(states: list[dict], path: tuple[str, ...]) -> int:
    total = 0
    for state in states:
        node = state
        for key in path:
            node = node[key]
        total += int(node)
    return total


def _kv_cache_summary(states: list[dict]) -> dict:
    audio_states = [state["qwen3_omni_audio"] for state in states]
    components = sorted(
        {
            component
            for state in audio_states
            for component in state["component_kv"].keys()
        }
    )
    summary = {
        "main": {
            key: _sum_field(audio_states, ("tree_cache", key))
            for key in (
                "used",
                "available",
                "size",
                "evictable",
                "protected",
                "full_evictable",
                "full_protected",
            )
        },
        "component_kv": {
            component: {
                key: sum(
                    int(state["component_kv"].get(component, {}).get(key, 0))
                    for state in audio_states
                )
                for key in ("used", "free", "size")
            }
            for component in components
        },
        "streaming_session": {
            key: _sum_field(audio_states, ("streaming_session", key))
            for key in ("sessions", "held_tokens", "held_req_slots")
        },
        "audio_sessions": sum(int(state["audio_sessions"]) for state in audio_states),
        "active_audio_turns": sum(
            int(state["active_audio_turns"]) for state in audio_states
        ),
    }
    main_size = summary["main"]["size"]
    summary["main"]["used_fraction"] = (
        summary["main"]["used"] / main_size if main_size else 0.0
    )
    for component_stats in summary["component_kv"].values():
        size = component_stats["size"]
        component_stats["used_fraction"] = (
            component_stats["used"] / size if size else 0.0
        )
    return summary


def _kv_cache_summary_from_runtime_stats(stats: dict) -> dict:
    summary = {
        "main": {
            key: int(stats["tree_cache"][key])
            for key in (
                "used",
                "available",
                "size",
                "evictable",
                "protected",
                "full_evictable",
                "full_protected",
            )
        },
        "component_kv": {
            component: {
                "used": int(component_stats["used"]),
                "free": int(component_stats["free"]),
                "size": int(component_stats["size"]),
            }
            for component, component_stats in stats["component_kv"].items()
        },
        "streaming_session": {
            key: int(stats["streaming_session"][key])
            for key in ("sessions", "held_tokens", "held_req_slots")
        },
        "audio_sessions": int(stats["audio_sessions"]),
        "active_audio_turns": int(stats["active_audio_turns"]),
    }
    main_size = summary["main"]["size"]
    summary["main"]["used_fraction"] = (
        summary["main"]["used"] / main_size if main_size else 0.0
    )
    for component_stats in summary["component_kv"].values():
        size = component_stats["size"]
        component_stats["used_fraction"] = (
            component_stats["used"] / size if size else 0.0
        )
    return summary


def _memory_usage_summary(states: list[dict]) -> dict:
    keys = ("weight", "kvcache", "graph")
    return {
        key: sum(float(state["memory_usage"][key]) for state in states)
        for key in keys
    }


async def _record_internal_state_telemetry(
    args: argparse.Namespace,
    stop_event: asyncio.Event,
    samples: list[dict],
) -> None:
    output_path = Path(args.telemetry_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = _internal_state_url(args)
    started_at = time.time()
    with output_path.open("w") as out:
        while True:
            now = time.time()
            payload = await asyncio.to_thread(
                _sample_internal_state,
                url,
                args.telemetry_timeout_s,
            )
            states = payload["states"]
            sample = {
                "time": now,
                "elapsed_s": now - started_at,
                "internal_state_url": url,
                "memory_usage": _memory_usage_summary(states),
                "kv_cache": _kv_cache_summary(states),
            }
            samples.append(sample)
            out.write(json.dumps(sample, sort_keys=True) + "\n")
            out.flush()
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=args.telemetry_interval_s,
                )
                break
            except asyncio.TimeoutError:
                continue


def _read_runtime_snapshot_samples(
    log_path: Path,
    offset: int,
    started_at: float,
) -> tuple[int, list[dict]]:
    samples = []
    with log_path.open(errors="ignore") as log_file:
        log_file.seek(offset)
        for line in log_file:
            match = RUNTIME_SNAPSHOT_RE.search(line)
            if match is None:
                continue
            now = time.time()
            stats = json.loads(match.group("stats"))
            samples.append(
                {
                    "time": now,
                    "elapsed_s": now - started_at,
                    "runtime_event": match.group("event"),
                    "runtime_snapshot_log": str(log_path),
                    "kv_cache": _kv_cache_summary_from_runtime_stats(stats),
                }
            )
        return log_file.tell(), samples


async def _record_runtime_snapshot_telemetry(
    args: argparse.Namespace,
    stop_event: asyncio.Event,
    samples: list[dict],
) -> None:
    output_path = Path(args.telemetry_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.runtime_snapshot_log)
    started_at = time.time()
    with log_path.open(errors="ignore") as log_file:
        offset = log_file.seek(0, 2)
    with output_path.open("w") as out:
        while True:
            offset, new_samples = await asyncio.to_thread(
                _read_runtime_snapshot_samples,
                log_path,
                offset,
                started_at,
            )
            for sample in new_samples:
                samples.append(sample)
                out.write(json.dumps(sample, sort_keys=True) + "\n")
            if new_samples:
                out.flush()
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=args.telemetry_interval_s,
                )
                break
            except asyncio.TimeoutError:
                continue


def _telemetry_summary(samples: list[dict]) -> dict | None:
    if not samples:
        return None
    components = sorted(
        {
            component
            for sample in samples
            for component in sample["kv_cache"]["component_kv"]
        }
    )
    summary = {
        "samples": len(samples),
        "elapsed_s": samples[-1]["elapsed_s"],
        "main": {
            "max_used": max(sample["kv_cache"]["main"]["used"] for sample in samples),
            "max_size": max(sample["kv_cache"]["main"]["size"] for sample in samples),
            "max_used_fraction": max(
                sample["kv_cache"]["main"]["used_fraction"] for sample in samples
            ),
        },
        "component_kv": {
            component: {
                "max_used": max(
                    sample["kv_cache"]["component_kv"].get(component, {}).get(
                        "used", 0
                    )
                    for sample in samples
                ),
                "max_size": max(
                    sample["kv_cache"]["component_kv"].get(component, {}).get(
                        "size", 0
                    )
                    for sample in samples
                ),
                "max_used_fraction": max(
                    sample["kv_cache"]["component_kv"].get(component, {}).get(
                        "used_fraction", 0.0
                    )
                    for sample in samples
                ),
            }
            for component in components
        },
        "streaming_session": {
            "max_sessions": max(
                sample["kv_cache"]["streaming_session"]["sessions"]
                for sample in samples
            ),
            "max_held_tokens": max(
                sample["kv_cache"]["streaming_session"]["held_tokens"]
                for sample in samples
            ),
            "max_held_req_slots": max(
                sample["kv_cache"]["streaming_session"]["held_req_slots"]
                for sample in samples
            ),
        },
    }
    memory_samples = [sample for sample in samples if "memory_usage" in sample]
    if memory_samples:
        summary["memory_usage"] = {
            key: max(sample["memory_usage"][key] for sample in memory_samples)
            for key in ("weight", "kvcache", "graph")
        }
    return summary


async def _run_turn(
    *,
    session_index: int,
    session_generation: int,
    turn_index: int,
    session_id: str,
    parent_rid: str | None,
    fixture: SoakFixture,
    stub: audio_pb2_grpc.Qwen3OmniAudioStub,
    args: argparse.Namespace,
    save_output: bool,
) -> SoakTurnResult:
    request_id = (
        f"{args.request_prefix}-s{session_index:02d}-g{session_generation:03d}"
        f"-t{turn_index:04d}-{uuid.uuid4().hex[:8]}"
    )

    audio, sample_rate = _load_wav(fixture.path)
    audio = _resample_linear(audio, sample_rate, INPUT_SAMPLE_RATE)
    chunks = _chunk_audio(audio, args.chunk_ms)

    start_time = time.perf_counter()
    request_start_wall = time.time()
    eoi_time = None
    eoi_wall = None

    async def request_stream():
        nonlocal eoi_time, eoi_wall
        yield audio_pb2.AudioTurnRequest(
            config=audio_pb2.AudioTurnConfig(
                request_id=request_id,
                session_id=session_id,
                parent_rid=parent_rid or "",
                sampling_params_json=json.dumps(_sampling_params(args)),
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
        eoi_wall = time.time()
        yield audio_pb2.AudioTurnRequest(end_of_input=True)

    output_audio = bytearray()
    text_parts: list[str] = []
    full_text = ""
    turn_rid = None
    first_audio_time = None
    first_audio_wall = None
    last_audio_time = None
    last_audio_wall = None
    complete_time = None
    complete_wall = None
    response_end_time = None
    response_end_wall = None
    prompt_tokens = 0
    completion_tokens = 0
    error = None

    try:
        timeout = None if args.rpc_timeout_s <= 0 else args.rpc_timeout_s
        async for event in stub.AudioTurn(request_stream(), timeout=timeout):
            if event.text_delta:
                text_parts.append(event.text_delta)
            elif event.pcm16_audio:
                last_audio_time = time.perf_counter()
                last_audio_wall = time.time()
                if first_audio_time is None:
                    first_audio_time = last_audio_time
                    first_audio_wall = last_audio_wall
                output_audio.extend(event.pcm16_audio)
            elif event.error.message:
                error = event.error.message
                break
            elif event.complete.turn_rid:
                complete_time = time.perf_counter()
                complete_wall = time.time()
                turn_rid = event.complete.turn_rid
                full_text = event.complete.full_text
                prompt_tokens = event.complete.prompt_tokens
                completion_tokens = event.complete.completion_tokens
        response_end_time = time.perf_counter()
        response_end_wall = time.time()
    except grpc.aio.AioRpcError as exc:
        response_end_time = time.perf_counter()
        response_end_wall = time.time()
        error = f"{exc.code().name}: {exc.details()}"

    if response_end_time is None:
        response_end_time = time.perf_counter()
    if response_end_wall is None:
        response_end_wall = time.time()
    metric_timestamps = [
        timestamp
        for timestamp in (complete_time, last_audio_time)
        if timestamp is not None
    ]
    metric_end_time = max(metric_timestamps) if metric_timestamps else response_end_time
    audio_bytes = len(output_audio)
    input_audio_s = (
        sum(len(chunk) for chunk in chunks) / PCM16_BYTES_PER_SAMPLE / INPUT_SAMPLE_RATE
    )
    output_audio_s = audio_bytes / PCM16_BYTES_PER_SAMPLE / OUTPUT_SAMPLE_RATE
    elapsed_after_eoi = metric_end_time - eoi_time if eoi_time is not None else None
    first_audio_after_eoi = (
        first_audio_time - eoi_time
        if first_audio_time is not None and eoi_time is not None
        else None
    )
    last_audio_after_eoi = (
        last_audio_time - eoi_time
        if last_audio_time is not None and eoi_time is not None
        else None
    )
    complete_after_eoi = (
        complete_time - eoi_time
        if complete_time is not None and eoi_time is not None
        else None
    )
    response_end_after_eoi = (
        response_end_time - eoi_time if eoi_time is not None else None
    )
    complete_tail_after_last_audio = (
        complete_time - last_audio_time
        if complete_time is not None and last_audio_time is not None
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
    if args.save_output_dir and save_output and output_audio:
        save_dir = (
            Path(args.save_output_dir)
            / f"session_{session_index:02d}"
            / f"generation_{session_generation:03d}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(save_dir / f"turn_{turn_index:04d}_{fixture.name}.wav")
        _write_wav(Path(output_path), bytes(output_audio))

    return SoakTurnResult(
        session_index=session_index,
        session_generation=session_generation,
        turn_index=turn_index,
        fixture=fixture.name,
        edge_case=fixture.edge_case,
        request_id=request_id,
        turn_rid=turn_rid,
        session_id=session_id,
        parent_rid=parent_rid,
        request_start_wall_s=request_start_wall,
        eoi_wall_s=eoi_wall,
        first_audio_wall_s=first_audio_wall,
        last_audio_wall_s=last_audio_wall,
        complete_wall_s=complete_wall,
        response_end_wall_s=response_end_wall,
        ok=error is None and audio_bytes > 0,
        error=error,
        text="".join(text_parts),
        full_text=full_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_audio_s=input_audio_s,
        input_chunk_count=len(chunks),
        audio_bytes=audio_bytes,
        output_audio_s=output_audio_s,
        first_audio_after_eoi_s=first_audio_after_eoi,
        last_audio_after_eoi_s=last_audio_after_eoi,
        complete_after_eoi_s=complete_after_eoi,
        response_end_after_eoi_s=response_end_after_eoi,
        audio_stream_elapsed_s=audio_stream_elapsed,
        complete_tail_after_last_audio_s=complete_tail_after_last_audio,
        elapsed_after_eoi_s=elapsed_after_eoi,
        elapsed_total_s=metric_end_time - start_time,
        rtf_after_eoi=rtf_after_eoi,
        realtime_after_eoi=realtime_after_eoi,
        audio_stream_rtf_after_first_audio=audio_stream_rtf_after_first_audio,
        realtime_audio_stream_after_first_audio=realtime_audio_stream_after_first_audio,
        output_path=output_path,
    )


async def _rebuild_session(
    *,
    session_index: int,
    session_generation: int,
    rollback_index: int,
    session_id: str,
    safe_turns: list[SoakSafeTurn],
    previous_parent_rid: str | None,
    stub: audio_pb2_grpc.Qwen3OmniAudioStub,
    args: argparse.Namespace,
) -> SoakRollbackResult:
    request_id = (
        f"{args.request_prefix}-s{session_index:02d}-g{session_generation:03d}"
        f"-rb{rollback_index:04d}-{uuid.uuid4().hex[:8]}"
    )
    start_time = time.perf_counter()

    async def request_stream():
        yield audio_pb2.RebuildAudioSessionRequest(
            config=audio_pb2.RebuildAudioSessionConfig(
                request_id=request_id,
                session_id=session_id,
                system_prompt=args.system_prompt,
                speaker=args.speaker,
                sampling_params_json=json.dumps(_sampling_params(args)),
            )
        )
        for turn_index, turn in enumerate(safe_turns):
            audio, sample_rate = _load_wav(turn.fixture.path)
            audio = _resample_linear(audio, sample_rate, INPUT_SAMPLE_RATE)
            for chunk in _chunk_audio(audio, args.chunk_ms):
                yield audio_pb2.RebuildAudioSessionRequest(
                    user_audio=audio_pb2.RebuildAudioSessionUserAudio(
                        turn_index=turn_index,
                        pcm16_audio=chunk,
                    )
                )
            yield audio_pb2.RebuildAudioSessionRequest(
                assistant_text=audio_pb2.RebuildAudioSessionAssistantText(
                    turn_index=turn_index,
                    turn_rid=turn.turn_rid,
                    text=turn.assistant_text,
                )
            )

    try:
        response = await stub.RebuildAudioSession(
            request_stream(), timeout=args.rollback_rpc_timeout_s
        )
        latest_turn_rid = response.latest_turn_rid or None
        expected_latest = safe_turns[-1].turn_rid if safe_turns else None
        if latest_turn_rid != expected_latest:
            raise RuntimeError(
                f"rollback latest_turn_rid {latest_turn_rid} != {expected_latest}"
            )
        return SoakRollbackResult(
            session_index=session_index,
            session_generation=session_generation,
            rollback_index=rollback_index,
            request_id=request_id,
            session_id=session_id,
            previous_parent_rid=previous_parent_rid,
            latest_turn_rid=latest_turn_rid,
            prefix_turn_count=response.prefix_turn_count,
            prompt_tokens=response.prompt_tokens,
            elapsed_s=time.perf_counter() - start_time,
            ok=response.success,
            error=None if response.success else response.message,
        )
    except (grpc.aio.AioRpcError, RuntimeError) as exc:
        return SoakRollbackResult(
            session_index=session_index,
            session_generation=session_generation,
            rollback_index=rollback_index,
            request_id=request_id,
            session_id=session_id,
            previous_parent_rid=previous_parent_rid,
            latest_turn_rid=None,
            prefix_turn_count=len(safe_turns),
            prompt_tokens=0,
            elapsed_s=time.perf_counter() - start_time,
            ok=False,
            error=str(exc),
        )


async def _close_session(
    stub: audio_pb2_grpc.Qwen3OmniAudioStub,
    session_id: str,
    args: argparse.Namespace,
) -> dict:
    try:
        timeout = None if args.close_rpc_timeout_s <= 0 else args.close_rpc_timeout_s
        response = await stub.CloseAudioSession(
            audio_pb2.CloseAudioSessionRequest(
                session_id=session_id, reason="soak-test cleanup"
            ),
            timeout=timeout,
        )
        return {
            "session_id": session_id,
            "success": response.success,
            "turn_count": response.turn_count,
            "active_request_count": response.active_request_count,
            "error": None,
        }
    except grpc.aio.AioRpcError as exc:
        return {
            "session_id": session_id,
            "success": False,
            "turn_count": 0,
            "active_request_count": 0,
            "error": f"{exc.code().name}: {exc.details()}",
        }


async def _run_session_generation(
    session_index: int,
    session_generation: int,
    session_id: str,
    deadline: float,
    stub: audio_pb2_grpc.Qwen3OmniAudioStub,
    fixtures: list[SoakFixture],
    edge_fixtures: list[SoakFixture],
    args: argparse.Namespace,
) -> tuple[list[SoakTurnResult], list[SoakRollbackResult]]:
    rng = random.Random(args.seed + session_index * 1009 + session_generation)
    parent_rid = None
    turn_index = 0
    rollback_index = 0
    results: list[SoakTurnResult] = []
    rollbacks: list[SoakRollbackResult] = []
    safe_turns: list[SoakSafeTurn] = []

    while time.monotonic() < deadline:
        edge_roll = rng.random()
        if edge_fixtures and edge_roll < args.edge_case_rate * len(edge_fixtures):
            fixture = edge_fixtures[int(edge_roll / args.edge_case_rate)]
        else:
            fixture = rng.choice(fixtures)
        save_output = (not fixture.edge_case) and rng.random() < args.save_sample_rate
        result = await _run_turn(
            session_index=session_index,
            session_generation=session_generation,
            turn_index=turn_index,
            session_id=session_id,
            parent_rid=parent_rid,
            fixture=fixture,
            stub=stub,
            args=args,
            save_output=save_output,
        )
        results.append(result)
        if result.ok:
            if result.turn_rid is None:
                raise RuntimeError(
                    f"completed audio turn {result.request_id} did not return turn_rid"
                )
            safe_turns.append(
                SoakSafeTurn(
                    turn_rid=result.turn_rid,
                    fixture=fixture,
                    assistant_text=result.full_text or result.text,
                )
            )
            parent_rid = result.turn_rid
            if (
                args.rollback_rate > 0
                and len(safe_turns) >= args.rollback_min_completed_turns
                and rng.random() < args.rollback_rate
            ):
                min_keep = max(0, len(safe_turns) - args.rollback_max_rewind_turns)
                keep_count = rng.randint(min_keep, len(safe_turns) - 1)
                rebuilt_turns = safe_turns[:keep_count]
                rollback = await _rebuild_session(
                    session_index=session_index,
                    session_generation=session_generation,
                    rollback_index=rollback_index,
                    session_id=session_id,
                    safe_turns=rebuilt_turns,
                    previous_parent_rid=parent_rid,
                    stub=stub,
                    args=args,
                )
                rollbacks.append(rollback)
                rollback_index += 1
                if not rollback.ok:
                    break
                safe_turns = rebuilt_turns
                parent_rid = rollback.latest_turn_rid
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(rng.uniform(args.turn_min_s, args.turn_max_s))
        turn_index += 1
    return results, rollbacks


async def _run_session_slot(
    session_index: int,
    stub: audio_pb2_grpc.Qwen3OmniAudioStub,
    fixtures: list[SoakFixture],
    edge_fixtures: list[SoakFixture],
    args: argparse.Namespace,
) -> tuple[list[SoakTurnResult], list[SoakRollbackResult], list[dict]]:
    rng = random.Random(args.seed + session_index)
    soak_deadline = time.monotonic() + args.duration_s
    if args.initial_jitter_s > 0:
        await asyncio.sleep(rng.uniform(0, args.initial_jitter_s))

    session_generation = 0
    results: list[SoakTurnResult] = []
    rollbacks: list[SoakRollbackResult] = []
    closes: list[dict] = []

    while time.monotonic() < soak_deadline:
        session_id = (
            f"{args.session_prefix}-s{session_index:02d}-g{session_generation:03d}"
        )
        generation_deadline = min(
            soak_deadline,
            time.monotonic() + args.session_duration_s,
        )
        session_results, session_rollbacks = await _run_session_generation(
            session_index,
            session_generation,
            session_id,
            generation_deadline,
            stub,
            fixtures,
            edge_fixtures,
            args,
        )
        results.extend(session_results)
        rollbacks.extend(session_rollbacks)
        closes.append(await _close_session(stub, session_id, args))
        if not closes[-1]["success"]:
            break
        session_generation += 1

    return results, rollbacks, closes


def _aggregate(
    results: list[SoakTurnResult],
    closes: list[dict],
    rollbacks: list[SoakRollbackResult],
) -> dict:
    ok = [result for result in results if result.ok]
    non_edge = [result for result in ok if not result.edge_case]
    rtfs = [result.rtf_after_eoi for result in ok if result.rtf_after_eoi is not None]
    first_audio = [
        result.first_audio_after_eoi_s
        for result in ok
        if result.first_audio_after_eoi_s is not None
    ]
    stream_rtfs = [
        result.audio_stream_rtf_after_first_audio
        for result in ok
        if result.audio_stream_rtf_after_first_audio is not None
    ]
    complete_tails = [
        result.complete_tail_after_last_audio_s
        for result in ok
        if result.complete_tail_after_last_audio_s is not None
    ]
    rollback_elapsed = [rollback.elapsed_s for rollback in rollbacks]
    saved_non_edge = [
        result for result in non_edge if result.output_path is not None
    ]
    return {
        "turns": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "non_edge_turns": len(non_edge),
        "edge_turns": len(ok) - len(non_edge),
        "realtime": sum(1 for result in ok if result.realtime_after_eoi is True),
        "non_realtime": sum(
            1 for result in ok if result.realtime_after_eoi is not True
        ),
        "mean_rtf_after_eoi": sum(rtfs) / len(rtfs) if rtfs else None,
        "max_rtf_after_eoi": max(rtfs) if rtfs else None,
        "max_first_audio_after_eoi_s": max(first_audio) if first_audio else None,
        "realtime_audio_stream": sum(
            1
            for result in ok
            if result.realtime_audio_stream_after_first_audio is True
        ),
        "non_realtime_audio_stream": sum(
            1
            for result in ok
            if result.realtime_audio_stream_after_first_audio is not True
        ),
        "mean_audio_stream_rtf_after_first_audio": (
            sum(stream_rtfs) / len(stream_rtfs) if stream_rtfs else None
        ),
        "max_audio_stream_rtf_after_first_audio": max(stream_rtfs)
        if stream_rtfs
        else None,
        "mean_complete_tail_after_last_audio_s": (
            sum(complete_tails) / len(complete_tails) if complete_tails else None
        ),
        "max_complete_tail_after_last_audio_s": (
            max(complete_tails) if complete_tails else None
        ),
        "saved_non_edge_outputs": len(saved_non_edge),
        "saved_non_edge_fraction": (
            len(saved_non_edge) / len(non_edge) if non_edge else 0.0
        ),
        "close_failures": sum(1 for close in closes if not close["success"]),
        "rollback_attempts": len(rollbacks),
        "rollback_failures": sum(1 for rollback in rollbacks if not rollback.ok),
        "max_rollback_elapsed_s": max(rollback_elapsed) if rollback_elapsed else None,
    }


async def _main_async(args: argparse.Namespace) -> None:
    fixtures = [_parse_fixture(raw) for raw in args.fixture]
    edge_fixtures = (
        _build_edge_fixtures(fixtures, Path(args.save_output_dir))
        if args.include_edge_cases
        else []
    )

    telemetry_samples: list[dict] = []
    telemetry_stop = asyncio.Event()
    telemetry_task = None
    if args.telemetry_jsonl:
        recorder = (
            _record_runtime_snapshot_telemetry
            if args.runtime_snapshot_log
            else _record_internal_state_telemetry
        )
        telemetry_task = asyncio.create_task(
            recorder(args, telemetry_stop, telemetry_samples)
        )
    try:
        async with grpc.aio.insecure_channel(f"{args.host}:{args.port}") as channel:
            stub = audio_pb2_grpc.Qwen3OmniAudioStub(channel)
            per_session_payloads = await asyncio.gather(
                *[
                    _run_session_slot(
                        session_index, stub, fixtures, edge_fixtures, args
                    )
                    for session_index in range(args.sessions)
                ]
            )
    finally:
        if telemetry_task is not None:
            telemetry_stop.set()
            await telemetry_task

    results = [
        result
        for session_results, _, _ in per_session_payloads
        for result in session_results
    ]
    rollbacks = [
        rollback
        for _, session_rollbacks, _ in per_session_payloads
        for rollback in session_rollbacks
    ]
    closes = [
        close
        for _, _, session_closes in per_session_payloads
        for close in session_closes
    ]
    payload = {
        "config": {
            "host": args.host,
            "port": args.port,
            "sessions": args.sessions,
            "duration_s": args.duration_s,
            "session_duration_s": args.session_duration_s,
            "turn_min_s": args.turn_min_s,
            "turn_max_s": args.turn_max_s,
            "initial_jitter_s": args.initial_jitter_s,
            "chunk_ms": args.chunk_ms,
            "realtime_input": args.realtime_input,
            "sleep_before_chunk": args.sleep_before_chunk,
            "speaker": args.speaker,
            "max_new_tokens": args.max_new_tokens,
            "talker_max_new_tokens": args.talker_max_new_tokens,
            "rpc_timeout_s": args.rpc_timeout_s,
            "close_rpc_timeout_s": args.close_rpc_timeout_s,
            "save_sample_rate": args.save_sample_rate,
            "min_saved_non_edge_fraction": args.min_saved_non_edge_fraction,
            "include_edge_cases": args.include_edge_cases,
            "edge_case_rate": args.edge_case_rate,
            "rollback_rate": args.rollback_rate,
            "rollback_min_completed_turns": args.rollback_min_completed_turns,
            "rollback_max_rewind_turns": args.rollback_max_rewind_turns,
            "telemetry_jsonl": args.telemetry_jsonl,
            "telemetry_interval_s": args.telemetry_interval_s,
            "runtime_snapshot_log": args.runtime_snapshot_log,
            "internal_state_url": _internal_state_url(args)
            if args.telemetry_jsonl and not args.runtime_snapshot_log
            else None,
        },
        "aggregate": _aggregate(results, closes, rollbacks),
        "telemetry_summary": _telemetry_summary(telemetry_samples),
        "close_results": closes,
        "rollback_results": [asdict(rollback) for rollback in rollbacks],
        "results": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    agg = payload["aggregate"]
    realtime_failures = agg["non_realtime"] or agg["non_realtime_audio_stream"]
    if (
        agg["failed"]
        or agg["close_failures"]
        or agg["rollback_failures"]
        or (realtime_failures and not args.ignore_realtime_failures)
        or agg["saved_non_edge_fraction"] < args.min_saved_non_edge_fraction
    ):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41012)
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--session-duration-s", type=float, default=1800.0)
    parser.add_argument("--turn-min-s", type=float, default=8.0)
    parser.add_argument("--turn-max-s", type=float, default=25.0)
    parser.add_argument("--initial-jitter-s", type=float, default=0.0)
    parser.add_argument("--fixture", action="append", required=True)
    parser.add_argument("--include-edge-cases", action="store_true")
    parser.add_argument("--edge-case-rate", type=float, default=0.05)
    parser.add_argument("--save-sample-rate", type=float, default=0.10)
    parser.add_argument("--min-saved-non-edge-fraction", type=float, default=0.05)
    parser.add_argument("--save-output-dir", required=True)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.set_defaults(realtime_input=True)
    parser.add_argument("--realtime-input", dest="realtime_input", action="store_true")
    parser.add_argument("--no-realtime-input", dest="realtime_input", action="store_false")
    parser.add_argument(
        "--sleep-before-chunk",
        action="store_true",
        help=(
            "With --realtime-input, wait for each chunk's duration before "
            "sending it. Use this for paced larger-chunk cadence sweeps, "
            "e.g. sending a 5s chunk at t=5s instead of t=0."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--talker-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--speaker", default="chelsie")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--request-prefix", default="qwen3-omni-soak")
    parser.add_argument("--session-prefix", default="qwen3-omni-soak")
    parser.add_argument(
        "--rpc-timeout-s",
        type=float,
        default=0.0,
        help="Per-turn gRPC deadline in seconds; <=0 means no deadline.",
    )
    parser.add_argument(
        "--close-rpc-timeout-s",
        type=float,
        default=0.0,
        help="CloseAudioSession gRPC deadline in seconds; <=0 means no deadline.",
    )
    parser.add_argument("--rollback-rate", type=float, default=0.0)
    parser.add_argument("--rollback-min-completed-turns", type=int, default=3)
    parser.add_argument("--rollback-max-rewind-turns", type=int, default=2)
    parser.add_argument("--rollback-rpc-timeout-s", type=float, default=300.0)
    parser.add_argument("--telemetry-jsonl")
    parser.add_argument("--telemetry-interval-s", type=float, default=5.0)
    parser.add_argument("--telemetry-timeout-s", type=float, default=5.0)
    parser.add_argument("--internal-state-url")
    parser.add_argument("--runtime-snapshot-log")
    parser.add_argument(
        "--ignore-realtime-failures",
        action="store_true",
        help=(
            "For prewarm only: still fail on request/close/rollback errors, "
            "but do not fail because cold traffic missed realtime."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.sessions <= 0:
        raise ValueError("--sessions must be positive")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if args.session_duration_s <= 0:
        raise ValueError("--session-duration-s must be positive")
    if args.turn_min_s < 0 or args.turn_max_s < args.turn_min_s:
        raise ValueError("--turn-min-s/--turn-max-s are invalid")
    if args.initial_jitter_s < 0:
        raise ValueError("--initial-jitter-s must be non-negative")
    if args.telemetry_interval_s <= 0:
        raise ValueError("--telemetry-interval-s must be positive")
    if args.telemetry_timeout_s <= 0:
        raise ValueError("--telemetry-timeout-s must be positive")
    if args.runtime_snapshot_log is not None and not Path(
        args.runtime_snapshot_log
    ).is_file():
        raise FileNotFoundError(args.runtime_snapshot_log)
    if not 0 <= args.save_sample_rate <= 1:
        raise ValueError("--save-sample-rate must be in [0, 1]")
    if not 0 <= args.min_saved_non_edge_fraction <= 1:
        raise ValueError("--min-saved-non-edge-fraction must be in [0, 1]")
    if args.save_sample_rate < args.min_saved_non_edge_fraction:
        raise ValueError(
            "--save-sample-rate must be >= --min-saved-non-edge-fraction"
        )
    if not 0 <= args.edge_case_rate <= 0.05:
        raise ValueError("--edge-case-rate must be in [0, 0.05]")
    if not 0 <= args.rollback_rate <= 1:
        raise ValueError("--rollback-rate must be in [0, 1]")
    if args.rollback_min_completed_turns <= 0:
        raise ValueError("--rollback-min-completed-turns must be positive")
    if args.rollback_max_rewind_turns <= 0:
        raise ValueError("--rollback-max-rewind-turns must be positive")
    if args.system_prompt_file:
        if args.system_prompt:
            raise ValueError("Use either --system-prompt or --system-prompt-file")
        args.system_prompt = Path(args.system_prompt_file).read_text()
    if not args.system_prompt:
        raise ValueError("Qwen3-Omni soak requires --system-prompt-file")
    if args.speaker != "chelsie":
        raise ValueError("Qwen3-Omni soak requires --speaker chelsie")

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
