"""Classify Qwen3-Omni realtime soak misses against server phase logs."""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?:(\d+):)?\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.*)$")
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_#-]*)=([^,]+)")


@dataclass(frozen=True)
class LogRow:
    line_no: int
    ts: float
    kind: str
    message: str
    fields: dict[str, Any]


def parse_log_ts(raw: str) -> float:
    parsed = dt.datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    return parsed.replace(tzinfo=dt.timezone.utc).timestamp()


def scalar(value: str) -> Any:
    value = value.strip()
    if value in {"True", "False"}:
        return value == "True"
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_key_values(text: str) -> dict[str, Any]:
    return {key: scalar(value) for key, value in KEY_VALUE_RE.findall(text)}


def parse_prefill_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, value in re.findall(r"(#[A-Za-z0-9_-]+|token usage|cuda graph|input throughput \(token/s\)): ([^,]+)", text):
        normalized = key.replace("#", "").replace(" ", "_").replace("-", "_")
        normalized = normalized.replace("(", "").replace(")", "").replace("/", "_per_")
        fields[normalized] = scalar(value)
    return fields


def parse_server_log(path: Path) -> list[LogRow]:
    rows: list[LogRow] = []
    for physical_line_no, raw_line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        match = TIMESTAMP_RE.match(raw_line)
        if match is None:
            continue
        line_no = int(match.group(1) or physical_line_no)
        ts = parse_log_ts(match.group(2))
        message = match.group(3)
        if "Qwen3-Omni audio phase drained:" in message:
            fields = parse_key_values(message.split(":", 1)[1])
            rows.append(LogRow(line_no, ts, "phase", message, fields))
        elif "Qwen3-Omni scheduler action selected:" in message:
            fields = parse_key_values(message.split(":", 1)[1])
            rows.append(LogRow(line_no, ts, "scheduler", message, fields))
        elif "Qwen3-Omni runtime snapshot:" in message:
            stats = json.loads(message.split(" stats=", 1)[1])
            rows.append(LogRow(line_no, ts, "snapshot", message, stats))
        elif "Prefill batch," in message:
            rows.append(LogRow(line_no, ts, "prefill", message, parse_prefill_fields(message)))
        elif "Qwen3-Omni Talker decode graph captured:" in message:
            fields = parse_key_values(message.split(":", 1)[1])
            rows.append(LogRow(line_no, ts, "capture", message, fields))
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def fmt_s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def load_results(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    results = data["results"]
    return data.get("aggregate", {}), results


def miss_tags(result: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if result["first_audio_after_eoi_s"] is not None and result["first_audio_after_eoi_s"] > 0.8:
        tags.append("first_audio>800ms")
    if (
        result["audio_stream_rtf_after_first_audio"] is not None
        and result["audio_stream_rtf_after_first_audio"] > 1.0
    ):
        tags.append("audio_stream>1")
    if (
        result["complete_tail_after_last_audio_s"] is not None
        and result["complete_tail_after_last_audio_s"] > 0.1
    ):
        tags.append("complete_tail>100ms")
    if not tags:
        tags.append("fixed_overhead")
    return tags


def between(rows: list[LogRow], times: list[float], start: float, end: float) -> list[LogRow]:
    lo = bisect.bisect_left(times, start)
    hi = bisect.bisect_right(times, end)
    return rows[lo:hi]


def nearest_snapshot(rows: list[LogRow], times: list[float], target: float) -> LogRow | None:
    idx = bisect.bisect_right(times, target) - 1
    while idx >= 0:
        if rows[idx].kind == "snapshot":
            return rows[idx]
        idx -= 1
    return None


def phase_summary(rows: list[LogRow]) -> dict[str, Any]:
    phase_rows = [row for row in rows if row.kind == "phase"]
    by_phase: dict[str, list[LogRow]] = defaultdict(list)
    for row in phase_rows:
        by_phase[str(row.fields["phase"])].append(row)

    summary: dict[str, Any] = {}
    for phase, phase_items in sorted(by_phase.items()):
        wall = [float(row.fields.get("wall_ms", 0.0)) for row in phase_items]
        queue = [float(row.fields.get("queue_wait_ms", 0.0)) for row in phase_items]
        summary[phase] = {
            "count": len(phase_items),
            "total_wall_ms": sum(wall),
            "max_wall_ms": max(wall),
            "p95_wall_ms": percentile(wall, 95),
            "max_queue_wait_ms": max(queue),
        }
    return summary


def top_phase_rows(rows: list[LogRow], limit: int) -> list[LogRow]:
    phase_rows = [row for row in rows if row.kind in {"phase", "prefill", "capture"}]

    def score(row: LogRow) -> float:
        if row.kind == "prefill":
            throughput = float(row.fields.get("input_throughput_token_per_s", 0.0))
            return 1000.0 / max(throughput, 1.0)
        return max(
            float(row.fields.get("wall_ms", 0.0)),
            float(row.fields.get("queue_wait_ms", 0.0)),
            float(row.fields.get("max_enqueue_wall_ms", 0.0)),
            float(row.fields.get("max_forward_enqueue_ms", 0.0)),
        )

    return sorted(phase_rows, key=score, reverse=True)[:limit]


def snapshot_line(snapshot: LogRow | None) -> str:
    if snapshot is None:
        return "snapshot=n/a"
    stats = snapshot.fields
    component_kv = stats["component_kv"]
    thinker = component_kv["thinker"]
    talker = component_kv["talker"]
    code_predictor = component_kv["code_predictor"]
    return (
        f"snapshot_line={snapshot.line_no} active={stats['active_audio_turns']} "
        f"queued={stats['queued_audio_turns']} running={stats['running_batch']} "
        f"thinker_kv={thinker['used']}/{thinker['size']} "
        f"talker_kv={talker['used']}/{talker['size']} "
        f"codepred_kv={code_predictor['used']}/{code_predictor['size']}"
    )


def print_interval(name: str, rows: list[LogRow]) -> None:
    summary = phase_summary(rows)
    print(f"    {name}:")
    if not summary:
        print("      no phase rows")
    for phase, values in summary.items():
        print(
            "      "
            f"{phase}: count={values['count']} total={values['total_wall_ms']:.1f}ms "
            f"p95/max={fmt_ms(values['p95_wall_ms'])}/{fmt_ms(values['max_wall_ms'])}ms "
            f"max_queue={fmt_ms(values['max_queue_wait_ms'])}ms"
        )
    top = top_phase_rows(rows, 3)
    for row in top:
        if row.kind == "prefill":
            print(f"      top line {row.line_no}: {row.message}")
        else:
            phase = row.fields.get("phase", row.kind)
            print(
                f"      top line {row.line_no}: {phase} action={row.fields.get('scheduler_action')} "
                f"bs={row.fields.get('batch_size')} wall={row.fields.get('wall_ms')} "
                f"queue={row.fields.get('queue_wait_ms')} msg={row.message[:220]}"
            )


def print_result_windows(
    *,
    title: str,
    results: list[dict[str, Any]],
    rows: list[LogRow],
    times: list[float],
    window_s: float,
) -> None:
    print(f"{title}:")
    for idx, result in enumerate(results, start=1):
        eoi = float(result["eoi_wall_s"])
        first = float(result["first_audio_wall_s"])
        last = float(result["last_audio_wall_s"])
        complete = float(result["complete_wall_s"])
        request_start = float(result["request_start_wall_s"])
        print()
        print(
            f"{idx}. rid={result['turn_rid']} session={result['session_index']} "
            f"turn={result['turn_index']} fixture={result['fixture']} "
            f"prompt={result['prompt_tokens']} completion={result['completion_tokens']} "
            f"output={fmt_s(result['output_audio_s'])}s"
        )
        print(
            "    "
            f"rtf={result['rtf_after_eoi']:.3f} "
            f"audio_stream={result['audio_stream_rtf_after_first_audio']:.3f} "
            f"first_audio={fmt_s(result['first_audio_after_eoi_s'])}s "
            f"tail={fmt_s(result['complete_tail_after_last_audio_s'])}s "
            f"tags={','.join(miss_tags(result))}"
        )
        print(f"    {snapshot_line(nearest_snapshot(rows, times, first))}")
        intervals = [
            ("pre_eoi", request_start, eoi),
            ("eoi_to_first_audio", eoi, first),
            ("audio_stream", first, last),
            ("completion_tail", last, complete),
            ("nearby_window", eoi - window_s, complete + window_s),
        ]
        for name, start, end in intervals:
            print_interval(name, between(rows, times, start, end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--window-s", type=float, default=2.0)
    args = parser.parse_args()

    aggregate, results = load_results(args.result_json)
    rows = parse_server_log(args.server_log)
    rows.sort(key=lambda row: (row.ts, row.line_no))
    times = [row.ts for row in rows]

    ok = [result for result in results if result["ok"]]
    misses = [result for result in ok if not result["realtime_after_eoi"]]
    tag_counts = Counter(tag for result in misses for tag in miss_tags(result))
    audio_stream_misses = [result for result in ok if not result["realtime_audio_stream_after_first_audio"]]
    first_audio_misses = [
        result
        for result in ok
        if result["first_audio_after_eoi_s"] is not None and result["first_audio_after_eoi_s"] > 0.8
    ]
    tail_misses = [
        result
        for result in ok
        if result["complete_tail_after_last_audio_s"] is not None
        and result["complete_tail_after_last_audio_s"] > 0.1
    ]

    print(f"# Qwen3-Omni Soak Miss Report")
    print()
    print(f"result: `{args.result_json}`")
    print(f"server_log: `{args.server_log}`")
    print()
    print(
        f"turns ok/total: {aggregate.get('ok')}/{aggregate.get('turns')}  "
        f"strict: {aggregate.get('realtime')}/{aggregate.get('turns')}  "
        f"audio_stream: {aggregate.get('realtime_audio_stream')}/{aggregate.get('turns')}"
    )
    print(
        "rtf mean/max: "
        f"{aggregate.get('mean_rtf_after_eoi'):.3f}/{aggregate.get('max_rtf_after_eoi'):.3f}  "
        "first_audio max: "
        f"{aggregate.get('max_first_audio_after_eoi_s'):.3f}s  "
        "complete_tail max: "
        f"{aggregate.get('max_complete_tail_after_last_audio_s'):.3f}s"
    )
    print()
    print("miss classes:")
    for tag, count in tag_counts.most_common():
        print(f"- {tag}: {count}")
    print(f"- audio-stream misses: {len(audio_stream_misses)}")
    print(f"- first-audio >800ms: {len(first_audio_misses)}")
    print(f"- complete-tail >100ms: {len(tail_misses)}")
    print()

    prompt_tokens = [int(result["prompt_tokens"]) for result in misses]
    turn_indices = [int(result["turn_index"]) for result in misses]
    output_s = [float(result["output_audio_s"]) for result in misses if result["output_audio_s"] is not None]
    if misses:
        print(
            "miss distribution: "
            f"turn_index median/max={statistics.median(turn_indices):.0f}/{max(turn_indices)}  "
            f"prompt_tokens median/max={statistics.median(prompt_tokens):.0f}/{max(prompt_tokens)}  "
            f"output_audio_s median/max={statistics.median(output_s):.2f}/{max(output_s):.2f}"
        )
        print()

    ranked = sorted(
        misses,
        key=lambda result: (
            float(result["rtf_after_eoi"]),
            float(result["first_audio_after_eoi_s"] or 0.0),
            float(result["audio_stream_rtf_after_first_audio"] or 0.0),
        ),
        reverse=True,
    )[: args.top]

    print_result_windows(
        title=f"top {len(ranked)} strict misses",
        results=ranked,
        rows=rows,
        times=times,
        window_s=args.window_s,
    )

    latency_ranked = sorted(
        first_audio_misses,
        key=lambda result: float(result["first_audio_after_eoi_s"] or 0.0),
        reverse=True,
    )[: args.top]
    if latency_ranked:
        print()
        print_result_windows(
            title=f"top {len(latency_ranked)} first-audio misses",
            results=latency_ranked,
            rows=rows,
            times=times,
            window_s=args.window_s,
        )


if __name__ == "__main__":
    main()
