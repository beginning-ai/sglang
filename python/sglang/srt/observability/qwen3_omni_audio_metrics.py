"""Prometheus metrics for Qwen3-Omni audio serving."""

from __future__ import annotations


TURN_LATENCY_BUCKETS = (
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    8.0,
    10.0,
    15.0,
    20.0,
    30.0,
    45.0,
    60.0,
    90.0,
    120.0,
)

FIRST_AUDIO_LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    5.0,
)

_TOKENS_TOTAL = None


def _base_labels(server_args) -> dict[str, str]:
    return {"model_name": server_args.served_model_name}


def _tokens_total():
    global _TOKENS_TOTAL
    if _TOKENS_TOTAL is None:
        from prometheus_client import Counter

        _TOKENS_TOTAL = Counter(
            name="sglang:qwen3_omni_audio_tokens_total",
            documentation="Qwen3-Omni audio-path token counts by component and phase.",
            labelnames=("model_name", "component", "phase"),
        )
    return _TOKENS_TOTAL


def count_qwen3_omni_tokens(value) -> int:
    if value is None:
        return 0
    if hasattr(value, "numel"):
        return int(value.numel())
    return len(value)


def observe_qwen3_omni_audio_tokens(
    server_args, component: str, phase: str, count: int
) -> None:
    if count and server_args.enable_metrics:
        _tokens_total().labels(
            **_base_labels(server_args),
            component=component,
            phase=phase,
        ).inc(count)


class Qwen3OmniAudioMetrics:
    def __init__(self, server_args):
        from prometheus_client import Counter, Histogram

        self.labels = _base_labels(server_args)
        self.turns_total = Counter(
            name="sglang:qwen3_omni_audio_turns_total",
            documentation="Number of Qwen3-Omni audio turns processed.",
            labelnames=[*self.labels.keys(), "status"],
        )
        self.turn_latency_seconds = Histogram(
            name="sglang:qwen3_omni_audio_turn_latency_seconds",
            documentation="Qwen3-Omni AudioTurn latency from config to terminal event.",
            labelnames=[*self.labels.keys(), "status"],
            buckets=TURN_LATENCY_BUCKETS,
        )
        self.first_audio_latency_seconds = Histogram(
            name="sglang:qwen3_omni_audio_first_audio_latency_seconds",
            documentation="Qwen3-Omni AudioTurn latency from EOI to first audio chunk.",
            labelnames=self.labels.keys(),
            buckets=FIRST_AUDIO_LATENCY_BUCKETS,
        )

    def observe_turn(self, status: str, latency_seconds: float):
        labels = {**self.labels, "status": status}
        self.turns_total.labels(**labels).inc()
        self.turn_latency_seconds.labels(**labels).observe(latency_seconds)

    def observe_first_audio(self, latency_seconds: float):
        self.first_audio_latency_seconds.labels(**self.labels).observe(latency_seconds)
