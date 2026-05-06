# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Audio input buffering for Qwen3-Omni streaming turns."""

from typing import Optional, Tuple

import numpy as np


def get_audio_token_count(num_frames: int) -> int:
    """Return Qwen3-Omni audio tokens produced by ``num_frames`` mel frames."""
    if num_frames == 0:
        return 0

    full_chunks = num_frames // 100
    remainder = num_frames % 100
    if remainder == 0:
        return full_chunks * 13

    feat_len = (remainder - 1) // 2 + 1
    remainder_tokens = ((feat_len - 1) // 2 + 1 - 1) // 2 + 1
    return full_chunks * 13 + remainder_tokens


class StreamingAudioBuffer:
    """Buffer PCM16 audio into 100-frame mel chunks plus one final partial."""

    SAMPLE_RATE = 16000
    HOP_LENGTH = 160
    SAMPLES_PER_100_FRAMES = 16000
    MIN_SAMPLES_FOR_FRAME = HOP_LENGTH

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.sample_buffer = np.array([], dtype=np.float32)

    def add_samples(self, pcm16_bytes: bytes) -> list[Tuple[dict, int]]:
        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.sample_buffer = np.concatenate([self.sample_buffer, audio])

        chunks = []
        while len(self.sample_buffer) >= self.SAMPLES_PER_100_FRAMES:
            chunk = self.sample_buffer[: self.SAMPLES_PER_100_FRAMES]
            self.sample_buffer = self.sample_buffer[self.SAMPLES_PER_100_FRAMES :]
            features = self.feature_extractor(
                chunk,
                sampling_rate=self.SAMPLE_RATE,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            chunks.append((features, 100))
        return chunks

    def flush(self) -> Optional[Tuple[dict, int]]:
        if len(self.sample_buffer) < self.MIN_SAMPLES_FOR_FRAME:
            self.sample_buffer = np.array([], dtype=np.float32)
            return None

        actual_num_frames = len(self.sample_buffer) // self.HOP_LENGTH
        if actual_num_frames == 0:
            self.sample_buffer = np.array([], dtype=np.float32)
            return None

        padded = np.pad(
            self.sample_buffer,
            (0, self.SAMPLES_PER_100_FRAMES - len(self.sample_buffer)),
            mode="constant",
            constant_values=0.0,
        )
        self.sample_buffer = np.array([], dtype=np.float32)
        features = self.feature_extractor(
            padded,
            sampling_rate=self.SAMPLE_RATE,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        return features, actual_num_frames

    def reset(self):
        self.sample_buffer = np.array([], dtype=np.float32)

    @property
    def buffered_samples(self) -> int:
        return len(self.sample_buffer)

    @property
    def buffered_duration_ms(self) -> float:
        return len(self.sample_buffer) / (self.SAMPLE_RATE / 1000)
