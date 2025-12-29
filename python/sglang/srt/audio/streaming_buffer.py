# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Audio streaming buffer for Qwen3-Omni real-time audio input.

Buffers PCM16 audio samples until 100 mel frames are available for processing.
The audio encoder CNN has hardcoded 100-frame boundaries, so we must buffer
until we have exactly 100 mel frames before sending to the model.

TODO: Investigate smaller chunks (e.g., 200ms) with zero-padding to 100-frame
boundary. May introduce artifacts but would reduce latency from ~1s to ~200ms.
"""

from typing import Optional, Tuple

import numpy as np


def get_audio_token_count(num_frames: int) -> int:
    """Calculate exact audio token count from mel frames using CNN formula.

    The Qwen3-Omni audio encoder uses a CNN with specific stride patterns.
    This formula matches transformers' _get_feat_extract_output_lengths().

    Args:
        num_frames: Number of mel spectrogram frames

    Returns:
        Number of audio tokens after CNN compression
    """
    if num_frames == 0:
        return 0
    remainder = num_frames % 100
    full_chunks = num_frames // 100
    if remainder == 0:
        return full_chunks * 13
    # CNN formula for remainder frames
    feat_len = (remainder - 1) // 2 + 1
    remainder_tokens = ((feat_len - 1) // 2 + 1 - 1) // 2 + 1
    return remainder_tokens + full_chunks * 13


class StreamingAudioBuffer:
    """Buffer audio samples until 100 mel frames are available.

    The audio encoder CNN has hardcoded 100-frame boundaries.
    Every 100 mel frames → exactly 13 audio tokens after CNN compression.

    Audio parameters (after resampling to 16kHz):
    - Target sample rate: 16kHz (for WhisperFeatureExtractor)
    - Hop length: 160 samples (10ms)
    - 100 mel frames = 16000 samples = 1 second of audio
    """

    # 100 mel frames * 160 hop_length = 16000 samples = 1 second at 16kHz
    SAMPLE_RATE = 16000
    SAMPLES_PER_100_FRAMES = 16000
    HOP_LENGTH = 160
    MIN_SAMPLES_FOR_FRAME = 160  # Minimum samples to produce at least 1 mel frame

    def __init__(self, feature_extractor):
        """Initialize the streaming audio buffer.

        Args:
            feature_extractor: WhisperFeatureExtractor instance from transformers
        """
        self.feature_extractor = feature_extractor
        self.sample_buffer = np.array([], dtype=np.float32)

    def add_samples(self, pcm16_bytes: bytes) -> Optional[Tuple[dict, int]]:
        """Add PCM16 bytes to buffer, return mel features if 100 frames available.

        Args:
            pcm16_bytes: Raw PCM16 audio bytes (16kHz, mono, little-endian)

        Returns:
            Tuple of (mel_features dict, num_frames) if 100 frames available, else None
        """
        # Convert PCM16 bytes to float32 normalized to [-1, 1]
        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.sample_buffer = np.concatenate([self.sample_buffer, audio])

        if len(self.sample_buffer) < self.SAMPLES_PER_100_FRAMES:
            return None

        # Extract exactly 16000 samples (1 second at 16kHz = 100 mel frames)
        chunk = self.sample_buffer[: self.SAMPLES_PER_100_FRAMES]
        self.sample_buffer = self.sample_buffer[self.SAMPLES_PER_100_FRAMES :]

        # Compute mel features (100 frames → 13 audio tokens after CNN)
        result = self.feature_extractor(
            chunk,
            sampling_rate=self.SAMPLE_RATE,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        return (result, 100)  # Full chunk always has 100 frames

    def flush(self) -> Optional[Tuple[dict, int]]:
        """Flush remaining buffer with padding for shape consistency.

        Called when audio input ends to process any remaining samples.
        Pads to 100 frames for tensor shape consistency, but returns actual
        frame count so the scheduler can calculate correct token count.

        Returns:
            Tuple of (mel_features dict, actual_num_frames) if samples remain, else None
        """
        if len(self.sample_buffer) < self.MIN_SAMPLES_FOR_FRAME:
            # Less than 1 mel frame worth of samples, discard
            self.sample_buffer = np.array([], dtype=np.float32)
            return None

        # Calculate actual number of mel frames before padding
        actual_num_frames = len(self.sample_buffer) // self.HOP_LENGTH
        if actual_num_frames == 0:
            self.sample_buffer = np.array([], dtype=np.float32)
            return None

        # Pad to 100 frames (16000 samples at 16kHz) for tensor shape consistency
        # The audio encoder requires consistent shapes for batching
        padded = np.pad(
            self.sample_buffer,
            (0, self.SAMPLES_PER_100_FRAMES - len(self.sample_buffer)),
            mode="constant",
            constant_values=0.0,
        )
        self.sample_buffer = np.array([], dtype=np.float32)

        result = self.feature_extractor(
            padded,
            sampling_rate=self.SAMPLE_RATE,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        # Return actual frame count (not 100) for correct token calculation
        return (result, actual_num_frames)

    def reset(self):
        """Reset the buffer, discarding any buffered samples."""
        self.sample_buffer = np.array([], dtype=np.float32)

    @property
    def buffered_samples(self) -> int:
        """Return the number of samples currently in the buffer."""
        return len(self.sample_buffer)

    @property
    def buffered_duration_ms(self) -> float:
        """Return the buffered audio duration in milliseconds."""
        return len(self.sample_buffer) / (self.SAMPLE_RATE / 1000)
