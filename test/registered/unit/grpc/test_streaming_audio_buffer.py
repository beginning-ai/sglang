import numpy as np

from sglang.srt.audio import StreamingAudioBuffer, get_audio_token_count


class FakeFeatureExtractor:
    def __call__(self, audio, **kwargs):
        return {"input_features": np.asarray(audio)[None, None, :]}


def test_audio_token_count_matches_full_and_partial_chunks():
    assert get_audio_token_count(0) == 0
    assert get_audio_token_count(100) == 13
    assert get_audio_token_count(200) == 26
    assert get_audio_token_count(1) == 1
    assert get_audio_token_count(99) == 13


def test_streaming_audio_buffer_emits_full_chunk_then_final_partial():
    buffer = StreamingAudioBuffer(FakeFeatureExtractor())
    full_second = (np.zeros(16000, dtype=np.int16)).tobytes()
    half_second = (np.zeros(8000, dtype=np.int16)).tobytes()

    chunks = buffer.add_samples(full_second)
    assert len(chunks) == 1
    features, frames = chunks[0]
    assert frames == 100
    assert features["input_features"].shape[-1] == 16000

    assert buffer.add_samples(half_second) == []
    features, frames = buffer.flush()
    assert frames == 50
    assert features["input_features"].shape[-1] == 16000


def test_streaming_audio_buffer_emits_all_complete_chunks_from_large_input():
    buffer = StreamingAudioBuffer(FakeFeatureExtractor())
    two_and_half_seconds = (np.zeros(40000, dtype=np.int16)).tobytes()

    chunks = buffer.add_samples(two_and_half_seconds)
    assert len(chunks) == 2
    assert [frames for _, frames in chunks] == [100, 100]
    assert buffer.buffered_samples == 8000

    features, frames = buffer.flush()
    assert frames == 50
    assert features["input_features"].shape[-1] == 16000


def test_streaming_audio_buffer_flush_is_empty_after_exact_large_input():
    buffer = StreamingAudioBuffer(FakeFeatureExtractor())
    five_seconds = (np.zeros(80000, dtype=np.int16)).tobytes()

    chunks = buffer.add_samples(five_seconds)
    assert len(chunks) == 5
    assert [frames for _, frames in chunks] == [100] * 5
    assert buffer.buffered_samples == 0
    assert buffer.flush() is None
