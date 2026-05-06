#!/usr/bin/env python3
"""Transcribe one WAV file with NVIDIA Parakeet for Qwen3-Omni validation."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)

    with contextlib.redirect_stdout(sys.stderr):
        import torch
        import nemo.collections.asr as nemo_asr
        from omegaconf import open_dict

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
        if getattr(model, "cfg", None) is not None and "decoding" in model.cfg:
            with open_dict(model.cfg.decoding.greedy):
                model.cfg.decoding.greedy.use_cuda_graph_decoder = False
            model.change_decoding_strategy(model.cfg.decoding)

        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for --device cuda")
            model = model.cuda()
        else:
            model = model.to(device="cpu")
        model.eval()

        output = model.transcribe([str(args.audio)], batch_size=args.batch_size)
    if not output:
        print("")
        return 0
    transcript = output[0].text if hasattr(output[0], "text") else str(output[0])
    print(transcript.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
