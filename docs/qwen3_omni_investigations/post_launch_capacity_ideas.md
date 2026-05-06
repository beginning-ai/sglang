# Post-Launch Capacity Ideas

## SpecForge / Speculative Decoding

Decision: post-launch only.

SpecForge or another speculative-decoding path may be a high-EV capacity
project after the launch stack is stable. It could help if a small draft
model can predict one of the Qwen3 Omni decode lanes well enough to
reduce accepted-model work.

Do not include this in the current bs=50 tail-reduction path. It adds
training, acceptance/correctness validation, audio-quality risk, and
integration work. The current active path should keep focusing on
semantic-preserving fast paths, batching, and measured p99/p100 tails in
the existing model.
