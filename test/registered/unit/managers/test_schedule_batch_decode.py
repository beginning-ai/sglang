from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class NoSpec:
    def is_none(self):
        return True


def test_prepare_for_decode_clears_prefill_embedding_overrides():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    batch = object.__new__(ScheduleBatch)
    batch.reqs = [
        SimpleNamespace(thinker_done=True, decode_batch_idx=0),
        SimpleNamespace(thinker_done=True, decode_batch_idx=0),
    ]
    batch.input_embeds = torch.ones((2, 8))
    batch.replace_embeds = torch.ones((1, 8))
    batch.replace_positions = torch.tensor([0])
    batch.enable_overlap = False
    batch.spec_algorithm = NoSpec()
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.output_ids = torch.tensor([10, 11])
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.hisparse_coordinator = None

    batch.prepare_for_decode()

    assert batch.input_embeds is None
    assert batch.replace_embeds is None
    assert batch.replace_positions is None
    assert [req.decode_batch_idx for req in batch.reqs] == [1, 1]
