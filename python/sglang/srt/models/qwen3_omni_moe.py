# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
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
"""Inference-only Qwen3-Omni model compatible with HuggingFace weights."""

import functools
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerResizeMLP,
)

from sglang.srt.configs.qwen3_omni import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.mm_utils import embed_mm_inputs
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeModel
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.models.qwen3_vl_moe import (
    Qwen3MoeLLMModel,
    Qwen3VLMoeForConditionalGeneration,
    load_fused_expert_weights,
)
from sglang.srt.speculative.spec_utils import fast_topk
from sglang.srt.utils import add_prefix, is_npu, logger


def _sample_codec_tokens(
    logits: torch.Tensor,
    codec_eos_token_id: int = 2150,
) -> torch.Tensor:
    """Sample codec tokens from logits. Works for both single and batched inputs.

    Args:
        logits: [vocab_size] or [batch_size, vocab_size]
        codec_eos_token_id: EOS token to preserve during suppression

    Returns:
        tokens: [] scalar or [batch_size] tensor of sampled token IDs
    """
    single_input = logits.dim() == 1
    if single_input:
        logits = logits.unsqueeze(0)

    vocab_size = logits.shape[-1]

    # Suppress special tokens in the last 1024 vocab IDs except codec_eos
    suppress_start = vocab_size - 1024
    suppress_mask = torch.zeros_like(logits, dtype=torch.bool)
    suppress_mask[:, suppress_start:vocab_size] = True
    if codec_eos_token_id < vocab_size:
        suppress_mask[:, codec_eos_token_id] = False
    logits = logits.masked_fill(suppress_mask, float("-inf"))

    # Apply temperature and top-k filtering
    logits = logits / 0.9
    topk_logits, topk_indices = torch.topk(logits, 50, dim=-1)
    probs = F.softmax(topk_logits, dim=-1)

    # Sample from top-k distribution
    sampled_idx = torch.multinomial(probs, 1).squeeze(-1)
    tokens = topk_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

    if single_input:
        return tokens.squeeze(0)
    return tokens


@dataclass
class ThinkerTalkerOutput:
    """Output from Qwen3-Omni forward pass with both thinker and talker results."""

    thinker_logits: LogitsProcessorOutput
    codec_frames: Optional[List[List[int]]] = None
    tts_pad_embed: Optional[torch.Tensor] = None
    talker_out_cache_loc_list: Optional[List[torch.Tensor]] = None
    updated_talker_kv_locs_list: Optional[List[torch.Tensor]] = None
    talker_needs_prefill: bool = False

    # Delegate LogitsProcessorOutput attributes to thinker_logits for compatibility
    # with BatchResult.copy_to_cpu() which expects these attributes
    @property
    def hidden_states(self):
        return self.thinker_logits.hidden_states if self.thinker_logits else None

    @property
    def next_token_logprobs(self):
        return self.thinker_logits.next_token_logprobs if self.thinker_logits else None

    @property
    def input_token_logprobs(self):
        return self.thinker_logits.input_token_logprobs if self.thinker_logits else None

    @property
    def customized_info(self):
        return self.thinker_logits.customized_info if self.thinker_logits else None


class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        embed_dim = config.d_model
        self.embed_dim = config.d_model
        self.self_attn = VisionAttention(
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            projection_size=embed_dim,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            x=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


class Qwen3OmniMoeAudioEncoder(PreTrainedModel):
    config: Qwen3OmniMoeAudioEncoderConfig

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.layers = nn.ModuleList(
            [
                Qwen3OmniMoeAudioEncoderLayer(config)
                for _ in range(config.encoder_layers)
            ]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv_out = nn.Linear(
            config.downsample_hidden_size
            * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(length, dtype=torch.bool, device=padded_feature.device)
                for length in feature_lens_after_cnn
            ],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            self.n_window_infer // (self.n_window * 2)
        )
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(
            -1, dtype=torch.int32
        )
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        if is_npu():
            cu_seqlens = cu_seqlens.to("cpu")

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class Qwen3OmniMoeVisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_postshuffle_norm=False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else context_dim, eps=1e-6
        )
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            x.view(-1, self.hidden_size)
            if self.use_postshuffle_norm
            else x.view(-1, x.shape[-1])
        )
        hidden = self.ln_q(x).view(-1, self.hidden_size)
        for layer in self.mlp:
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            hidden = layer(hidden)

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        return hidden


class Qwen3OmniMoeVisionEncoder(Qwen3VLMoeVisionModel):
    config: Qwen3OmniMoeVisionEncoderConfig

    def __init__(
        self,
        config: Qwen3OmniMoeVisionEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = None,
        **kwargs,
    ):
        super().__init__(
            vision_config=config,
            quant_config=quant_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )

        self.merger = Qwen3OmniMoeVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            quant_config=quant_config,
            use_postshuffle_norm=False,
            prefix=add_prefix("merger", prefix),
        )
        self.merger_list = nn.ModuleList(
            [
                Qwen3OmniMoeVisionPatchMerger(
                    dim=config.out_hidden_size,
                    context_dim=config.hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                    use_postshuffle_norm=True,
                    quant_config=quant_config,
                    prefix=add_prefix("merger_list", prefix),
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )
        del self.deepstack_merger_list

    @property
    def deepstack_merger_list(self):
        return self.merger_list

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device


class Qwen3OmniMoeThinkerForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    config: Qwen3OmniMoeThinkerConfig

    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        accept_hidden_layer: int = 24,
    ):
        super().__init__(
            config, quant_config, prefix, language_model_cls=Qwen3MoeLLMModel
        )
        self.audio_tower = Qwen3OmniMoeAudioEncoder(config.audio_config)
        self.visual = Qwen3OmniMoeVisionEncoder(
            config.vision_config,
            quant_config=None,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            prefix=add_prefix("visual", prefix),
        )
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        # Enable layer capture for audio output (talker needs accept_hidden_layer states)
        self.model.layers_to_capture = [accept_hidden_layer]

    def get_audio_feature(self, items: List[MultimodalDataItem]):
        feature_attention_mask = torch.cat(
            [item.feature_attention_mask for item in items], dim=0
        ).type(torch.long)
        input_features = (
            torch.cat([item.feature for item in items])
            .type(self.audio_tower.dtype)
            .to(next(self.audio_tower.parameters()).device)
        )
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        audio_features = audio_outputs.last_hidden_state

        return audio_features


class Qwen3OmniMoeTalkerCodePredictorModel(Qwen3MoeModel):
    """Code predictor model with ModuleList of embeddings for each code group."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id_offset: int = 0,
    ):
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=functools.partial(
                Qwen3MoeDecoderLayer,
                is_layer_sparse=False,
                is_previous_layer_sparse=False,
            ),
            layer_id_offset=layer_id_offset,
        )
        if self.pp_group.is_first_rank:
            self.embed_tokens = nn.ModuleList(
                [
                    VocabParallelEmbedding(
                        config.vocab_size,
                        config.hidden_size,
                        prefix=add_prefix(f"embed_tokens.{i}", prefix),
                    )
                    for i in range(config.num_code_groups - 1)
                ]
            )
        else:
            self.embed_tokens = PPMissingLayer()


class Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(nn.Module):
    """Code predictor for conditional generation with ModuleList of lm_heads."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id_offset: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_id_offset = layer_id_offset

        self.model = Qwen3OmniMoeTalkerCodePredictorModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            layer_id_offset=layer_id_offset,
        )

        # ModuleList of lm_heads for each code group (num_code_groups - 1)
        num_heads = config.num_code_groups - 1
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(num_heads)]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        generation_steps: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for code predictor.

        Args:
            input_ids: Input token ids
            positions: Position indices
            forward_batch: Forward batch info
            input_embeds: Optional pre-computed embeddings (for prefill stage)
            generation_steps: Current generation step (0 to num_code_groups-2),
                              selects which embedding and lm_head to use

        Returns:
            logits: Output logits from the selected lm_head
        """

        if input_embeds is not None and input_embeds.shape[0] > 1:
            # Prefill stage
            generation_steps = input_embeds.shape[0] - 2
        else:
            # Generation stage
            embed_layer = self.model.embed_tokens[generation_steps - 1]
            input_embeds = embed_layer(input_ids)

        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        lm_head = self.lm_head[generation_steps]
        logits = lm_head(hidden_states)

        return logits

    def _sample(
        self,
        logits: torch.Tensor,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.8,
    ) -> torch.Tensor:
        """Sample from logits with top-k and top-p (nucleus) filtering.

        Matches transformers sampling parameters: do_sample=True, top_k=50, top_p=0.8
        """
        probs = torch.softmax(logits, dim=-1)
        if do_sample:
            # Top-k filtering
            topk_p, topk_ids = fast_topk(probs, top_k, dim=-1)
            # Top-p (nucleus) filtering on the top-k probs
            sorted_p, sorted_idx = torch.sort(topk_p, descending=True, dim=-1)
            cumsum_p = torch.cumsum(sorted_p, dim=-1)
            # Remove tokens with cumulative probability above threshold (keep first token above)
            mask = cumsum_p - sorted_p > top_p
            sorted_p[mask] = 0.0
            # Renormalize
            sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
            # Sample from filtered distribution
            idx = torch.multinomial(sorted_p, 1)
            # Map back to original top-k indices
            sampled_sorted_idx = torch.gather(sorted_idx, -1, idx)
            next_token = torch.gather(topk_ids, -1, sampled_sorted_idx)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token

    @torch.no_grad()
    def generate(
        self,
        input_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        num_tokens: int,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.8,
    ) -> torch.Tensor:
        """Generate num_tokens codes autoregressively."""
        device = input_embeds.device
        generated_tokens = []
        all_locs = []

        # === Save original forward_batch state ===
        # The code predictor needs isolated KV cache state, separate from talker
        original_seq_lens = forward_batch.seq_lens
        original_seq_lens_cpu = forward_batch.seq_lens_cpu
        original_seq_lens_sum = forward_batch.seq_lens_sum
        original_req_pool_indices = forward_batch.req_pool_indices
        original_out_cache_loc = forward_batch.out_cache_loc
        original_forward_mode = forward_batch.forward_mode
        original_extend_prefix_lens = forward_batch.extend_prefix_lens
        original_extend_prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu
        original_extend_seq_lens = forward_batch.extend_seq_lens
        original_extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        original_extend_num_tokens = forward_batch.extend_num_tokens
        original_extend_start_loc = forward_batch.extend_start_loc
        # Save override_req_to_token - code predictor uses actual pool, not override
        original_override_req_to_token = forward_batch.override_req_to_token
        forward_batch.override_req_to_token = None  # Clear for code predictor

        # Get memory pool references
        allocator = forward_batch.token_to_kv_pool_allocator
        req_to_token_pool = forward_batch.req_to_token_pool

        # === Allocate isolated state for code predictor ===
        # Allocate a new request slot for code predictor's KV cache
        predictor_req_slots = req_to_token_pool.alloc(1)
        if predictor_req_slots is None:
            raise RuntimeError("Failed to allocate request slot for code predictor")
        predictor_req_idx = predictor_req_slots[0]

        # Clear any stale KV locations from previous use of this slot.
        # This is necessary because slots are reused across requests in a batch loop,
        # and stale indices could cause the attention backend to read wrong KV cache.
        req_to_token_pool.req_to_token[predictor_req_idx, :] = 0

        # Set code predictor's req_pool_indices
        forward_batch.req_pool_indices = torch.tensor(
            [predictor_req_idx], dtype=torch.int32, device=device
        )

        # === Prefill ===
        prefill_len = input_embeds.shape[0]
        prefill_loc = allocator.alloc(prefill_len)
        all_locs.append(prefill_loc)

        # Write prefill locations to req_to_token mapping
        req_to_token_pool.req_to_token[predictor_req_idx, :prefill_len] = prefill_loc

        # Set code predictor's seq_lens state for extend mode
        forward_batch.seq_lens = torch.tensor([prefill_len], dtype=torch.int32, device=device)
        forward_batch.seq_lens_cpu = torch.tensor([prefill_len], dtype=torch.int32)
        forward_batch.seq_lens_sum = prefill_len
        forward_batch.out_cache_loc = prefill_loc
        forward_batch.forward_mode = ForwardMode.EXTEND
        # For fresh prefill, no prefix (all new tokens)
        forward_batch.extend_prefix_lens = torch.zeros(1, dtype=torch.int32, device=device)
        forward_batch.extend_prefix_lens_cpu = [0]
        forward_batch.extend_seq_lens = torch.tensor([prefill_len], dtype=torch.int32, device=device)
        forward_batch.extend_seq_lens_cpu = [prefill_len]
        forward_batch.extend_num_tokens = prefill_len
        forward_batch.extend_start_loc = torch.tensor([0], dtype=torch.int32, device=device)

        # Create code predictor's own positions starting from 0
        predictor_positions = torch.arange(0, prefill_len, dtype=torch.long, device=device)

        # Reinitialize attention metadata with updated forward_batch state
        forward_batch.attn_backend.init_forward_metadata(forward_batch)

        # Note: Code predictor assumes batch_size=1 (single request processing)
        # Clone input_embeds to prevent in-place modification by fused_add_rmsnorm
        hidden_states = self.model(
            input_ids=None,
            positions=predictor_positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds.clone(),
        )
        # Get last token's hidden state: [1, hidden]
        last_hidden = hidden_states[-1:]
        logits = self.lm_head[0](last_hidden)

        # Sample first token - logits is [1, vocab], result is [1] after squeeze
        next_token = self._sample(logits, do_sample, top_k, top_p).squeeze(-1)
        generated_tokens.append(next_token)

        # === Decode loop ===
        current_seq_len = prefill_len
        for step in range(1, num_tokens):
            # Get embedding for the token we just generated
            embed_layer = self.model.embed_tokens[step - 1]
            step_embeds = embed_layer(next_token)  # [1, hidden]

            # Increment predictor's own position
            predictor_positions = predictor_positions[-1:] + 1
            current_seq_len += 1

            # Allocate KV cache for this decode step
            decode_loc = allocator.alloc(1)
            all_locs.append(decode_loc)

            # Update req_to_token mapping with new token location
            req_to_token_pool.req_to_token[predictor_req_idx, current_seq_len - 1] = decode_loc

            # Update forward_batch state for this decode step
            forward_batch.seq_lens = torch.tensor([current_seq_len], dtype=torch.int32, device=device)
            forward_batch.seq_lens_cpu = torch.tensor([current_seq_len], dtype=torch.int32)
            forward_batch.seq_lens_sum = current_seq_len
            forward_batch.out_cache_loc = decode_loc
            forward_batch.forward_mode = ForwardMode.DECODE

            # Reinitialize attention metadata with updated forward_batch state
            forward_batch.attn_backend.init_forward_metadata(forward_batch)

            # Forward pass - step_embeds is [1, hidden]
            hidden_states = self.model(
                input_ids=None,
                positions=predictor_positions,
                forward_batch=forward_batch,
                input_embeds=step_embeds,
            )

            # Use appropriate lm_head for this step
            logits = self.lm_head[step](hidden_states)

            # Sample next token
            next_token = self._sample(logits, do_sample, top_k, top_p).squeeze(-1)
            generated_tokens.append(next_token)

        # === Cleanup ===
        # Free code predictor's KV cache tokens
        allocator.free(torch.cat(all_locs))
        # Free request slot
        req_to_token_pool.free(predictor_req_idx)

        # Restore original forward_batch state
        forward_batch.seq_lens = original_seq_lens
        forward_batch.seq_lens_cpu = original_seq_lens_cpu
        forward_batch.seq_lens_sum = original_seq_lens_sum
        forward_batch.req_pool_indices = original_req_pool_indices
        forward_batch.out_cache_loc = original_out_cache_loc
        forward_batch.forward_mode = original_forward_mode
        forward_batch.extend_prefix_lens = original_extend_prefix_lens
        forward_batch.extend_prefix_lens_cpu = original_extend_prefix_lens_cpu
        forward_batch.extend_seq_lens = original_extend_seq_lens
        forward_batch.extend_seq_lens_cpu = original_extend_seq_lens_cpu
        forward_batch.extend_num_tokens = original_extend_num_tokens
        forward_batch.extend_start_loc = original_extend_start_loc
        # Restore override_req_to_token for talker
        forward_batch.override_req_to_token = original_override_req_to_token
        # Reinitialize attention metadata for caller (uses override if set)
        forward_batch.attn_backend.init_forward_metadata(forward_batch)

        # Cat generated tokens: each is [1], cat to [num_tokens], then add batch dim
        sequences = torch.cat(generated_tokens, dim=0).unsqueeze(0)  # [1, num_tokens]
        return sequences


class Qwen3OmniMoeTalkerForConditionalGeneration(nn.Module):
    """Top-level Talker module with projections, codec_head, model, and code_predictor."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id_offset: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_id_offset = layer_id_offset
        text_config = config.text_config

        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(config)
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(config)

        self.codec_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        self.model = Qwen3MoeModel(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            decoder_layer_type=functools.partial(
                Qwen3MoeDecoderLayer,
                is_layer_sparse=True,
                is_previous_layer_sparse=True,
                sparse_moe_block_type=Qwen2MoeSparseMoeBlock,
            ),
            layer_id_offset=layer_id_offset,
        )

        # Code predictor - offset by talker's num_layers
        code_predictor_offset = layer_id_offset + text_config.num_hidden_layers
        self.code_predictor = Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(
            config=config.code_predictor_config,
            quant_config=quant_config,
            prefix=add_prefix("code_predictor", prefix),
            layer_id_offset=code_predictor_offset,
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        prev_codec_id: int = None,
        prev_residual_codes: List[int] = None,
        trailing_text_hidden: torch.Tensor = None,
        tts_pad_embed: torch.Tensor = None,
        codec_eos_token_id: int = 2150,
    ) -> Tuple[List[int], int]:
        """
        Args:
            positions: Position indices
            forward_batch: Forward batch info
            input_embeds: Pre-computed embeddings (for prefill stage only)
            prev_codec_id: Previous step's first codec token (decode stage)
            prev_residual_codes: Previous step's 15 residual codes (decode stage)
            trailing_text_hidden: Text hidden states from thinker (look-ahead)
            tts_pad_embed: Embedding for tts_pad_token_id (when thinker done)
            codec_eos_token_id: EOS token ID for codec sampling

        Returns:
            codec_frame: Complete 16-code frame [codec, 15 residuals]
            codec_token: The sampled codec token (for checking EOS)
        """
        device = positions.device

        # Prefill stage: input_embeds provided
        if input_embeds is not None:
            hidden_states = self.model(
                input_ids=None,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )
            # Get last position's hidden state and logits
            last_hidden = hidden_states[-1:]  # [1, hidden_dim]

            logits = self.codec_head(last_hidden)  # [1, vocab_size]

            # Sample codec token
            codec_token = _sample_codec_tokens(logits.squeeze(0), codec_eos_token_id).item()

            # Run code predictor: (current_hidden, embed(codec))
            codec_embed = self.model.embed_tokens(
                torch.tensor([codec_token], device=device, dtype=torch.long)
            )

            predictor_input = torch.stack(
                (last_hidden.squeeze(0), codec_embed.squeeze(0)), dim=0
            )  # [2, hidden]
            residual_codes = self.code_predictor.generate(
                input_embeds=predictor_input,
                forward_batch=forward_batch,
                num_tokens=self.config.code_predictor_config.num_code_groups - 1,
                do_sample=True,
                top_k=50,
                top_p=0.8,
            )

            # Build complete frame
            codec_frame = [codec_token] + residual_codes.squeeze(0).tolist()
            return codec_frame, codec_token

        # Decode stage: build input_embeds from prev_codec + prev_residual_codes
        # 1. Build codec embeddings from previous frame
        prev_codec_tensor = torch.tensor([prev_codec_id], device=device, dtype=torch.long)
        codec_hiddens = [self.model.embed_tokens(prev_codec_tensor)]  # [1, hidden]

        # Add embeddings for prev_residual_codes
        for i, code in enumerate(prev_residual_codes):
            embed_layer = self.code_predictor.get_input_embeddings()[i]
            code_tensor = torch.tensor([code], device=device, dtype=torch.long)
            codec_hiddens.append(embed_layer(code_tensor))  # [1, hidden]

        # Sum all codec embeddings
        codec_hiddens_stacked = torch.stack(codec_hiddens, dim=0)  # [16, 1, hidden]
        input_embeds_decode = codec_hiddens_stacked.sum(dim=0)  # [1, hidden]

        # 2. Add trailing_text_hidden or tts_pad_embed
        if trailing_text_hidden is not None:
            input_embeds_decode = input_embeds_decode + trailing_text_hidden
        elif tts_pad_embed is not None:
            input_embeds_decode = input_embeds_decode + tts_pad_embed

        # 3. Run talker model
        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds_decode,
        )

        # 4. Get logits and sample codec token
        logits = self.codec_head(hidden_states)
        codec_token = _sample_codec_tokens(logits.squeeze(0), codec_eos_token_id).item()

        # 5. Run code predictor: (current_hidden, embed(codec))
        codec_embed = self.model.embed_tokens(
            torch.tensor([codec_token], device=device, dtype=torch.long)
        )  # [1, hidden]

        predictor_input = torch.stack(
            (hidden_states.squeeze(0), codec_embed.squeeze(0)), dim=0
        )  # [2, hidden]
        residual_codes = self.code_predictor.generate(
            input_embeds=predictor_input,
            forward_batch=forward_batch,
            num_tokens=self.config.code_predictor_config.num_code_groups - 1,
            do_sample=True,
            top_k=50,
            top_p=0.8,
        )

        # 6. Build complete frame
        codec_frame = [codec_token] + residual_codes.squeeze(0).tolist()
        return codec_frame, codec_token


class Qwen3OmniMoeCausalConvNet(nn.Module):
    """Causal 1D convolution with left-only padding for streaming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state, (self.padding, extra_padding), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()


class Qwen3OmniMoeCausalTransConvNet(nn.Module):
    """Causal transposed 1D convolution for upsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = pad - self.left_pad

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv(hidden_state)
        hidden_state = hidden_state[..., self.left_pad : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


class Qwen3OmniMoeConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with causal depthwise conv."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=7, groups=dim, dilation=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.permute(0, 2, 1)
        return residual + hidden_states


class Qwen3OmniMoeCode2WavRMSNorm(nn.Module):
    """RMSNorm for Code2Wav transformer layers."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3OmniMoeCode2WavLayerScale(nn.Module):
    """Layer scale for residual connections."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class Qwen3OmniMoeCode2WavMlp(nn.Module):
    """MLP for Code2Wav transformer layers."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3OmniMoeCode2WavRotaryEmbedding(nn.Module):
    """Rotary positional embedding for Code2Wav transformer."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to q and k."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3OmniMoeCode2WavAttention(nn.Module):
    """Sliding window causal attention for Code2Wav transformer."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.sliding_window = config.sliding_window

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention with sliding window causal mask
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

        # Create causal + sliding window mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
        if self.sliding_window is not None:
            window_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device), diagonal=-self.sliding_window)
            window_mask = ~window_mask
            causal_mask = causal_mask | window_mask
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class Qwen3OmniMoeCode2WavTransformerLayer(nn.Module):
    """Single transformer layer for Code2Wav."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3OmniMoeCode2WavAttention(config, layer_idx)
        self.mlp = Qwen3OmniMoeCode2WavMlp(config)
        self.input_layernorm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)
        self.mlp_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return hidden_states


class Qwen3OmniMoeCode2WavTransformerModel(nn.Module):
    """Transformer model for Code2Wav (pre_transformer)."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            Qwen3OmniMoeCode2WavTransformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3OmniMoeCode2WavRotaryEmbedding(config)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs_embeds.shape
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        return self.norm(hidden_states)


class SnakeBeta(nn.Module):
    """Snake activation with learnable alpha and beta parameters."""

    def __init__(self, in_features: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        return hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(hidden_states * alpha), 2)


class Qwen3OmniMoeCode2WavDecoderResidualUnit(nn.Module):
    """Residual unit for Code2Wav decoder."""

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class Qwen3OmniMoeCode2WavDecoderBlock(nn.Module):
    """Upsampling decoder block for Code2Wav."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig, layer_idx: int):
        super().__init__()
        in_dim = config.decoder_dim // (2 ** layer_idx)
        out_dim = config.decoder_dim // (2 ** (layer_idx + 1))
        upsample_rate = config.upsample_rates[layer_idx]

        self.block = nn.ModuleList([
            SnakeBeta(in_dim),
            Qwen3OmniMoeCausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ])
        for dilation in (1, 3, 9):
            self.block.append(Qwen3OmniMoeCode2WavDecoderResidualUnit(out_dim, dilation))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            hidden = block(hidden)
        return hidden


class Qwen3OmniMoeCode2Wav(nn.Module):
    """
    Code2Wav: Converts codec codes to waveform.

    Input: codes tensor of shape [batch, num_quantizers, num_frames]
    Output: waveform tensor of shape [batch, 1, num_samples] in range [-1, 1]
    """

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig):
        super().__init__()
        self.config = config
        self.total_upsample = int(np.prod(list(config.upsample_rates) + list(config.upsampling_ratios)))

        # Code embedding: maps each quantizer's codes to hidden_size
        self.code_embedding = nn.Embedding(config.codebook_size * config.num_quantizers, config.hidden_size)
        # Offset buffer: shift codes for each quantizer into its own embedding range
        self.register_buffer(
            "code_offset",
            torch.arange(config.num_quantizers).view(1, -1, 1) * config.codebook_size,
            persistent=False,
        )

        # Pre-transformer: sliding window causal attention
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel(config)

        # Upsample blocks (after transformer, before decoder)
        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(nn.ModuleList([
                Qwen3OmniMoeCausalTransConvNet(config.hidden_size, config.hidden_size, factor, factor),
                Qwen3OmniMoeConvNeXtBlock(config.hidden_size),
            ]))
        self.upsample = nn.ModuleList(upsample)

        # Decoder: progressive upsampling to waveform
        decoder = [Qwen3OmniMoeCausalConvNet(config.hidden_size, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(config, i))
        output_dim = config.decoder_dim // (2 ** len(config.upsample_rates))
        decoder.extend([
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, 7),
        ])
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: codes -> waveform.

        Args:
            codes: [batch, num_quantizers, num_frames] integer codes

        Returns:
            wav: [batch, 1, num_samples] float waveform in [-1, 1]
        """
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} quantizers, got {codes.shape[1]}")

        # Embed codes and average across quantizers
        hidden = self.code_embedding(codes + self.code_offset).mean(1)  # [B, T, hidden_size]

        # Transformer
        hidden = self.pre_transformer(hidden)

        # Permute for conv: [B, T, C] -> [B, C, T]
        hidden = hidden.permute(0, 2, 1)

        # Upsample
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)

        # Decoder
        wav = hidden
        for block in self.decoder:
            wav = block(wav)

        return wav.clamp(min=-1, max=1)

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        """
        Decode codes in chunks with left context overlap for streaming.

        Args:
            codes: [batch, num_quantizers, num_frames]
            chunk_size: number of frames per chunk
            left_context_size: number of frames of left context for overlap

        Returns:
            wav: [batch, 1, num_samples] concatenated waveform
        """
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            # Crop away the left context samples
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    def decode_streaming_chunk(
        self,
        codes: torch.Tensor,
        last_decoded_frame: int,
        left_context_size: int = 25,
    ) -> tuple[torch.Tensor, int]:
        """
        Decode a single streaming chunk from accumulated codes.

        Args:
            codes: [batch, num_quantizers, total_frames_so_far]
            last_decoded_frame: index of last frame that was decoded
            left_context_size: number of frames of left context

        Returns:
            (pcm_chunk, new_last_decoded_frame)
        """
        current_frame = codes.shape[-1]
        if current_frame <= last_decoded_frame:
            return torch.tensor([], device=codes.device), last_decoded_frame

        context_start = max(0, last_decoded_frame - left_context_size)
        context_size = last_decoded_frame - context_start

        codes_chunk = codes[..., context_start:current_frame]
        wav_chunk = self(codes_chunk)

        # Crop away the context samples
        pcm_chunk = wav_chunk[..., context_size * self.total_upsample :]
        return pcm_chunk, current_frame


class Qwen3OmniMoeForConditionalGeneration(PreTrainedModel):
    def __init__(
        self,
        config: Qwen3OmniMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.config = config

        self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration(
            config.thinker_config,
            quant_config=quant_config,
            prefix=prefix,
            accept_hidden_layer=config.talker_config.accept_hidden_layer,
        )

        thinker_num_layers = config.thinker_config.text_config.num_hidden_layers
        # Talker is not quantized in GPTQ checkpoints - use bf16
        self.talker = Qwen3OmniMoeTalkerForConditionalGeneration(
            config=config.talker_config,
            quant_config=None,
            prefix="talker",
            layer_id_offset=thinker_num_layers,
        )

        # Code2Wav: codec codes to waveform decoder
        self.code2wav = Qwen3OmniMoeCode2Wav(config.code2wav_config)

        self.pad_input_ids = self.thinker.pad_input_ids
        self.accept_hidden_layer = config.talker_config.accept_hidden_layer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> Union[torch.Tensor, ThinkerTalkerOutput]:
        """
        Unified forward pass for Qwen3-Omni with parallel thinker + talker.

        Always runs both thinker and talker together for audio output.
        """
        is_prefill = forward_batch.forward_mode.is_extend()

        if is_prefill:
            return self._forward_prefill(input_ids, positions, forward_batch, input_embeds)
        else:
            return self._forward_decode(input_ids, positions, forward_batch)

    def _forward_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> ThinkerTalkerOutput:
        """Prefill both thinker and talker in parallel."""
        # 0. Process multimodal inputs (audio/image) if present
        # This computes embeddings and clamps input_ids to valid vocab range
        if input_embeds is None and forward_batch.contains_mm_inputs():
            mm_inputs_list = [
                mm_input
                for mm_input in forward_batch.mm_inputs
                if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            input_embeds, _ = embed_mm_inputs(
                mm_inputs_list=mm_inputs_list,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
                input_ids=input_ids,
                multimodal_model=self.thinker,
                input_embedding=self.thinker.model.embed_tokens,
            )
            forward_batch.mm_inputs = None

        # 1. Run thinker forward - model returns (hidden_states, aux_hidden_states)
        #    because layers_to_capture is set
        thinker_result = self.thinker.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        # Unpack result - with layers_to_capture, returns tuple
        if isinstance(thinker_result, tuple):
            thinker_hidden, aux_hidden_states = thinker_result
            accept_hidden = aux_hidden_states[0]  # accept_hidden_layer (e.g., layer 24) hidden states
        else:
            thinker_hidden = thinker_result
            accept_hidden = None

        # 2. Get thinker logits
        thinker_logits = self.thinker.logits_processor(
            input_ids, thinker_hidden, self.thinker.lm_head, forward_batch
        )

        # 3. Get embeddings for talker input preparation
        # IMPORTANT: Use full input_ids from request, not the extend-only input_ids
        # On cache hits, input_ids only contains new tokens, but talker needs full sequence
        full_input_ids = forward_batch.model_specific_states["origin_input_ids"][0]
        # Clamp to valid vocab range (multimodal placeholders may have hash values outside vocab)
        vocab_size = self.thinker.model.embed_tokens.num_embeddings
        full_input_ids = full_input_ids.clamp(min=0, max=vocab_size - 1)
        thinker_embeds = self.thinker.model.embed_tokens(full_input_ids)

        # 4. Compute tts_pad_embed for later use (when thinker finishes)
        # tts_pad is a TEXT token - embed with thinker, then project through talker
        tts_pad_token_id = self.config.tts_pad_token_id
        tts_pad_text_embed = self.thinker.model.embed_tokens(
            torch.tensor([tts_pad_token_id], device=input_ids.device)
        )
        tts_pad_embed = self.talker.text_projection(tts_pad_text_embed).squeeze(0)  # [hidden_dim]

        # 5. With 1-step delay: DON'T run talker prefill here
        # Talker prefill will run at first decode step when we have actual sampled token
        # Set talker_needs_prefill=True to signal scheduler
        return ThinkerTalkerOutput(
            thinker_logits=thinker_logits,
            tts_pad_embed=tts_pad_embed,
            talker_needs_prefill=True,  # Signal to run talker prefill at first decode
        )

    def _save_forward_batch_state(
        self, forward_batch: ForwardBatch, save_extend: bool = False
    ) -> dict:
        """Save forward_batch state for later restoration after talker forward.

        Since we use override_req_to_token instead of mutating the shared pool,
        we only need to save batch dimensions (no req_to_token cloning needed).
        """
        state = {
            "batch_size": forward_batch.batch_size,
            "req_pool_indices": forward_batch.req_pool_indices.clone(),
            "seq_lens": forward_batch.seq_lens,
            "seq_lens_cpu": forward_batch.seq_lens_cpu,
            "seq_lens_sum": forward_batch.seq_lens_sum,
            "out_cache_loc": forward_batch.out_cache_loc,
        }

        if save_extend:
            state.update({
                "forward_mode": forward_batch.forward_mode,
                "extend_prefix_lens_cpu": forward_batch.extend_prefix_lens_cpu,
                "extend_seq_lens_cpu": forward_batch.extend_seq_lens_cpu,
                "extend_prefix_lens": forward_batch.extend_prefix_lens,
                "extend_seq_lens": forward_batch.extend_seq_lens,
                "extend_num_tokens": forward_batch.extend_num_tokens,
                "extend_start_loc": forward_batch.extend_start_loc,
            })

        return state

    def _restore_forward_batch_state(
        self, forward_batch: ForwardBatch, saved_state: dict
    ) -> None:
        """Restore forward_batch state after talker forward.

        Since we use override_req_to_token instead of mutating the shared pool,
        we just need to clear the override and restore batch dimensions.
        """
        # Clear the override (no pool mutation to undo)
        forward_batch.override_req_to_token = None

        # Restore thinker's batch dimensions
        forward_batch.batch_size = saved_state["batch_size"]
        forward_batch.req_pool_indices = saved_state["req_pool_indices"]
        forward_batch.seq_lens = saved_state["seq_lens"]
        forward_batch.seq_lens_cpu = saved_state["seq_lens_cpu"]
        forward_batch.seq_lens_sum = saved_state["seq_lens_sum"]
        forward_batch.out_cache_loc = saved_state["out_cache_loc"]

        if "forward_mode" in saved_state:
            forward_batch.forward_mode = saved_state["forward_mode"]
            forward_batch.extend_prefix_lens_cpu = saved_state["extend_prefix_lens_cpu"]
            forward_batch.extend_seq_lens_cpu = saved_state["extend_seq_lens_cpu"]
            forward_batch.extend_prefix_lens = saved_state["extend_prefix_lens"]
            forward_batch.extend_seq_lens = saved_state["extend_seq_lens"]
            forward_batch.extend_num_tokens = saved_state["extend_num_tokens"]
            forward_batch.extend_start_loc = saved_state["extend_start_loc"]

        # Reinitialize attention metadata for thinker (no override)
        forward_batch.attn_backend.init_forward_metadata(forward_batch)

    def _switch_to_talker_kv_cache_batched(
        self, forward_batch: ForwardBatch, model_states: dict
    ) -> List[torch.Tensor]:
        """Switch forward_batch to talker KV caches using override (no pool mutation).

        Uses forward_batch.override_req_to_token instead of mutating the shared
        req_to_token_pool. This enables overlap scheduling by avoiding race conditions.

        Returns:
            updated_kv_locs_list: List of updated KV cache locations per request
        """
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        allocator = forward_batch.token_to_kv_pool_allocator

        talker_kv_locs_list = model_states.get("talker_kv_cache_locs_list", [])
        new_seq_lens = []
        new_out_cache_locs = []
        updated_kv_locs_list = []

        # Allocate new KV slots and build updated locations
        for i in range(batch_size):
            talker_kv_locs = talker_kv_locs_list[i] if i < len(talker_kv_locs_list) else None

            if talker_kv_locs is not None:
                # Allocate new KV slot for this decode step
                new_loc = allocator.alloc(1)
                updated_locs = torch.cat([talker_kv_locs, new_loc])
                new_seq_lens.append(len(updated_locs))
                new_out_cache_locs.append(new_loc)
                updated_kv_locs_list.append(updated_locs)
            else:
                # No talker KV cache yet (shouldn't happen in decode)
                new_loc = allocator.alloc(1)
                new_seq_lens.append(1)
                new_out_cache_locs.append(new_loc)
                updated_kv_locs_list.append(new_loc)

        # Build small temporary tensor for talker KV locations (NOT mutating shared pool)
        max_len = max(new_seq_lens)
        talker_req_to_token = torch.zeros(
            batch_size, max_len, dtype=torch.int32, device=device
        )
        for i, locs in enumerate(updated_kv_locs_list):
            talker_req_to_token[i, : len(locs)] = locs

        # Set override for attention backend (avoids mutating shared req_to_token_pool)
        forward_batch.override_req_to_token = talker_req_to_token

        # Update forward_batch with batched values
        forward_batch.seq_lens = torch.tensor(new_seq_lens, dtype=torch.int32, device=device)
        forward_batch.seq_lens_cpu = torch.tensor(new_seq_lens, dtype=torch.int32)
        forward_batch.seq_lens_sum = sum(new_seq_lens)
        forward_batch.out_cache_loc = torch.cat(new_out_cache_locs)

        # Reinitialize attention metadata (will use override_req_to_token)
        forward_batch.attn_backend.init_forward_metadata(forward_batch)

        return updated_kv_locs_list

    def _forward_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> ThinkerTalkerOutput:
        """Decode both thinker and talker with 1-step delay. Supports batching.

        With 1-step delay design:
        - Talker runs 1 step behind thinker
        - At first decode: run talker prefill using prev_sampled_thinker_token
        - At subsequent decodes: run talker decode using prev_sampled_thinker_token for look-ahead
        """
        model_states = forward_batch.model_specific_states or {}
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        # === 1. Check request states ===
        thinker_done_tensor = model_states.get("thinker_done")
        talker_needs_prefill_tensor = model_states.get("talker_needs_prefill")

        # Check if ALL requests have thinker done -> talker-only mode
        if thinker_done_tensor is not None and thinker_done_tensor.all():
            return self._forward_talker_only(forward_batch)

        # Check if any request needs talker prefill (TODO: handle mixed prefill/decode)
        if talker_needs_prefill_tensor is not None and talker_needs_prefill_tensor.any():
            return self._run_talker_prefill_at_decode(
                thinker_logits=None,
                forward_batch=forward_batch,
                model_states=model_states,
            )

        # === 2. Run thinker decode on full batch ===
        thinker_result = self.thinker.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
        )

        if isinstance(thinker_result, tuple):
            thinker_hidden, aux_hidden_states = thinker_result
        else:
            thinker_hidden = thinker_result

        thinker_logits = self.thinker.logits_processor(
            input_ids, thinker_hidden, self.thinker.lm_head, forward_batch
        )

        # === 3. Get batched talker state ===
        prev_codec_ids = model_states.get("prev_codec_id")  # [batch_size]
        prev_residual_codes = model_states.get("prev_residual_codes")  # [batch_size, 15]
        tts_pad_embed = model_states.get("tts_pad_embed")  # [batch_size, hidden] or None
        talker_positions = model_states.get("talker_positions", positions)  # [batch_size]
        prev_sampled_tokens = model_states.get("prev_sampled_thinker_token")  # [batch_size]
        codec_eos_token_id = model_states.get("codec_eos_token_id", 2150)

        if prev_codec_ids is None or prev_residual_codes is None:
            # No previous codec frame yet - shouldn't happen after prefill
            return ThinkerTalkerOutput(thinker_logits=thinker_logits)

        # === 4. Build batched talker input_embeds: [batch_size, hidden] ===
        # Embed prev_codec_ids: [batch_size, hidden]
        codec_embed = self.talker.model.embed_tokens(prev_codec_ids)

        # Embed prev_residual_codes and sum: [batch_size, hidden]
        residual_sum = torch.zeros_like(codec_embed)
        for i in range(15):
            embed_layer = self.talker.code_predictor.get_input_embeddings()[i]
            residual_sum = residual_sum + embed_layer(prev_residual_codes[:, i])

        # Compute trailing_text_hidden: [batch_size, hidden]
        # For requests with thinker_done, use tts_pad_embed instead of trailing_text_hidden
        if prev_sampled_tokens is not None:
            prev_embeds = self.thinker.model.embed_tokens(prev_sampled_tokens)
            trailing_text_hidden = self.talker.text_projection(prev_embeds)
        else:
            trailing_text_hidden = None

        # Combine: [batch_size, hidden]
        input_embeds = codec_embed + residual_sum

        # Handle mixed thinker_done states: use tts_pad_embed for done requests
        if thinker_done_tensor is not None and thinker_done_tensor.any() and not thinker_done_tensor.all():
            # Mixed case: some done, some not
            for i in range(batch_size):
                if thinker_done_tensor[i]:
                    # Use tts_pad_embed for this request
                    if tts_pad_embed is not None and tts_pad_embed.dim() == 2:
                        input_embeds[i] = input_embeds[i] + tts_pad_embed[i]
                else:
                    # Use trailing_text_hidden for this request
                    if trailing_text_hidden is not None:
                        input_embeds[i] = input_embeds[i] + trailing_text_hidden[i]
        elif trailing_text_hidden is not None:
            # All requests use trailing_text_hidden (none done)
            input_embeds = input_embeds + trailing_text_hidden

        # === 5. Switch forward_batch to talker KV cache (batched) ===
        saved_state = self._save_forward_batch_state(forward_batch)
        updated_kv_locs_list = self._switch_to_talker_kv_cache_batched(forward_batch, model_states)

        # === 6. Run talker.model ONCE for all requests ===
        hidden_states = self.talker.model(
            input_ids=None,
            positions=talker_positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )  # [batch_size, hidden]

        # === 7. Sample codec tokens (batched) ===
        logits = self.talker.codec_head(hidden_states)  # [batch_size, vocab]
        codec_tokens = _sample_codec_tokens(logits, codec_eos_token_id)

        # === 8. Run code predictor per-request (loop) ===
        codec_frames = []
        for i in range(batch_size):
            codec_embed_i = self.talker.model.embed_tokens(codec_tokens[i:i+1])
            predictor_input = torch.stack([hidden_states[i], codec_embed_i.squeeze(0)], dim=0)
            residual_codes = self.talker.code_predictor.generate(
                input_embeds=predictor_input,
                forward_batch=forward_batch,
                num_tokens=self.talker.config.code_predictor_config.num_code_groups - 1,
                do_sample=True,
                top_k=50,
                top_p=0.8,
            )
            frame = [codec_tokens[i].item()] + residual_codes.squeeze(0).tolist()
            codec_frames.append(frame)

        # === 9. Restore forward_batch state ===
        self._restore_forward_batch_state(forward_batch, saved_state)

        return ThinkerTalkerOutput(
            thinker_logits=thinker_logits,
            codec_frames=codec_frames,
            updated_talker_kv_locs_list=updated_kv_locs_list,
        )

    def _run_talker_prefill_at_decode(
        self,
        thinker_logits: Optional[LogitsProcessorOutput],
        forward_batch: ForwardBatch,
        model_states: dict,
    ) -> ThinkerTalkerOutput:
        """Run talker prefill at first decode step (1-step delay). Supports batching.

        Uses prev_sampled_thinker_token (token 0) as first response token,
        and current thinker input_ids (token 1) as look-ahead.

        With new design: talker prefill samples codec and runs code predictor,
        returning complete first frame.
        """
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        prev_sampled_token = model_states.get("prev_sampled_thinker_token")
        tts_pad_embed = model_states.get("tts_pad_embed")
        codec_eos_token_id = model_states.get("codec_eos_token_id", 2150)

        # Get full input_ids for ChatML parsing
        origin_input_ids = model_states.get("origin_input_ids")
        if origin_input_ids is None:
            return ThinkerTalkerOutput(thinker_logits=thinker_logits)

        # First run thinker decode if thinker_logits not provided
        if thinker_logits is None:
            thinker_result = self.thinker.model(
                input_ids=forward_batch.input_ids,
                positions=forward_batch.positions,
                forward_batch=forward_batch,
            )
            if isinstance(thinker_result, tuple):
                thinker_hidden, _ = thinker_result
            else:
                thinker_hidden = thinker_result

            thinker_logits = self.thinker.logits_processor(
                forward_batch.input_ids, thinker_hidden, self.thinker.lm_head, forward_batch
            )

        # === Save thinker's state ===
        saved_state = self._save_forward_batch_state(forward_batch, save_extend=True)

        allocator = forward_batch.token_to_kv_pool_allocator
        req_to_token_pool = forward_batch.req_to_token_pool
        vocab_size = self.thinker.model.embed_tokens.num_embeddings

        # Save original req_pool_indices before we modify it in the loop
        original_req_pool_indices = saved_state["req_pool_indices"]

        # Process each request's talker prefill
        codec_frames = []
        talker_out_cache_locs = []
        tts_pad_embeds_out = []

        for i in range(batch_size):
            full_input_ids = origin_input_ids[i]
            full_input_ids = full_input_ids.clamp(min=0, max=vocab_size - 1)
            first_token = prev_sampled_token[i] if prev_sampled_token is not None else None


            # Compute thinker_embeds for building talker input
            thinker_embeds = self.thinker.model.embed_tokens(full_input_ids)
            talker_input_embeds, tts_pad_embed_new = self._prepare_talker_prefill_1step(
                thinker_embeds=thinker_embeds,
                input_ids=full_input_ids,
                first_response_token=first_token,
            )

            # Use new tts_pad_embed if computed
            req_tts_pad_embed = tts_pad_embed_new if tts_pad_embed_new is not None else (
                tts_pad_embed[i] if tts_pad_embed is not None and tts_pad_embed.dim() == 2 else tts_pad_embed
            )
            tts_pad_embeds_out.append(req_tts_pad_embed)

            # Build positions for talker prefill
            talker_seq_len = talker_input_embeds.shape[0]
            talker_positions = torch.arange(talker_seq_len, device=device, dtype=torch.long)

            # Get this request's pool index from the ORIGINAL saved indices
            req_idx = original_req_pool_indices[i].item()

            # Allocate KV cache for talker prefill
            talker_out_cache_loc = allocator.alloc(talker_seq_len)
            talker_out_cache_locs.append(talker_out_cache_loc)

            # Build override tensor for this single request (avoids mutating shared pool)
            talker_req_to_token = torch.zeros(
                1, talker_seq_len, dtype=torch.int32, device=device
            )
            talker_req_to_token[0, :talker_seq_len] = talker_out_cache_loc
            forward_batch.override_req_to_token = talker_req_to_token

            forward_batch.batch_size = 1
            forward_batch.req_pool_indices = torch.tensor([0], device=device, dtype=torch.int32)  # Use 0 for override
            forward_batch.out_cache_loc = talker_out_cache_loc
            forward_batch.seq_lens = torch.tensor([talker_seq_len], device=device, dtype=torch.int32)
            forward_batch.seq_lens_cpu = [talker_seq_len]
            forward_batch.seq_lens_sum = talker_seq_len
            forward_batch.extend_prefix_lens_cpu = [0]
            forward_batch.extend_seq_lens_cpu = [talker_seq_len]
            forward_batch.extend_prefix_lens = torch.tensor([0], device=device, dtype=torch.int32)
            forward_batch.extend_seq_lens = torch.tensor([talker_seq_len], device=device, dtype=torch.int32)
            forward_batch.extend_num_tokens = talker_seq_len
            forward_batch.extend_start_loc = torch.tensor([0], device=device, dtype=torch.int32)
            forward_batch.forward_mode = ForwardMode.EXTEND

            forward_batch.attn_backend.init_forward_metadata(forward_batch)

            # Run talker prefill for this request
            codec_frame, codec_token = self.talker.forward(
                positions=talker_positions,
                forward_batch=forward_batch,
                input_embeds=talker_input_embeds,
                codec_eos_token_id=codec_eos_token_id,
            )
            codec_frames.append(codec_frame)

        # === Restore thinker's state ===
        self._restore_forward_batch_state(forward_batch, saved_state)

        # Stack tts_pad_embeds if all valid
        tts_pad_embed_out = None
        if all(e is not None for e in tts_pad_embeds_out):
            tts_pad_embed_out = torch.stack(tts_pad_embeds_out, dim=0)

        return ThinkerTalkerOutput(
            thinker_logits=thinker_logits,
            codec_frames=codec_frames,
            tts_pad_embed=tts_pad_embed_out,
            talker_out_cache_loc_list=talker_out_cache_locs if talker_out_cache_locs else None,
        )

    def _forward_talker_only(
        self,
        forward_batch: ForwardBatch,
    ) -> ThinkerTalkerOutput:
        """Continue talker generation after thinker has finished (hit EOS). Supports batching.

        When thinker is done, talker continues to generate audio using tts_pad_embed
        instead of trailing_text_hidden. This uses the batched helper functions.
        """
        model_states = forward_batch.model_specific_states or {}
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        # Get batched state
        prev_codec_ids = model_states.get("prev_codec_id")  # [batch_size]
        prev_residual_codes = model_states.get("prev_residual_codes")  # [batch_size, 15]
        tts_pad_embed = model_states.get("tts_pad_embed")  # [batch_size, hidden]
        talker_positions = model_states.get("talker_positions")  # [batch_size]
        codec_eos_token_id = model_states.get("codec_eos_token_id", 2150)

        if prev_codec_ids is None or prev_residual_codes is None:
            return ThinkerTalkerOutput(thinker_logits=None)

        # === 1. Build batched talker input_embeds: [batch_size, hidden] ===
        # Embed prev_codec_ids: [batch_size, hidden]
        codec_embed = self.talker.model.embed_tokens(prev_codec_ids)

        # Embed prev_residual_codes and sum: [batch_size, hidden]
        residual_sum = torch.zeros_like(codec_embed)
        for i in range(15):
            embed_layer = self.talker.code_predictor.get_input_embeddings()[i]
            residual_sum = residual_sum + embed_layer(prev_residual_codes[:, i])

        # Use tts_pad_embed instead of trailing_text_hidden (thinker is done)
        input_embeds = codec_embed + residual_sum
        if tts_pad_embed is not None:
            input_embeds = input_embeds + tts_pad_embed

        # === 2. Switch forward_batch to talker KV cache (batched) ===
        saved_state = self._save_forward_batch_state(forward_batch)
        updated_kv_locs_list = self._switch_to_talker_kv_cache_batched(forward_batch, model_states)

        # === 3. Run talker.model ONCE for all requests ===
        hidden_states = self.talker.model(
            input_ids=None,
            positions=talker_positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )  # [batch_size, hidden]

        # === 4. Sample codec tokens (batched) ===
        logits = self.talker.codec_head(hidden_states)  # [batch_size, vocab]
        codec_tokens = _sample_codec_tokens(logits, codec_eos_token_id)

        # === 5. Run code predictor per-request (loop) ===
        codec_frames = []
        for i in range(batch_size):
            codec_embed_i = self.talker.model.embed_tokens(codec_tokens[i : i + 1])
            predictor_input = torch.stack(
                [hidden_states[i], codec_embed_i.squeeze(0)], dim=0
            )

            residual_codes = self.talker.code_predictor.generate(
                input_embeds=predictor_input,
                forward_batch=forward_batch,
                num_tokens=self.talker.config.code_predictor_config.num_code_groups - 1,
                do_sample=True,
                top_k=50,
                top_p=0.8,
            )
            frame = [codec_tokens[i].item()] + residual_codes.squeeze(0).tolist()
            codec_frames.append(frame)

        # === 6. Restore forward_batch state ===
        self._restore_forward_batch_state(forward_batch, saved_state)

        return ThinkerTalkerOutput(
            thinker_logits=None,  # Thinker is done
            codec_frames=codec_frames,
            updated_talker_kv_locs_list=updated_kv_locs_list,
        )

    def _prepare_talker_prefill_1step(
        self,
        thinker_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        first_response_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build talker prefill input with actual sampled first response token.

        This is a simplified version of _prepare_talker_prefill for 1-step delay.
        Uses the actual sampled token (first_response_token) instead of placeholder.

        Returns:
            talker_input_embeds: Embeddings for talker prefill [seq_len, hidden_dim]
            tts_pad_embed: Padding embed for when thinker is done [hidden_dim]
        """
        device = input_ids.device
        dtype = thinker_embeds.dtype
        talker_hidden_size = self.config.talker_config.text_config.hidden_size

        # Config values
        im_start_token_id = self.config.im_start_token_id
        system_token_id = self.config.system_token_id
        user_token_id = self.config.user_token_id
        assistant_token_id = self.config.assistant_token_id
        tts_bos_token_id = self.config.tts_bos_token_id
        tts_pad_token_id = self.config.tts_pad_token_id

        talker_config = self.config.talker_config
        codec_nothink_id = talker_config.codec_nothink_id
        codec_think_bos_id = talker_config.codec_think_bos_id
        codec_think_eos_id = talker_config.codec_think_eos_id
        codec_pad_id = talker_config.codec_pad_id
        codec_bos_id = talker_config.codec_bos_id
        speaker_id = talker_config.speaker_id.get("ethan", 2302)

        # Get tts special token embeddings
        tts_special_tokens = torch.tensor(
            [tts_bos_token_id, tts_pad_token_id], device=device, dtype=torch.long
        )
        tts_special_embeds = self.thinker.model.embed_tokens(tts_special_tokens)
        tts_projected = self.talker.text_projection(tts_special_embeds)
        tts_bos_embed = tts_projected[0:1]
        tts_pad_embed = tts_projected[1:2]

        # Find all <|im_start|> positions
        im_start_mask = input_ids == im_start_token_id
        im_start_positions = torch.nonzero(im_start_mask, as_tuple=True)[0]
        im_start_indexes = torch.cat([
            im_start_positions,
            torch.tensor([input_ids.shape[0]], device=device, dtype=im_start_positions.dtype)
        ])

        talker_input_parts = []
        num_segments = len(im_start_indexes) - 1

        for i in range(num_segments):
            im_start_index = im_start_indexes[i].item()
            segment_end_index = im_start_indexes[i + 1].item()
            role_token = input_ids[im_start_index + 1].item()

            if role_token == system_token_id:
                continue

            elif role_token == user_token_id:
                user_embeds = thinker_embeds[im_start_index:segment_end_index]
                user_projected = self.talker.text_projection(user_embeds)
                talker_input_parts.append(user_projected)

            elif role_token == assistant_token_id and i == num_segments - 1:
                assistant_embeds = thinker_embeds[im_start_index:segment_end_index]
                assistant_hidden = self.talker.text_projection(assistant_embeds)

                # Use actual sampled token for first_text_embed (key difference from original)
                if first_response_token is not None:
                    first_token_embed = self.thinker.model.embed_tokens(first_response_token.unsqueeze(0))
                    first_text_embed = self.talker.text_projection(first_token_embed)
                else:
                    first_text_embed = tts_pad_embed

                # Build text side: [header, 4x pad, bos, first_text]
                assistant_text_hidden = torch.cat([
                    assistant_hidden[:3],
                    tts_pad_embed.expand(4, -1),
                    tts_bos_embed,
                    first_text_embed,
                ], dim=0)

                # Build codec side
                codec_special_tokens = torch.tensor(
                    [codec_nothink_id, codec_think_bos_id, codec_think_eos_id,
                     speaker_id, codec_pad_id, codec_bos_id],
                    device=device, dtype=torch.long
                )
                codec_embeds = self.talker.model.embed_tokens(codec_special_tokens)
                assistant_codec_hidden = torch.cat([
                    torch.zeros(3, talker_hidden_size, device=device, dtype=dtype),
                    codec_embeds,
                ], dim=0)

                assistant_input = assistant_text_hidden + assistant_codec_hidden
                talker_input_parts.append(assistant_input)

            elif role_token == assistant_token_id:
                continue

        if not talker_input_parts:
            talker_input_embeds = self.talker.text_projection(thinker_embeds)
            return talker_input_embeds, tts_pad_embed.squeeze(0)

        talker_input_embeds = torch.cat(talker_input_parts, dim=0)
        return talker_input_embeds, tts_pad_embed.squeeze(0)

    def _prepare_talker_prefill(
        self,
        thinker_embeds: torch.Tensor,
        accept_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build talker prefill input from thinker outputs.

        Follows the transformers implementation:
        1. Parse ChatML segments from input_ids
        2. Skip system parts
        3. For user parts: text_projection(thinker_embeds)
        4. For assistant parts: complex structure with text+codec embeddings (element-wise add)

        Returns:
            talker_input_embeds: Embeddings for talker prefill [seq_len, hidden_dim]
            trailing_text_hidden: Projected hidden states for decode [seq_len, hidden_dim]
            tts_pad_embed: Padding embed for when thinker is done [1, hidden_dim]
        """
        device = input_ids.device
        dtype = thinker_embeds.dtype
        talker_hidden_size = self.config.talker_config.text_config.hidden_size

        # Config values (read directly, no getattr)
        im_start_token_id = self.config.im_start_token_id
        system_token_id = self.config.system_token_id
        user_token_id = self.config.user_token_id
        assistant_token_id = self.config.assistant_token_id
        tts_bos_token_id = self.config.tts_bos_token_id
        tts_eos_token_id = self.config.tts_eos_token_id
        tts_pad_token_id = self.config.tts_pad_token_id

        talker_config = self.config.talker_config
        codec_nothink_id = talker_config.codec_nothink_id
        codec_think_bos_id = talker_config.codec_think_bos_id
        codec_think_eos_id = talker_config.codec_think_eos_id
        codec_pad_id = talker_config.codec_pad_id
        codec_bos_id = talker_config.codec_bos_id
        # Default speaker: ethan (2302)
        speaker_id = talker_config.speaker_id.get("ethan", 2302)

        # Get tts special token embeddings from THINKER, then project with talker's text_projection
        tts_special_tokens = torch.tensor(
            [tts_bos_token_id, tts_eos_token_id, tts_pad_token_id],
            device=device,
            dtype=torch.long,
        )
        tts_special_embeds = self.thinker.model.embed_tokens(tts_special_tokens)
        tts_projected = self.talker.text_projection(tts_special_embeds)
        tts_bos_embed = tts_projected[0:1]  # [1, hidden_dim]
        tts_eos_embed = tts_projected[1:2]  # [1, hidden_dim]
        tts_pad_embed = tts_projected[2:3]  # [1, hidden_dim]

        # Find all <|im_start|> positions
        im_start_mask = input_ids == im_start_token_id
        im_start_positions = torch.nonzero(im_start_mask, as_tuple=True)[0]
        # Append sequence end position
        im_start_indexes = torch.cat([
            im_start_positions,
            torch.tensor([input_ids.shape[0]], device=device, dtype=im_start_positions.dtype)
        ])

        talker_input_parts = []
        trailing_text_hidden = None

        # Process each ChatML segment
        num_segments = len(im_start_indexes) - 1
        for i in range(num_segments):
            im_start_index = im_start_indexes[i].item()
            segment_end_index = im_start_indexes[i + 1].item()
            role_token = input_ids[im_start_index + 1].item()

            # Skip system parts
            if role_token == system_token_id:
                continue

            # User parts: text_projection(thinker_embeds)
            elif role_token == user_token_id:
                user_embeds = thinker_embeds[im_start_index:segment_end_index]
                user_projected = self.talker.text_projection(user_embeds)
                talker_input_parts.append(user_projected)

            # Last assistant part: complex structure
            elif role_token == assistant_token_id and i == num_segments - 1:
                # Project assistant embeddings with text_projection
                assistant_embeds = thinker_embeds[im_start_index:segment_end_index]
                assistant_hidden = self.talker.text_projection(assistant_embeds)
                segment_len = segment_end_index - im_start_index

                # Build text side:
                # [<|im_start|>assistant\n (3 tokens), 4x tts_pad, tts_bos, first_text_token]
                # In SGLang streaming, segment_len may be <= 3 (no response tokens yet)
                # Use tts_pad_embed as placeholder for first_text if no response tokens exist
                if segment_len > 3:
                    first_text_embed = assistant_hidden[3:4]  # Pre-computed from thinker_embed
                else:
                    first_text_embed = tts_pad_embed  # Placeholder
                assistant_text_hidden = torch.cat([
                    assistant_hidden[:3],                          # <|im_start|>assistant\n
                    tts_pad_embed.expand(4, -1),                    # 4 padding embeddings
                    tts_bos_embed,                                  # TTS BOS
                    first_text_embed,                              # First text token or placeholder
                ], dim=0)  # [9, hidden_dim]

                # Build codec side:
                # [3x zeros, codec_nothink, codec_think_bos, codec_think_eos, speaker_id, codec_pad, codec_bos]
                codec_special_tokens = torch.tensor(
                    [codec_nothink_id, codec_think_bos_id, codec_think_eos_id,
                     speaker_id, codec_pad_id, codec_bos_id],
                    device=device,
                    dtype=torch.long,
                )
                codec_embeds = self.talker.model.embed_tokens(codec_special_tokens)  # [6, hidden_dim]
                assistant_codec_hidden = torch.cat([
                    torch.zeros(3, talker_hidden_size, device=device, dtype=dtype),
                    codec_embeds,
                ], dim=0)  # [9, hidden_dim]

                # Element-wise add (not concatenate!)
                assistant_input = assistant_text_hidden + assistant_codec_hidden
                talker_input_parts.append(assistant_input)

                # Build trailing_text_hidden for decode phase
                if segment_len > 4:
                    trailing_text_hidden = torch.cat([
                        assistant_hidden[4:],  # Remaining text tokens after first
                        tts_eos_embed,
                    ], dim=0)
                else:
                    # No remaining text tokens, just use tts_eos_embed
                    trailing_text_hidden = tts_eos_embed

            # Skip history assistant parts
            elif role_token == assistant_token_id:
                continue

        # Fallback for non-ChatML inputs (e.g., warmup requests)
        if not talker_input_parts:
            # Simple fallback: project all embeddings with text_projection
            # and create dummy trailing_text_hidden
            talker_input_embeds = self.talker.text_projection(thinker_embeds)
            trailing_text_hidden = tts_eos_embed
            return talker_input_embeds, trailing_text_hidden, tts_pad_embed

        # Concatenate all parts
        talker_input_embeds = torch.cat(talker_input_parts, dim=0)

        return talker_input_embeds, trailing_text_hidden, tts_pad_embed

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]

        # Get num_experts for thinker and talker (they may differ)
        thinker_num_experts = self.config.thinker_config.text_config.num_experts
        talker_num_experts = self.config.talker_config.text_config.num_experts

        thinker_expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=thinker_num_experts,
        )

        talker_expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=talker_num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
            # GPTQ-specific suffixes for non-quantized layers (e.g., mlp.gate)
            ".g_idx",
            ".qweight",
            ".qzeros",
            ".scales",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        # Cache params_dict to avoid repeated expensive traversal of model parameters
        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict

        for name, loaded_weight in weights:
            name = name.replace(r"model.language_model.", r"model.")

            # Code2Wav weights: load directly without sharding
            if "code2wav" in name:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            # Rename codec_embedding to embed_tokens for talker model and code predictor
            if "talker.model.codec_embedding" in name:
                name = name.replace("talker.model.codec_embedding", "talker.model.embed_tokens")
            if "talker.code_predictor.model.codec_embedding" in name:
                name = name.replace(
                    "talker.code_predictor.model.codec_embedding",
                    "talker.code_predictor.model.embed_tokens",
                )

            # Determine if this is a talker weight and select appropriate expert mapping
            is_talker_weight = "talker" in name
            expert_params_mapping = (
                talker_expert_params_mapping if is_talker_weight else thinker_expert_params_mapping
            )
            num_experts = talker_num_experts if is_talker_weight else thinker_num_experts

            name = name.replace(".self_attn.out_proj", ".self_attn.proj")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Check for fused experts: ".experts.gate_up_proj" or ".experts.down_proj"
                # Use ".experts." prefix to avoid matching "shared_expert.down_proj"
                if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                # [TODO] Skip layers that are on other devices (check if sglang has a similar function)
                # if is_pp_missing_parameter(name, self):
                #     continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or "audio_tower" in name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        loaded_weight = loaded_weight.transpose(-1, -2)  # no bias
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        if name_mapped in params_dict.keys():
                            param = params_dict[name_mapped]
                        else:
                            continue
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue
                    if "visual" in name or "audio_tower" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                        name = name.replace(r"model.visual.", r"visual.")
                        name = name.replace(r"attn.out_proj.", r"attn.proj.")

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(
                            f"Loaded weight with {name=} not found in params_dict"
                        )


EntryClass = Qwen3OmniMoeForConditionalGeneration
