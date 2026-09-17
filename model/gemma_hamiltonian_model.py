"""
Trained-from-scratch Gemma 3 backbone + Hamiltonian-conditioned wrapper,
reproducing Section 3.2.3 of ADAPT-GQE (arXiv:2607.22468).

VALIDATION WARNING: build_gemma3_backbone() has NOT been exercised against a
live `transformers` install in this environment. Gemma 3's config/model
class names have shifted during rollout across transformers versions
(Gemma3Config vs Gemma3TextConfig, vision-inclusive vs text-only variants).
If the import in build_gemma3_backbone() fails, check
`transformers.models.gemma3` in your installed version and fix the two
lines there -- everything else in this module only depends on the returned
backbone exposing `get_input_embeddings()` and `forward(inputs_embeds=...,
attention_mask=..., labels=...)`, which is standard across all HF
CausalLM models and should not need changes.
"""
import torch
import torch.nn as nn


def build_gemma3_backbone(
    vocab_size,
    hidden_size,
    num_layers,
    num_heads,
    num_kv_heads,
    context_length,
    sliding_window,
    rope_theta=10_000,
    rms_norm_eps=1e-6,
    dropout=0.0,
):
    """Construct a randomly-initialized, text-only Gemma 3 backbone sized
    per Appendix Table 6 of ADAPT-GQE. See module docstring for the
    version-compatibility caveat on the two import/config lines below."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    config = Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=hidden_size // num_heads,
        max_position_embeddings=context_length,
        sliding_window=sliding_window,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        attention_dropout=dropout,
    )
    return Gemma3ForCausalLM(config)


class HamiltonianConditionedLM(nn.Module):
    """Wraps a causal-LM backbone + a HamiltonianEncoder, replacing every
    occurrence of the <ham> token's embedding with the encoder's output for
    that example's (normalized) Hamiltonian coefficient vector (Sec. 3.2.2).
    The SAME per-example Hamiltonian embedding is broadcast to every <ham>
    position in that example's sequence -- one encoder call per example,
    not per occurrence.
    """

    def __init__(self, backbone, hamiltonian_encoder, ham_token_id: int):
        super().__init__()
        self.backbone = backbone
        self.hamiltonian_encoder = hamiltonian_encoder
        self.ham_token_id = ham_token_id

    def forward(self, input_ids, attention_mask=None, hamiltonian=None, labels=None):
        embed_layer = self.backbone.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        if hamiltonian is not None:
            ham_embed = self.hamiltonian_encoder(hamiltonian)  # (batch, hidden)
            mask = input_ids == self.ham_token_id  # (batch, seq)
            if mask.any():
                inputs_embeds = inputs_embeds.clone()
                batch_idx, seq_idx = mask.nonzero(as_tuple=True)
                inputs_embeds[batch_idx, seq_idx] = ham_embed[batch_idx]

        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "Standard model.generate() does not support re-injecting the "
            "Hamiltonian embedding at freshly generated <ham> token "
            "positions during incremental decoding (HF's generation loop "
            "only ever calls get_input_embeddings() on token ids, with no "
            "hook to substitute a specific position's embedding once that "
            "id has been sampled). This forward() path covers TRAINING "
            "(teacher forcing) only. Inference/RL rollouts need a custom "
            "autoregressive loop: at each step, embed the last sampled "
            "token normally UNLESS it equals ham_token_id, in which case "
            "substitute self.hamiltonian_encoder(hamiltonian) for that "
            "position before the next forward step -- this is a distinct "
            "piece of work, tracked separately for the RL/evaluation stage."
        )
