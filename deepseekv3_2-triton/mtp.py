# MTP (Multi-Token Prediction) -- DeepSeek-V3's 1-layer MTP architecture.
#
# DeepSeek-V3 uses 1 MTP layer that predicts the next token beyond the standard
# LM head prediction. During training, this provides an auxiliary loss that
# improves base model quality. During inference, it can serve as a draft model
# for speculative decoding.
#
# KEY DIFFERENCES FROM GLM5:
# - DeepSeek-V3: 1 MTP layer (predicts position i+2 given context at position i)
# - GLM5: 3 shared MTP layers (same weights applied 3 times)
#   - GLM5's approach: single set of parameters run 3 times, predicting positions i+2, i+3, i+4
#   - GLM5's MTP acts as a draft model for speculative decoding (4 speculative steps)
#   - GLM5's acceptance length: 2.76 tokens (vs DeepSeek-V3's 2.55)
#   - GLM5's key insight: parameter sharing keeps memory cost = 1 layer but increases acceptance
#
# STATUS: Implemented in model.py as MTPLayer class. This file provides the
# MTPHead class for standalone use and documentation.
#
# Architecture (from paper, arXiv 2412.19437, Section 2.3):
#   - Each MTP layer combines hidden state from position i with embedding of token i+1
#   - Combined representation is projected through a linear layer + RMSNorm
#   - Output head projects to vocabulary for predicting token at position i+2
#   - Training loss: auxiliary cross-entropy over predicted tokens
#   - Inference: draft model for speculative decoding
#
# Training diagram:
#   hidden[i] + embed(token[i+1]) -> concat -> project -> norm -> head -> logits[i+2]
#   Loss = CE(logits[i+2], token[i+2])
#
# Paper quote (Section 2.3):
# "We keep the MTP modules only during training, and discard them during
# inference. We will later discuss the potential use of MTP for speculative
# decoding."
#
# Speculative decoding potential:
# "For DeepSeek-V3, the average acceptance length of the generated tokens
# per step by the MTP module is around 1.55 additional tokens per step."

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """Multi-Token Prediction head for DeepSeek-V3.

    This is a standalone implementation of the MTP layer described in
    arXiv 2412.19437, Section 2.3. The model.py file also contains
    an MTPLayer class that is integrated into DeepSeekV3ForCausalLM.

    DeepSeek-V3 uses 1 MTP layer (num_nextn_predict_layers=1), meaning
    it predicts 1 additional token beyond the standard next-token prediction.

    For comparison, GLM5 uses 3 shared MTP layers (same weights applied 3 times),
    predicting 3 additional tokens. The parameter sharing means GLM5's memory
    cost is the same as a single MTP layer, but the acceptance rate is higher.

    Args:
        cfg: model config dict with keys:
            - hidden_size: 7168
            - vocab_size: 129280
            - rms_norm_eps: 1e-6
            - num_nextn_predict_layers: 1
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_mtp_layers = cfg.get("num_nextn_predict_layers", 1)
        hidden_size = cfg["hidden_size"]

        # Combine previous hidden state with next-token embedding
        self.embed_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.norm = nn.Module()  # Placeholder -- use RMSNorm from mla_attention
        self._init_norm(cfg)
        self.head = nn.Linear(hidden_size, cfg["vocab_size"], bias=False)

    def _init_norm(self, cfg):
        """Initialize RMSNorm without circular import."""
        from .mla_attention import RMSNorm
        self.norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(self, hidden_states, embed_tokens, input_ids=None, labels=None):
        """Forward pass for MTP prediction.

        Args:
            hidden_states: [B, S, hidden_size] from the main model
            embed_tokens: nn.Embedding module for token lookup
            input_ids: [B, S] original input token ids
            labels: [B, S] target token ids for MTP loss

        Returns:
            mtp_logits: [B, S-1, vocab_size]
            mtp_loss: scalar loss (None if labels not provided)
        """
        if input_ids is None:
            return None, None

        batch_size, seq_len, _ = hidden_states.shape
        if seq_len <= 1:
            return None, None

        # Shift: for position i, combine hidden_states[i] with embedding of input[i+1]
        shifted_embeds = embed_tokens(input_ids[:, 1:])  # [B, S-1, hidden_size]
        truncated_hidden = hidden_states[:, :-1, :]       # [B, S-1, hidden_size]

        # Concatenate and project
        combined = torch.cat([truncated_hidden, shifted_embeds], dim=-1)
        mtp_hidden = self.embed_proj(combined)
        mtp_hidden = self.norm(mtp_hidden)
        mtp_logits = self.head(mtp_hidden)

        mtp_loss = None
        if labels is not None:
            # MTP predicts position i+2 given position i context
            shifted_labels = labels[:, 2:] if labels.shape[1] > 2 else labels[:, 1:]
            mtp_logits_for_loss = mtp_logits[:, :shifted_labels.shape[1], :]
            mtp_loss = F.cross_entropy(
                mtp_logits_for_loss.reshape(-1, mtp_logits_for_loss.shape[-1]),
                shifted_labels.reshape(-1),
            )

        return mtp_logits, mtp_loss

    def speculative_decode(self, hidden_states, embed_tokens, last_token_id):
        """Generate a single draft token for speculative decoding.

        During inference, the MTP layer predicts the next token beyond
        the standard LM head's prediction. This draft token can be verified
        by the main model in its next forward pass.

        Args:
            hidden_states: [B, 1, hidden_size] -- last position's hidden state
            embed_tokens: nn.Embedding module
            last_token_id: [B, 1] -- the last predicted token

        Returns:
            draft_logits: [B, 1, vocab_size]
        """
        shifted_embeds = embed_tokens(last_token_id)  # [B, 1, hidden_size]
        combined = torch.cat([hidden_states, shifted_embeds], dim=-1)
        mtp_hidden = self.embed_proj(combined)
        mtp_hidden = self.norm(mtp_hidden)
        draft_logits = self.head(mtp_hidden)
        return draft_logits
