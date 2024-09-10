import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

class StickbreakingFromLogits(torch.autograd.Function):
    @staticmethod
    # @torch.jit.script
    def forward_scriptable(logits: torch.Tensor, mask: torch.Tensor, cum_weight: torch.Tensor):
        zero_mask = mask #  if mask is not None else None
        log_betas = F.logsigmoid(logits)
        neg_log_neg_betas = F.softplus(logits)
        if mask is not None:
            neg_log_neg_betas.masked_fill_(zero_mask, 0.)
        log_att = F.linear(neg_log_neg_betas, cum_weight.t())
        log_att.neg_()
        log_att.add_(log_betas)
        global att
        att = torch.exp(log_att)

        if mask is not None:
            att.masked_fill_(zero_mask, 0.)

        return att, zero_mask, log_betas

    @staticmethod
    def forward(ctx, logits, mask, cum_weight):
        att, zero_mask, log_betas = StickbreakingFromLogits.forward_scriptable(logits, mask, cum_weight)
        ctx.save_for_backward(att, zero_mask, log_betas, cum_weight)
        return att

    @staticmethod
    @torch.jit.script
    def backward_scriptable(
            dA: torch.Tensor,
            # grad_out: torch.Tensor,
            att: torch.Tensor, zero_mask: torch.Tensor, log_betas: torch.Tensor,
            cum_weight: torch.Tensor):
        dlogA = att * dA
        logit_grad = dlogA - torch.exp(log_betas) * (dlogA + F.linear(dlogA, cum_weight))

        # att_dA = att * dA
        # beta = torch.exp(log_betas)
        # logit_grad = (1 - beta) * att_dA - beta * F.linear(att_dA, cum_weight)
        if zero_mask is not None:
            logit_grad.masked_fill_(zero_mask, 0.)
        return logit_grad

    @staticmethod
    def backward(ctx, grad_out):
        att, zero_mask, log_betas, cum_weight = ctx.saved_tensors
        logit_grad = StickbreakingFromLogits.backward_scriptable(
            grad_out, att, zero_mask, log_betas, cum_weight)
        return logit_grad, None, None

def stickbreaking(q, k, v, mask, cum_weight) -> torch.Tensor:
    """
    Stick-breaking attention weights.
    """
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])

    original_dtype = logits.dtype
    # logits = logits.float()
    att = StickbreakingFromLogits.apply(logits, mask, cum_weight)
    # print(att[..., :8, :8])
    return att @ v, 1 - att.sum(-1)

class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout
        self.attn_holder = None

    def forward(self, qkv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
        """
        seqlen = qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = 1.5 / math.sqrt(q.shape[-1])
        scores = softmax_scale * torch.einsum("bthd,bshd->bhts", q, k)
        cumweight = torch.ones(seqlen, seqlen).tril(-1).to(q)
        mask = torch.ones(seqlen, seqlen, dtype=torch.bool, device=q.device).triu(0)
        attention = StickbreakingFromLogits.apply(scores, mask, cumweight)
        self.attn_holder = attention
        rem = 1 - attention.sum(-1)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        rem = rem.permute(0, 2, 1)
        output = output.add_(rem[..., None] * v)
        return output


class MHA(nn.Module):
    """Multi-head self-attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        bias: bool=True,
        dropout: float=0.0,
        layer_idx: int=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_model % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """"""
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        context = self.inner_attn(qkv)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
