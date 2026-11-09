"""CS-781 Phase 1 - four legal language-model architectures, written from scratch in PyTorch.

  gpt2    : classic decoder-only Transformer (learned absolute positions, LayerNorm, GELU MLP, multi-head attention)
  llama   : modern decoder-only Transformer (RoPE, RMSNorm, SwiGLU, grouped-query attention, no biases)
  shared  : lightweight decoder (ALBERT-style factorised embeddings + cycle cross-layer weight sharing + multi-query attn)
  encdec  : encoder-decoder Transformer (bidirectional encoder over the prompt, causal decoder with cross-attention)

All models: tied input/output embeddings, pre-norm residual blocks, PyTorch SDPA attention, KV-cache generation.
"""
import math
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    arch: str = "llama"
    vocab_size: int = 32768
    d_model: int = 768
    n_layer: int = 12            # decoder-only: layers; encdec: layers in EACH of encoder and decoder; shared: effective depth
    n_head: int = 12
    n_kv_head: int = 4           # GQA groups (llama/encdec); shared uses 1 (MQA); gpt2 uses n_head (MHA)
    d_ff: int = 2304
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    n_unique_layers: int = 6     # shared only: number of distinct blocks, reused cyclically
    d_embed: int = 256           # shared only: factorised embedding size
    norm_eps: float = 1e-5

    def to_dict(self):
        return asdict(self)


def preset(arch: str, **overrides) -> ModelConfig:
    base = {
        "gpt2":   dict(arch="gpt2", d_model=768, n_layer=12, n_head=12, n_kv_head=12, d_ff=3072),
        "llama":  dict(arch="llama", d_model=768, n_layer=12, n_head=12, n_kv_head=4, d_ff=2304),
        "shared": dict(arch="shared", d_model=768, n_layer=12, n_head=12, n_kv_head=1, d_ff=3072,
                       n_unique_layers=6, d_embed=256),
        "encdec": dict(arch="encdec", d_model=768, n_layer=6, n_head=12, n_kv_head=4, d_ff=2304),
    }[arch]
    base.update(overrides)
    return ModelConfig(**base)


# ----------------------------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d, eps=eps)

    def forward(self, x):
        return F.layer_norm(x.float(), self.ln.normalized_shape, self.ln.weight.float(), self.ln.bias.float(),
                            self.ln.eps).to(x.dtype)


def rope_cache(head_dim, max_len, theta, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv)                       # (T, hd/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); cos/sin: (T, hd/2) - rotate pairs (x[..., :hd/2], x[..., hd/2:])
    xf = x.float()
    x1, x2 = xf[..., : xf.shape[-1] // 2], xf[..., xf.shape[-1] // 2:]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class Attention(nn.Module):
    """Multi-head / grouped-query / multi-query attention. Self-attention (optionally causal) or cross-attention."""

    def __init__(self, cfg: ModelConfig, n_kv_head: int, bias: bool, use_rope: bool, cross: bool = False):
        super().__init__()
        d, h = cfg.d_model, cfg.n_head
        assert d % h == 0 and h % n_kv_head == 0
        self.h, self.kv, self.hd = h, n_kv_head, d // h
        self.use_rope, self.cross = use_rope, cross
        self.q = nn.Linear(d, h * self.hd, bias=bias)
        self.k = nn.Linear(d, n_kv_head * self.hd, bias=bias)
        self.v = nn.Linear(d, n_kv_head * self.hd, bias=bias)
        self.o = nn.Linear(h * self.hd, d, bias=bias)
        self.o.is_residual_proj = True
        self.dropout = cfg.dropout

    def _split(self, x, n):
        B, T, _ = x.shape
        return x.view(B, T, n, self.hd).transpose(1, 2)          # (B, n, T, hd)

    def _expand(self, x):
        return x if self.kv == self.h else x.repeat_interleave(self.h // self.kv, dim=1)

    def forward(self, x, rope=None, pos=0, causal=True, cache=None, memory=None):
        """
        x: (B, T, d).  rope: (cos, sin) full tables.  pos: absolute position of x[:,0] (for RoPE during decoding).
        cache: dict used during generation. Self-attn: keys 'k','v' accumulate. Cross-attn: 'k','v' computed once.
        memory: encoder output (B, S, d) for cross-attention.
        """
        q = self._split(self.q(x), self.h)
        if self.cross:
            if cache is not None and "k" in cache:
                k, v = cache["k"], cache["v"]
            else:
                k, v = self._split(self.k(memory), self.kv), self._split(self.v(memory), self.kv)
                if cache is not None:
                    cache["k"], cache["v"] = k, v
            is_causal = False
        else:
            k, v = self._split(self.k(x), self.kv), self._split(self.v(x), self.kv)
            if self.use_rope:
                cos, sin = rope
                T = x.shape[1]
                q = apply_rope(q, cos[pos:pos + T], sin[pos:pos + T])
                k = apply_rope(k, cos[pos:pos + T], sin[pos:pos + T])
            if cache is not None:
                if "k" in cache:
                    k = torch.cat([cache["k"], k], dim=2); v = torch.cat([cache["v"], v], dim=2)
                cache["k"], cache["v"] = k, v
            # causal mask only needed when queries and keys cover the same span (training / prefill)
            is_causal = causal and q.shape[2] == k.shape[2] and q.shape[2] > 1
            if causal and q.shape[2] != k.shape[2] and q.shape[2] != 1:
                raise ValueError("multi-token decode with a non-empty cache is not supported")
        y = F.scaled_dot_product_attention(q, self._expand(k), self._expand(v), is_causal=is_causal,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.h * self.hd)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str, bias: bool):
        super().__init__()
        self.kind = kind
        if kind == "swiglu":
            self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=bias)
            self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=bias)
        else:
            self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=bias)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=bias)
        self.down.is_residual_proj = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            h = F.silu(self.gate(x)) * self.up(x)
        else:
            h = F.gelu(self.up(x), approximate="tanh")
        return self.drop(self.down(h))


class Block(nn.Module):
    """Pre-norm Transformer block: x + Attn(Norm(x)) [+ CrossAttn(Norm(x), memory)] + MLP(Norm(x))."""

    def __init__(self, cfg: ModelConfig, style: str, n_kv_head: int, cross: bool = False):
        super().__init__()
        gpt2 = style == "gpt2"
        Norm = LayerNorm if gpt2 else RMSNorm
        self.n1 = Norm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, n_kv_head, bias=gpt2, use_rope=not gpt2)
        self.has_cross = cross
        if cross:
            self.nc = Norm(cfg.d_model, cfg.norm_eps)
            self.xattn = Attention(cfg, n_kv_head, bias=False, use_rope=False, cross=True)
        self.n2 = Norm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg, "gelu" if gpt2 else "swiglu", bias=gpt2)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None, pos=0, causal=True, cache=None, memory=None, xcache=None):
        x = x + self.drop(self.attn(self.n1(x), rope=rope, pos=pos, causal=causal, cache=cache))
        if self.has_cross:
            x = x + self.drop(self.xattn(self.nc(x), memory=memory, cache=xcache))
        return x + self.mlp(self.n2(x))


# ----------------------------------------------------------------------------------------------------------------
# the language model
# ----------------------------------------------------------------------------------------------------------------
class LegalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        a = cfg.arch
        assert a in ("gpt2", "llama", "shared", "encdec"), a
        d = cfg.d_model
        if a == "shared":
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_embed)
            self.emb_proj = nn.Linear(cfg.d_embed, d, bias=False)          # E -> d   (factorised embedding)
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        if a == "gpt2":
            self.pos_emb = nn.Embedding(cfg.max_seq_len, d)
        self.drop = nn.Dropout(cfg.dropout)

        if a == "gpt2":
            self.blocks = nn.ModuleList([Block(cfg, "gpt2", cfg.n_head) for _ in range(cfg.n_layer)])
        elif a == "llama":
            self.blocks = nn.ModuleList([Block(cfg, "llama", cfg.n_kv_head) for _ in range(cfg.n_layer)])
        elif a == "shared":
            assert cfg.n_layer % cfg.n_unique_layers == 0
            self.blocks = nn.ModuleList([Block(cfg, "llama", cfg.n_kv_head) for _ in range(cfg.n_unique_layers)])
        else:  # encdec
            self.enc_blocks = nn.ModuleList([Block(cfg, "llama", cfg.n_kv_head) for _ in range(cfg.n_layer)])
            self.enc_norm = RMSNorm(d, cfg.norm_eps)
            self.blocks = nn.ModuleList([Block(cfg, "llama", cfg.n_kv_head, cross=True) for _ in range(cfg.n_layer)])
        self.norm = LayerNorm(d, cfg.norm_eps) if a == "gpt2" else RMSNorm(d, cfg.norm_eps)

        if a != "gpt2":
            hd = d // cfg.n_head
            cos, sin = rope_cache(hd, max(2 * cfg.max_seq_len, 2048), cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        n_res = 2 * self.n_layer_applications() + (cfg.n_layer if a == "encdec" else 0)
        for name, p in self.named_parameters():
            if name.endswith("o.weight") or name.endswith("down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(n_res))

    # -- helpers ------------------------------------------------------------------------------------------------
    def n_layer_applications(self):
        return self.cfg.n_layer * (2 if self.cfg.arch == "encdec" else 1)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _rope(self):
        return None if self.cfg.arch == "gpt2" else (self.rope_cos, self.rope_sin)

    def _embed(self, idx, pos=0):
        x = self.tok_emb(idx)
        if self.cfg.arch == "shared":
            x = self.emb_proj(x)
        if self.cfg.arch == "gpt2":
            T = idx.shape[1]
            if pos + T > self.cfg.max_seq_len:
                raise ValueError(f"gpt2 positions exceed max_seq_len ({pos + T} > {self.cfg.max_seq_len})")
            x = x + self.pos_emb(torch.arange(pos, pos + T, device=idx.device))[None]
        return self.drop(x)

    def _logits(self, h):
        h = self.norm(h)
        if self.cfg.arch == "shared":
            h = F.linear(h, self.emb_proj.weight.t())     # d -> E   (tied with the input projection)
        return F.linear(h, self.tok_emb.weight)            # tied output embedding

    def _decoder_layers(self):
        if self.cfg.arch == "shared":                       # cycle sharing: 0,1,..,5,0,1,..,5
            return [self.blocks[i % self.cfg.n_unique_layers] for i in range(self.cfg.n_layer)]
        return list(self.blocks)

    def encode(self, src):
        x = self._embed(src)
        for blk in self.enc_blocks:
            x = blk(x, rope=self._rope(), pos=0, causal=False)
        return self.enc_norm(x)

    # -- training forward -----------------------------------------------------------------------------------------
    def forward(self, idx, targets=None, src=None, loss_mask=None):
        """Decoder-only: idx (B,T) -> logits; loss vs targets (B,T).  encdec: src (B,S) is the encoder input."""
        memory = self.encode(src) if self.cfg.arch == "encdec" else None
        x = self._embed(idx)
        for blk in self._decoder_layers():
            x = blk(x, rope=self._rope(), pos=0, causal=True, memory=memory)
        logits = self._logits(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        if loss_mask is not None:
            m = loss_mask.reshape(-1).float()
            return logits, (loss * m).sum() / m.sum().clamp(min=1)
        return logits, loss.mean()

    # -- generation -----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                 no_repeat_ngram=0, banned_ids=(), bos_id=2, generator=None, use_cache=True):
        """prompt: (B, T) LongTensor, all rows the same length (no padding). Returns (B, max_new_tokens) new ids.
        temperature 0 = greedy."""
        B = prompt.shape[0]
        dev = prompt.device
        layers = self._decoder_layers()
        if self.cfg.arch == "encdec":
            memory = self.encode(prompt)
            seq = torch.full((B, 1), bos_id, dtype=torch.long, device=dev)   # decoder starts from <|bos|>
            history = prompt                                                   # for repetition penalties
        else:
            memory, seq, history = None, prompt, prompt
        caches = [dict() for _ in layers] if use_cache else None
        xcaches = [dict() for _ in layers] if (use_cache and memory is not None) else [None] * len(layers)
        out = []
        cur, pos = seq, 0
        for step in range(max_new_tokens):
            if use_cache:
                x = self._embed(cur, pos=pos)
                for li, blk in enumerate(layers):
                    x = blk(x, rope=self._rope(), pos=pos, causal=True, cache=caches[li], memory=memory,
                            xcache=xcaches[li])
                pos += cur.shape[1]
            else:
                x = self._embed(seq)
                for blk in layers:
                    x = blk(x, rope=self._rope(), pos=0, causal=True, memory=memory)
            logits = self._logits(x[:, -1:, :])[:, -1, :].float()
            full_hist = torch.cat([history] + out, dim=1) if out else history
            nxt = sample_next(logits, full_hist, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram,
                              banned_ids, generator)
            out.append(nxt)
            seq = torch.cat([seq, nxt], dim=1)
            cur = nxt
        return torch.cat(out, dim=1)


def sample_next(logits, history, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram, banned_ids,
                generator=None):
    """logits (B,V) float32, history (B,L) long (prompt + generated so far). Returns (B,1)."""
    logits = logits.clone()
    if banned_ids:
        logits[:, list(banned_ids)] = -float("inf")
    if repetition_penalty != 1.0:
        # CTRL-style penalty on tokens already present in the recent history (last 256 tokens)
        recent = history[:, -256:]
        sc = torch.gather(logits, 1, recent)
        sc = torch.where(sc > 0, sc / repetition_penalty, sc * repetition_penalty)
        logits.scatter_(1, recent, sc)
    if no_repeat_ngram and history.shape[1] >= no_repeat_ngram:
        n = no_repeat_ngram
        hist = history[:, -512:].tolist()
        for b, h in enumerate(hist):
            prefix = tuple(h[-(n - 1):]) if n > 1 else tuple()
            banned = {tuple(h[i:i + n])[-1] for i in range(len(h) - n + 1) if tuple(h[i:i + n - 1]) == prefix}
            if banned:
                logits[b, list(banned)] = -float("inf")
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cp = torch.softmax(sl, dim=-1).cumsum(dim=-1)
        remove = cp - torch.softmax(sl, dim=-1) > top_p          # keep the smallest set with mass >= top_p
        sl = sl.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(1, si, sl)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)