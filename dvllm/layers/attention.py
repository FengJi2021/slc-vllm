import torch
from torch import nn
import torch.nn.functional as F

# lazy imports for optional accel libs
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    HAS_FLASH_ATTN = True
except Exception:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    HAS_FLASH_ATTN = False

from dvllm.utils.context import get_context

# Only define Triton kernel if triton is present
if HAS_TRITON:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    def store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        store_kvcache_kernel[(N,)](
            key,
            key.stride(0),
            value,
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping,
            D,
        )

else:

    def store_kvcache_triton(*args, **kwargs):
        raise RuntimeError("triton not available")


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _fallback_prefill_attention(
        self, q, k, v, context, chunk_size: int | None = None
    ):
        # q: [N, num_heads, D], k/v: [N, num_kv_heads, D] (tokens of all seqs concatenated)
        # Run causal SDPA per-sequence using cu_seqlens_q so tokens of different
        # sequences never attend to each other.
        N, H, D = q.shape
        kv_heads = k.shape[1]
        cu = context.cu_seqlens_q
        if cu is None:
            # single sequence covering all tokens
            cu = torch.tensor([0, N], device=q.device, dtype=torch.int32)

        outs = []
        for s in range(len(cu) - 1):
            start = int(cu[s].item())
            end = int(cu[s + 1].item())
            if end <= start:
                continue
            # [1, heads, L, D]
            q_s = q[start:end].permute(1, 0, 2).unsqueeze(0)
            k_s = k[start:end].permute(1, 0, 2).unsqueeze(0)
            v_s = v[start:end].permute(1, 0, 2).unsqueeze(0)
            if chunk_size is not None and end - start > chunk_size:
                out = self._fallback_chunked_prefill_sequence(q_s, k_s, v_s, chunk_size)
            else:
                if H != kv_heads:
                    factor = H // kv_heads
                    k_s = k_s.repeat_interleave(factor, dim=1)
                    v_s = v_s.repeat_interleave(factor, dim=1)
                out = F.scaled_dot_product_attention(
                    q_s, k_s, v_s, attn_mask=None, is_causal=True, scale=self.scale
                )  # [1, heads, L, D]
            outs.append(out.squeeze(0).permute(1, 0, 2))  # [L, heads, D]
        return torch.cat(outs, dim=0)  # [N, num_heads, D]

    def _fallback_chunked_prefill_sequence(self, q, k, v, chunk_size: int):
        """Compute one causal prefill sequence without materializing full attention."""
        _, num_heads, seq_len, _ = q.shape
        num_kv_heads = k.shape[1]
        factor = num_heads // num_kv_heads
        out_chunks = []

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, :end, :]
            v_chunk = v[:, :, :end, :]

            if factor > 1:
                k_chunk = k_chunk.repeat_interleave(factor, dim=1)
                v_chunk = v_chunk.repeat_interleave(factor, dim=1)

            query_positions = torch.arange(start, end, device=q.device).unsqueeze(1)
            key_positions = torch.arange(end, device=q.device).unsqueeze(0)
            causal_mask = key_positions <= query_positions
            out_chunks.append(
                F.scaled_dot_product_attention(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    attn_mask=causal_mask,
                    is_causal=False,
                    scale=self.scale,
                )
            )

        return torch.cat(out_chunks, dim=2)

    def _fallback_decode_attention(self, q, k_cache, v_cache, context):
        # q: [N, num_heads, head_dim] (one new token per sequence)
        # k_cache / v_cache: paged [num_blocks, block_size, num_kv_heads, head_dim]
        # For each sequence we gather its full K/V history from the paged cache
        # using block_tables + context_lens, then run causal SDPA.
        N, H, D = q.shape
        block_size = k_cache.size(1)
        block_tables = context.block_tables  # [N, max_num_blocks]
        context_lens = context.context_lens  # [N]

        outs = []
        for i in range(N):
            ctx_len = int(context_lens[i].item())
            num_blocks = (ctx_len + block_size - 1) // block_size
            blocks = block_tables[i, :num_blocks]  # [num_blocks]

            # gather blocks: [num_blocks, block_size, num_kv_heads, head_dim]
            k_blocks = k_cache[blocks]
            v_blocks = v_cache[blocks]
            kv_heads = k_blocks.size(2)
            # flatten block/offset dims and trim padding to real context length
            k_i = k_blocks.reshape(-1, kv_heads, D)[:ctx_len]  # [ctx_len, kv_heads, D]
            v_i = v_blocks.reshape(-1, kv_heads, D)[:ctx_len]

            # [1, num_heads, 1, D]
            q_i = q[i].unsqueeze(0).unsqueeze(2)
            # [1, kv_heads, ctx_len, D]
            k_i = k_i.permute(1, 0, 2).unsqueeze(0)
            v_i = v_i.permute(1, 0, 2).unsqueeze(0)

            # GQA: repeat kv heads to match query heads
            if H != kv_heads:
                factor = H // kv_heads
                k_i = k_i.repeat_interleave(factor, dim=1)
                v_i = v_i.repeat_interleave(factor, dim=1)

            out = F.scaled_dot_product_attention(
                q_i, k_i, v_i, attn_mask=None, is_causal=False, scale=self.scale
            )  # [1, num_heads, 1, D]
            outs.append(out.reshape(H, D))

        return torch.stack(outs, dim=0)  # [N, num_heads, head_dim]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prefill_chunk_size: int | None = None,
    ):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            if HAS_TRITON and q.device.type == "cuda":
                store_kvcache_triton(k, v, k_cache, v_cache, context.slot_mapping)
            else:
                slots = context.slot_mapping.cpu().tolist()
                for i, slot in enumerate(slots):
                    if slot == -1:
                        continue
                    key_row = k[i].reshape(-1)
                    val_row = v[i].reshape(-1)
                    D = k.shape[1] * k.shape[2]
                    k_cache.view(-1, D)[slot] = key_row.to(k_cache.device)
                    v_cache.view(-1, D)[slot] = val_row.to(v_cache.device)

        use_flash = (q.device.type == "cuda") and HAS_FLASH_ATTN

        if context.is_prefill:
            if use_flash:
                return flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:
                return self._fallback_prefill_attention(
                    q, k, v, context, chunk_size=prefill_chunk_size
                )
        else:
            if use_flash:
                return flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
            else:
                return self._fallback_decode_attention(q, k_cache, v_cache, context)
