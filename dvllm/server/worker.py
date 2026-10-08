"""Single-threaded generation worker that owns the dvllm engine.

Why serialize?
  The dvllm engine is a learning implementation and is not safe to drive from
  several threads, and its batched paths are not yet correct for a server:

  * Multi-sequence prefill only samples one token (``compute_logits`` keeps
    the last position of the flattened batch), so batched prefill would
    hand the wrong token to the wrong request.
  * The built-in ``Sampler`` applies a single temperature to the whole batch.
  * The non-CUDA prefill attention ignores prefix-cached KV blocks, so a
    prefix-cache hit (e.g. two requests sharing a long system prompt) would
    produce garbage.

  So the server runs exactly one sequence at a time on a dedicated thread,
  clears the prefix-cache hash table before each request, and replaces the
  engine sampler with a per-request one that supports temperature / top_p /
  top_k / presence & frequency penalties. Concurrent HTTP requests queue up
  (FIFO) instead of crashing. Each ``engine.step()`` produces one token, so
  streaming is truly incremental.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from dvllm.engine.seq import Sequence, SequenceStatus
from dvllm.sparams import SParams

logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class RequestSampler(nn.Module):
    """Sampler for a single running sequence, configured per request."""

    def __init__(self):
        super().__init__()
        self.cfg = SamplingConfig()
        self.seq: Sequence | None = None

    @torch.inference_mode()
    def forward(self, logits: torch.Tensor, temperatures=None) -> torch.Tensor:
        cfg = self.cfg
        vocab = logits.shape[-1]
        # One sequence at a time: the next-token logits are the last row.
        x = logits.reshape(-1, vocab)[-1].float()

        if self.seq is not None and (cfg.presence_penalty or cfg.frequency_penalty):
            hist = self.seq.completion_token_ids
            if hist:
                counts = torch.bincount(
                    torch.tensor(hist, device=x.device), minlength=vocab
                )[:vocab].to(x.dtype)
                x = x - cfg.frequency_penalty * counts
                x = x - cfg.presence_penalty * (counts > 0).to(x.dtype)

        if cfg.temperature < 1e-4:
            token = x.argmax()
            return token.view(1, 1)

        x = x / cfg.temperature
        if 0 < cfg.top_k < vocab:
            vals, idx = torch.topk(x, cfg.top_k)  # sorted descending
        else:
            vals, idx = torch.sort(x, descending=True)
        if cfg.top_p < 1.0:
            probs = torch.softmax(vals, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            # drop tokens whose preceding mass already exceeds top_p
            # (always keeps the most likely token)
            vals = vals.masked_fill((cum - probs) > cfg.top_p, float("-inf"))
        probs = torch.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, 1)
        return idx[choice].view(1, 1)


@dataclass
class Job:
    prompt_ids: list[int]
    max_tokens: int
    sampling: SamplingConfig
    stop: list[str]
    stop_token_ids: set[int]
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue
    cancelled: threading.Event = field(default_factory=threading.Event)
    enqueued_at: float = field(default_factory=time.perf_counter)

    def emit(self, *event):
        self.loop.call_soon_threadsafe(self.events.put_nowait, event)


def _partial_suffix_len(text: str, needles: list[str]) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of a needle."""
    best = 0
    for n in needles:
        for k in range(min(len(n) - 1, len(text)), 0, -1):
            if text.endswith(n[:k]):
                best = max(best, k)
                break
    return best


class EngineWorker:
    """Owns the engine; processes one job at a time on a background thread."""

    def __init__(self, engine):
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.scheduler = engine.scheduler
        self.sampler = RequestSampler()
        # Replace the engine's sampler (temperature-only, one temp per batch).
        engine.model_runner.sampler = self.sampler
        self.jobs: queue.Queue[Job | None] = queue.Queue()
        self.busy = False
        self.thread = threading.Thread(target=self._loop, name="dvllm-worker", daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, job: Job):
        self.jobs.put(job)

    def queue_size(self) -> int:
        return self.jobs.qsize() + (1 if self.busy else 0)

    # ------------------------------------------------------------------ #
    def _loop(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            if job.cancelled.is_set():
                continue
            self.busy = True
            try:
                self._run(job)
            except Exception as e:  # keep the worker alive
                logger.exception("generation failed")
                self._reset_scheduler()
                job.emit("error", f"{type(e).__name__}: {e}")
            finally:
                self.busy = False
                self.sampler.seq = None

    def _reset_scheduler(self):
        sch = self.scheduler
        for seq in list(sch.running) + list(sch.waiting):
            if seq.block_table:
                sch.block_manager.deallocate(seq)
            seq.status = SequenceStatus.FINISHED
        sch.running.clear()
        sch.waiting.clear()

    def _abort(self, seq: Sequence):
        sch = self.scheduler
        if seq in sch.running:
            sch.block_manager.deallocate(seq)
            sch.running.remove(seq)
        elif seq in sch.waiting:
            sch.waiting.remove(seq)
        seq.status = SequenceStatus.FINISHED

    def _run(self, job: Job):
        tok = self.tokenizer
        sch = self.scheduler
        t_start = time.perf_counter()

        # Prefix caching is unsafe with the non-CUDA attention path (see module
        # docstring), so forget all cached block hashes before each request.
        sch.block_manager.hash_to_block_id.clear()

        sp = SParams(
            temperature=max(job.sampling.temperature, 1e-5),
            max_tokens=job.max_tokens,
        )
        seq = Sequence(job.prompt_ids, sp)
        self.sampler.cfg = job.sampling
        self.sampler.seq = seq
        sch.add(seq)

        finish_reason = None
        text = ""
        emitted_len = 0
        t_first = None
        stop = job.stop
        max_stop = max((len(s) for s in stop), default=0)

        while True:
            if job.cancelled.is_set():
                self._abort(seq)
                logger.info("request cancelled after %d tokens", seq.num_completion_tokens)
                return
            self.engine.step()
            if t_first is None:
                t_first = time.perf_counter()

            ids = seq.completion_token_ids
            if ids and ids[-1] in job.stop_token_ids:
                ids = ids[:-1]
                finish_reason = "stop"
            elif seq.is_finished:
                finish_reason = "length"

            text = tok.decode(ids, skip_special_tokens=False)

            if stop:
                search_from = max(0, emitted_len - max_stop)
                hits = [i for s in stop if (i := text.find(s, search_from)) != -1]
                if hits:
                    text = text[: min(hits)]
                    finish_reason = "stop"

            if finish_reason is not None:
                if not seq.is_finished:
                    self._abort(seq)
                break

            # Hold back incomplete UTF-8 and possible stop-string prefixes.
            safe = text.rstrip("�")
            if stop:
                safe = safe[: len(safe) - _partial_suffix_len(safe, stop)]
            if len(safe) > emitted_len:
                emitted_len = len(safe)
                job.emit("text", safe)

        t_end = time.perf_counter()
        n_out = len(seq.completion_token_ids)
        stats = {
            "prompt_tokens": len(job.prompt_ids),
            "completion_tokens": n_out,
            "queue_s": t_start - job.enqueued_at,
            "ttft_s": (t_first or t_end) - t_start,
            "total_s": t_end - t_start,
            "decode_tps": (n_out - 1) / (t_end - t_first) if t_first and n_out > 1 else 0.0,
        }
        logger.info(
            "done: prompt=%d completion=%d queue=%.2fs ttft=%.2fs total=%.2fs decode=%.1f tok/s finish=%s",
            stats["prompt_tokens"],
            n_out,
            stats["queue_s"],
            stats["ttft_s"],
            stats["total_s"],
            stats["decode_tps"],
            finish_reason,
        )
        job.emit("done", text, finish_reason, stats)
