"""Ray actor for multi-turn tool-integrated sampling.

Unlike the baseline ``SamplingWorker`` which generates in a single pass,
this worker implements iterative generation:

1. Generate until ``</use_tool>`` is emitted (or ``</answer>`` / max tokens).
2. Parse the tool call, execute it deterministically, append
   ``<tool_result>output</tool_result>`` to the context.
3. Resume generation from the extended context.
4. Repeat until ``</answer>`` or the token budget is exhausted.

This allows the policy to *condition on real tool outputs* during
generation, which is the standard approach in TIR papers (Tool-Star,
ToRL, etc.).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from tir_extension.tools.tool_pool import (
    execute_tool_calls,
    parse_tool_calls,
    relevant_tool_names,
)
from tir_extension.tools.system_prompt import build_tool_system_prompt


@ray.remote(num_gpus=1)
class TIRSamplingWorker:
    """GPU worker that generates with iterative tool execution."""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.9,
        max_num_batched_tokens: int = 8192,
        enable_chunked_prefill: bool = True,
        max_num_seqs: int = 64,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 1024,
        group_size: int = 16,
        max_tool_turns: int = 5,
        active_tools: set[str] | None = None,
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_seqs = max_num_seqs
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.group_size = group_size
        self.max_tool_turns = max_tool_turns
        self.active_tools = active_tools or relevant_tool_names()

    def load_checkpoint(self):
        """(Re)load tokenizer + vLLM engine."""
        self.tear_down()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        effective_max_model_len = self.max_model_len
        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)
        if (
            effective_max_model_len is not None
            and isinstance(tokenizer_max_length, int)
            and 0 < tokenizer_max_length < 1_000_000
        ):
            effective_max_model_len = min(effective_max_model_len, tokenizer_max_length)
        effective_max_num_batched_tokens = self.max_num_batched_tokens
        if (
            effective_max_model_len is not None
            and effective_max_num_batched_tokens is not None
            and effective_max_num_batched_tokens < effective_max_model_len
        ):
            effective_max_num_batched_tokens = effective_max_model_len

        llm_kwargs = {
            "model": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_batched_tokens": effective_max_num_batched_tokens,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "max_num_seqs": self.max_num_seqs,
        }
        if effective_max_model_len is not None:
            llm_kwargs["max_model_len"] = effective_max_model_len

        self.llm = LLM(**llm_kwargs)

    def tear_down(self):
        """Release GPU memory."""
        import gc
        import torch

        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "llm"):
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass
            del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @staticmethod
    def _extract_sequence_logprob(output) -> float:
        """Best-effort extraction of sequence logprob across vLLM versions."""
        if hasattr(output, "cumulative_logprob") and output.cumulative_logprob is not None:
            return float(output.cumulative_logprob)
        if hasattr(output, "logprob") and output.logprob is not None:
            return float(output.logprob)

        token_ids = getattr(output, "token_ids", None)
        token_logprobs = getattr(output, "logprobs", None)
        if token_ids is None or token_logprobs is None:
            raise RuntimeError(
                "Could not extract sequence logprob from vLLM output."
            )

        seq_logprob = 0.0
        for token_id, token_topk in zip(token_ids, token_logprobs):
            if token_topk is None:
                continue
            token_entry = None
            if isinstance(token_topk, dict):
                token_entry = token_topk.get(token_id)
                if token_entry is None and len(token_topk) == 1:
                    token_entry = next(iter(token_topk.values()))
            else:
                token_entry = token_topk
            if token_entry is None:
                continue
            if hasattr(token_entry, "logprob"):
                seq_logprob += float(token_entry.logprob)
            else:
                seq_logprob += float(token_entry)

        return float(seq_logprob)

    def _generate_single_turn(
        self, prompts: list[str], max_tokens: int, stop: list[str]
    ) -> list[list]:
        """Run one vLLM generate call. Returns raw vLLM output objects."""
        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=max_tokens,
            n=1,
            stop=stop,
            include_stop_str_in_output=True,
            logprobs=1,
        )
        outputs = self.llm.generate(prompts, params)
        return outputs

    def generate(
        self,
        prompts: list[str],
        n: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Multi-turn tool-integrated generation.

        For each prompt, generates ``n`` (default ``group_size``) independent
        rollouts. Each rollout iterates: generate -> check for tool call ->
        execute tool -> append result -> continue.

        Returns:
            (all_responses, all_logprobs): Lists of shape [n_prompts][n_per_prompt].
            Each response is the full generated text (including tool results).
            Logprobs are summed across all generation turns.
        """
        if n is None:
            n = self.group_size

        # Expand each prompt into n independent rollouts
        expanded_prompts = []
        for prompt in prompts:
            for _ in range(n):
                expanded_prompts.append(prompt)

        # Track state for each rollout
        num_rollouts = len(expanded_prompts)
        contexts = list(expanded_prompts)  # full context so far
        generated_texts = [""] * num_rollouts  # just the generated part
        cumulative_logprobs = [0.0] * num_rollouts
        finished = [False] * num_rollouts
        tokens_used = [0] * num_rollouts

        for turn in range(self.max_tool_turns + 1):
            # Collect unfinished rollouts
            active_indices = [i for i in range(num_rollouts) if not finished[i]]
            if not active_indices:
                break

            active_contexts = [contexts[i] for i in active_indices]
            remaining_tokens = [
                max(1, self.max_tokens - tokens_used[i]) for i in active_indices
            ]
            # Use the minimum remaining tokens for this batch (vLLM needs uniform params)
            # but cap individual rollouts by their own budget
            max_tokens_this_turn = max(remaining_tokens)

            # Stop at </use_tool> (to execute tool) or </answer> (done)
            outputs = self._generate_single_turn(
                active_contexts,
                max_tokens=max_tokens_this_turn,
                stop=["</use_tool>", "</answer>"],
            )

            for idx, output in zip(active_indices, outputs):
                o = output.outputs[0]
                new_text = o.text
                new_logprob = self._extract_sequence_logprob(o)
                new_token_count = len(o.token_ids) if hasattr(o, "token_ids") else 0

                generated_texts[idx] += new_text
                cumulative_logprobs[idx] += new_logprob
                tokens_used[idx] += new_token_count

                # Check if we hit </answer> or ran out of tokens
                if "</answer>" in new_text or tokens_used[idx] >= self.max_tokens:
                    finished[idx] = True
                    continue

                # Check if we hit </use_tool> — execute tool and continue
                if "</use_tool>" in new_text:
                    # Execute the tool call(s) in the generated text so far
                    executed = execute_tool_calls(
                        generated_texts[idx],
                        active_tools=self.active_tools,
                        max_calls=1,
                    )
                    # Find what was appended (the <tool_result>...</tool_result>)
                    if executed != generated_texts[idx]:
                        generated_texts[idx] = executed
                    else:
                        # Tool execution didn't add anything — mark finished
                        finished[idx] = True
                        continue

                    # Update context for next turn
                    contexts[idx] = expanded_prompts[idx] + generated_texts[idx]

                    # Check if context exceeds max_model_len — stop if so
                    ctx_tokens = len(self.tokenizer.encode(contexts[idx]))
                    if ctx_tokens >= self.max_model_len - 10:
                        finished[idx] = True
                        continue
                else:
                    # Stopped for another reason (max tokens, etc.)
                    finished[idx] = True

        # Reshape back into [n_prompts][n_per_prompt]
        all_responses = []
        all_logprobs = []
        for i in range(len(prompts)):
            group_responses = []
            group_logprobs = []
            for j in range(n):
                flat_idx = i * n + j
                group_responses.append(generated_texts[flat_idx])
                group_logprobs.append(cumulative_logprobs[flat_idx])
            all_responses.append(group_responses)
            all_logprobs.append(group_logprobs)

        return all_responses, all_logprobs
