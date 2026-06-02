"""Ray actor that serves batched generation for RLOO training.

This process owns a vLLM engine and is restarted when checkpoints change.
"""

import warnings
warnings.filterwarnings("ignore")

import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

@ray.remote(num_gpus=1)
class SamplingWorker:
    """GPU worker responsible for policy rollouts (text sampling)."""
    def __init__(
        self, 
        model_path, 
        max_model_len=2048, 
        gpu_memory_utilization=0.9, 
        max_num_batched_tokens=8192, 
        enable_chunked_prefill=True, 
        max_num_seqs=64,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=1024,
        group_size=16
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

    def load_checkpoint(self):
        """(Re)load tokenizer + vLLM engine for current model path."""
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
            
        self.effective_max_model_len = effective_max_model_len
        effective_max_num_batched_tokens = self.max_num_batched_tokens
        if (
            effective_max_model_len is not None
            and effective_max_num_batched_tokens is not None
            and effective_max_num_batched_tokens < effective_max_model_len
        ):
            effective_max_num_batched_tokens = effective_max_model_len

        # Build vLLM config once so callers can hot-swap model paths cleanly.
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
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            n=self.group_size,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            logprobs=1
        )

    def update_model_path(self, model_path):
        """Switch to a new checkpoint and reload generation engine."""
        self.model_path = model_path
        self.load_checkpoint()

    def tear_down(self):
        """Release GPU memory and distributed state before reload/exit."""
        import gc
        import torch
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'llm'):
            # vLLM requires explicit cleanup of distributed resources
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass
            del self.llm
        if hasattr(self, 'sampling_params'):
            del self.sampling_params
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

        # Fallback path: reconstruct from per-token logprobs if cumulative logprob
        # is not present in this vLLM version.
        token_ids = getattr(output, "token_ids", None)
        token_logprobs = getattr(output, "logprobs", None)
        if token_ids is None or token_logprobs is None:
            raise RuntimeError(
                "Could not extract sequence logprob from vLLM output: missing cumulative_logprob/logprob and token-level logprobs."
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

    def generate_multi_turn(
        self,
        prompts: list[str],
        active_tools: list[str] | None = None,
        max_tool_calls: int = 8,
        n: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Multi-turn generation: stop on </use_tool>, execute the tool,
        feed the real result back, then continue — the model actually SEES
        tool outputs instead of hallucinating them.

        Returns the same (responses, logprobs) shape as generate().
        """
        from vllm import SamplingParams as _SP
        from tir_extension.tools.tool_pool import TOOL_REGISTRY, get_tool, parse_tool_calls

        group = n or self.group_size
        n_prompts = len(prompts)
        active_set = set(active_tools) if active_tools is not None else set(TOOL_REGISTRY)

        # Per-(prompt, response) state.
        contexts = [[prompts[p] for _ in range(group)] for p in range(n_prompts)]
        responses = [[""] * group for _ in range(n_prompts)]
        logprobs = [[0.0] * group for _ in range(n_prompts)]
        done = [[False] * group for _ in range(n_prompts)]

        for turn in range(max_tool_calls + 1):
            pending = [(p, r) for p in range(n_prompts) for r in range(group) if not done[p][r]]
            if not pending:
                break

            # On the final turn, only stop on </answer> (no more tool calls).
            stop_tokens = ["</answer>", "</use_tool>"] if turn < max_tool_calls else ["</answer>"]
            sp = _SP(
                temperature=self.temperature, top_p=self.top_p,
                top_k=self.top_k, min_p=self.min_p,
                max_tokens=self.max_tokens, n=1,
                stop=stop_tokens, include_stop_str_in_output=True,
                logprobs=1,
            )

            # Filter out contexts that exceed max_model_len using the tokenizer.
            # Use getattr to safely pull the effective length
            current_max = getattr(self, "effective_max_model_len", self.max_model_len) 
            max_ctx_len = (current_max or 2048) - 256
            safe_pending = []
            safe_ctxs = []
            for p, r in pending:
                ctx = contexts[p][r]
                tok_len = len(self.tokenizer.encode(ctx, add_special_tokens=False))
                if tok_len < max_ctx_len:
                    safe_pending.append((p, r))
                    safe_ctxs.append(ctx)
                else:
                    done[p][r] = True  # context too long, stop here

            if not safe_ctxs:
                break

            outputs = self.llm.generate(safe_ctxs, sp)
            pending = safe_pending

            for (p, r), out in zip(pending, outputs):
                o = out.outputs[0]
                generated = o.text
                responses[p][r] += generated
                logprobs[p][r] += self._extract_sequence_logprob(o)

                if "</use_tool>" in generated and turn < max_tool_calls:
                    # Execute the last tool call in the generated text.
                    calls = parse_tool_calls(generated)
                    if calls:
                        name, inp = calls[-1]
                        if name in TOOL_REGISTRY and name in active_set:
                            try:
                                result = get_tool(name)(inp)
                            except Exception as exc:
                                result = f"error: {exc}"
                        else:
                            result = f"error: tool '{name}' not available"
                        tool_block = f"<tool_result>{result}</tool_result>"
                        responses[p][r] += tool_block
                        contexts[p][r] += generated + tool_block
                    else:
                        done[p][r] = True
                else:
                    done[p][r] = True

        return responses, logprobs

    def generate(
        self,
        prompts: list[str],
        n: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        """Sample responses per prompt and return sequence logprobs.

        Args:
            prompts: Input prompt strings.
            n: Number of responses per prompt. Defaults to ``self.group_size``.
               Pass a larger value for the self-critic phase to get more
               diverse rollouts without restarting the worker.
        """
        if n is None or n == self.group_size:
            sampling_params = self.sampling_params
        else:
            from vllm import SamplingParams as _SP
            sampling_params = _SP(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_tokens=self.max_tokens,
                n=n,
                stop=["</answer>"],
                include_stop_str_in_output=True,
                logprobs=1,
            )
        outputs = self.llm.generate(prompts, sampling_params)
        all_responses = []
        all_logprobs = []
        for output in outputs:
            curr_responses = []
            curr_logprobs = []
            for o in output.outputs:
                curr_responses.append(o.text)
                curr_logprobs.append(self._extract_sequence_logprob(o))
            all_responses.append(curr_responses)
            all_logprobs.append(curr_logprobs)
        return all_responses, all_logprobs
