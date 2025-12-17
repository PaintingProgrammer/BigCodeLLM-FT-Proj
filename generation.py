# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str
    destination: str  # required for model responses


class InfillingPrediction(TypedDict, total=False):
    generation: str
    full_text: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>", "<step>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
    ) -> "Llama":
        if not torch.distributed.is_initialized():
            if device == "cuda":
                torch.distributed.init_process_group("nccl")
            else:
                torch.distributed.init_process_group("gloo")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device == "cuda":
            torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        # support for mac
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            else:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        stop_token: Optional[int] = None,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        if stop_token is None:
            stop_token = self.tokenizer.eos_id
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        if echo:
            prev_pos = 0
        else:
            prev_pos = min_prompt_len
        cur_pos = min_prompt_len
        for _ in range(max_gen_len):
            with torch.no_grad():
                logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                tokens[:, cur_pos - 1] == pad_id, tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if echo:
                prev_pos = cur_pos
            cur_pos = cur_pos + 1
            if stop_token is not None and torch.all(next_token == stop_token):
                break

        out_tokens = []
        out_logprobs = [] if logprobs else None
        for prompt_idx, prompt in enumerate(prompt_tokens):
            if echo:
                start = 0
                generated = tokens[prompt_idx, cur_pos:]
            else:
                start = len(prompt)
                generated = tokens[prompt_idx, start:cur_pos]
            out_tokens.append(generated.tolist())
            if logprobs:
                out_logprobs.append(
                    [
                        logprobs_list[prompt_idx][idx].item()
                        for idx in range(start, cur_pos)
                    ]
                )

        return (out_tokens, out_logprobs)

    @torch.inference_mode()
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - len(prompts[0])
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens,
            max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_t,
                }
                for t, logprobs_t in zip(generation_tokens, generation_logprobs)
            ]
        else:
            return [self.tokenizer.decode(t) for t in generation_tokens]

    @torch.inference_mode()
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ):
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len
        assert len(dialogs) > 0
        assert all(["role" in msg for dialog in dialogs for msg in dialog])
        # Pre tokenize the prompts
        prompt_tokens = []
        for dialog in dialogs:
            if dialog[0]["role"] == "system":
                assert len(dialog) >= 2 and dialog[1]["role"] == "user"
                dialog = [
                    {"role": dialog[1]["role"], "content": dialog[1]["content"]},
                    {"role": "assistant", "content": dialog[0]["content"]},
                ] + dialog[2:]
            else:
                assert dialog[0]["role"] == "user"
            # Format the prompt
            formatted = (
                f"{B_INST} {(dialog[0]["content"]).strip()} {E_INST} {(dialog[1]["content"]).strip()}"
            )
            assert len(dialog) % 2 == 1
            for i in range(2, len(dialog), 2):
                formatted += (
                    f" {B_INST} {(dialog[i]["content"]).strip()} {E_INST} {(dialog[i+1]["content"]).strip()}"
                )
            prompt_tokens.append(self.tokenizer.encode(formatted, bos=True, eos=True))

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_t,
                }
                for t, logprobs_t in zip(generation_tokens, generation_logprobs)
            ]
        else:
            return [
                {"generation": {"role": "assistant", "content": self.tokenizer.decode(t)}}
                for t in generation_tokens
            ]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
