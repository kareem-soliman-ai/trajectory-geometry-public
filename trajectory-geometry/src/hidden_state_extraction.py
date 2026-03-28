"""Hidden-state extraction pipeline for trajectory geometry experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from .contamination_filter import (
    find_clean_token_boundary,
    parse_numeric_answer,
    truncate_cot_response,
    truncate_direct_response,
    validate_response,
)


@dataclass
class ExtractionConfig:
    """Configuration for prompt construction and greedy generation."""

    max_new_tokens_direct: int = 15
    max_new_tokens_cot: int = 200
    do_sample: bool = False
    temperature: float = 0.0
    pad_to_eos_token: bool = True
    direct_stop_strings: tuple[str, ...] = (
        "\n",
        "Question",
        "Calculate",
        "Let",
        "Step",
        "Because",
    )
    cot_stop_strings: tuple[str, ...] = (
        "Question:",
        "Calculate:",
        "Problem:",
        "\n\nQ:",
        "\n\nCalculate",
        "Let me verify",
        "Alternatively",
        "Double check",
    )


def build_prompt(question: str, condition: str) -> str:
    """Build the public reproduction prompt."""

    if condition == "direct":
        return (
            "Calculate the following arithmetic problems. Answer with ONLY the numerical result.\n\n"
            "Question: 3 + 4\nAnswer: 7\n\n"
            "Question: 10 - 2\nAnswer: 8\n\n"
            f"Question: {question}\nAnswer:"
        )
    return (
        "Calculate the following arithmetic problems step by step. After your working, "
        "write \"ANSWER: \" followed by the number.\n\n"
        "Question: 3 + 4\nWorking: 3 plus 4 is 7. ANSWER: 7\n\n"
        "Question: 10 - 2\nWorking: 10 minus 2 is 8. ANSWER: 8\n\n"
        f"Question: {question}\nWorking:"
    )


def clean_response(text: str, condition: str) -> str:
    """Apply the condition-specific truncation rule."""

    return truncate_direct_response(text) if condition == "direct" else truncate_cot_response(text)


def extract_generation_trajectory(outputs, num_layers: int) -> np.ndarray:
    """Convert `generate(..., output_hidden_states=True)` output to [L, T, D]."""

    token_layer_stacks = []
    for step_hidden_states in outputs.hidden_states:
        layers_at_step = torch.stack([layer[:, -1, :] for layer in step_hidden_states]).squeeze(1)
        token_layer_stacks.append(layers_at_step)
    if not token_layer_stacks:
        return np.empty((num_layers, 0, 0), dtype=np.float32)
    trajectory_gpu = torch.stack(token_layer_stacks, dim=1)
    return trajectory_gpu.cpu().float().numpy()


def generate_and_extract(
    model,
    tokenizer,
    question: str,
    truth: int,
    condition: str,
    config: ExtractionConfig | None = None,
    device=None,
) -> dict:
    """Run greedy decoding and return cleaned text, metadata, and hidden states."""

    config = config or ExtractionConfig()
    prompt = build_prompt(question, condition)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    max_new = config.max_new_tokens_direct if condition == "direct" else config.max_new_tokens_cot
    pad_token_id = tokenizer.eos_token_id if config.pad_to_eos_token else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=config.do_sample,
            temperature=config.temperature,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=pad_token_id,
        )

    generated_tokens = outputs.sequences[0][input_len:]
    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    clean = clean_response(full_response, condition)
    clean_boundary = find_clean_token_boundary(generated_tokens, tokenizer, clean)
    parsed_answer = parse_numeric_answer(clean, condition)
    issues = validate_response(full_response, clean, condition)
    correct = parsed_answer == truth if parsed_answer is not None else False
    trajectory = extract_generation_trajectory(outputs, model.config.num_hidden_layers + 1)

    return {
        "prompt": prompt,
        "condition": condition,
        "response": full_response,
        "clean_response": clean,
        "parsed_answer": parsed_answer,
        "correct": bool(correct),
        "issues": issues,
        "n_generated_tokens": int(len(generated_tokens)),
        "n_clean_tokens": int(clean_boundary),
        "trajectory": trajectory,
    }
