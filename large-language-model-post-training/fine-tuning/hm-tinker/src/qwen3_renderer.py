# https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/renderers.py

import logging
from typing import TypedDict

import tinker
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class Message(TypedDict, total=False):
    role: str
    content: str


def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


class Qwen3Renderer:
    """
    Renderer for Qwen3 models using the chat template format.

    Format:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        <think>
        ...
        </think>
        Hi there!<|im_end|>
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        self._bos_tokens: list[int] = []  # No BOS token for Qwen

    def _render_message(
        self,
        index: int,
        message: Message,
    ) -> tuple[list[int], list[int], list[int]]:
        maybe_newline = "\n" if index > 0 else ""
        observation_string = f"{maybe_newline}<|im_start|>{message['role']}\n"

        action_content = message.get("content", "")
        if message["role"] == "assistant" and "</think>" in action_content:
            # Multi-turn conversation, remove the thinking section from the assistant message
            action_content = action_content.split("</think>")[1].lstrip()
        elif message["role"] == "assistant" and "<think>" not in action_content:
            # Force the assistant to start with <think> if not already present
            observation_string += "<think>\n"

        action_content += "<|im_end|>"

        observation_tokens = self.tokenizer.encode(
            observation_string,
            add_special_tokens=False,
        )
        action_tokens = self.tokenizer.encode(action_content, add_special_tokens=False)
        action_tail_tokens: list[int] = []  # No action tail needed for Qwen format

        return observation_tokens, action_tokens, action_tail_tokens

    def build_supervised_example(
        self,
        messages: list[Message],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build tokens and weights for supervised fine-tuning.

        Only the last assistant message gets weight=1 (trained on).
        All other tokens get weight=0 (context only).
        """
        tokens_weights: list[tuple[int, int]] = []

        for index, message in enumerate(messages):
            is_last_message = index == len(messages) - 1
            is_assistant = message["role"] == "assistant"

            observation_tokens, action_tokens, action_tail_tokens = (
                self._render_message(
                    index,
                    message,
                )
            )

            tokens_weights.extend((token, 0) for token in observation_tokens)

            if is_last_message:
                action_tokens = action_tokens + action_tail_tokens

            action_weight = 1 if (is_last_message and is_assistant) else 0
            tokens_weights.extend((token, action_weight) for token in action_tokens)

        tokens, weights = zip(*tokens_weights, strict=True)
        return torch.tensor(tokens), torch.tensor(weights)

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        """Build a prompt for generation by rendering all messages and adding assistant header."""
        tokens: list[int] = []
        tokens.extend(self._bos_tokens)  # No BOS for Qwen, but keep for consistency

        for index, message in enumerate(messages):
            observation_tokens, action_tokens, _ = self._render_message(index, message)
            tokens.extend(observation_tokens)
            tokens.extend(action_tokens)

        # Add generation prompt
        new_message = Message(role=role, content="")
        observation_tokens, _, _ = self._render_message(len(messages), new_message)
        tokens.extend(observation_tokens)
        tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))

        return tinker.ModelInput.from_ints(tokens)

    @property
    def _end_message_token(self) -> int:
        """Get the end-of-message token ID."""
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, (
            f"Expected single token for <|im_end|>, got {len(tokens)}"
        )
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        """Return the end-of-message token for Qwen3."""
        return [self._end_message_token]

    def parse_response(self, response_tokens: list[int]) -> tuple[Message, bool]:
        """
        Parse generated tokens into a message.

        Returns:
            tuple: (Message, parse_success)
                - Message with role="assistant" and content
                - parse_success is True if response has exactly one stop token

        """
        end_token = self._end_message_token
        end_token_count = response_tokens.count(end_token)

        if end_token_count == 0:
            string_response = self.tokenizer.decode(response_tokens)
            logger.debug(
                f"Response is not a valid assistant response: {string_response}",
            )
            return Message(role="assistant", content=string_response), False

        if end_token_count == 1:
            end_index = response_tokens.index(end_token)
            string_response = self.tokenizer.decode(response_tokens[:end_index])
            return Message(role="assistant", content=string_response), True

        msg = (
            f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {end_token_count}. "
            "You probably are using the wrong stop tokens when sampling"
        )
        raise ValueError(msg)
