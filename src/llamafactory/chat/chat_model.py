# Copyright 2025 THUDM and the LlamaFactory team.
#
# This code is inspired by the THUDM's ChatGLM implementation.
# https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

from ..extras.constants import EngineName
from ..extras.misc import torch_gc
from ..hparams import get_infer_args


if TYPE_CHECKING:
    from ..data.mm_plugin import AudioInput, ImageInput, VideoInput
    from .base_engine import BaseEngine, Response


def _start_background_loop(loop: "asyncio.AbstractEventLoop") -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    r"""General class for chat models. Backed by huggingface or vllm engines.

    Supports both sync and async methods.
    Sync methods: chat(), stream_chat() and get_scores().
    Async methods: achat(), astream_chat() and aget_scores().
    """

    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        self.data_args = data_args  # Store for access in run_chat

        if model_args.infer_backend == EngineName.HF:
            from .hf_engine import HuggingfaceEngine

            self.engine: BaseEngine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        elif model_args.infer_backend == EngineName.VLLM:
            try:
                from .vllm_engine import VllmEngine

                self.engine: BaseEngine = VllmEngine(model_args, data_args, finetuning_args, generating_args)
            except ImportError as e:
                raise ImportError(
                    "vLLM not install, you may need to run `pip install vllm`\n"
                    "or try to use HuggingFace backend: --infer_backend huggingface"
                ) from e
        elif model_args.infer_backend == EngineName.SGLANG:
            try:
                from .sglang_engine import SGLangEngine

                self.engine: BaseEngine = SGLangEngine(model_args, data_args, finetuning_args, generating_args)
            except ImportError as e:
                raise ImportError(
                    "SGLang not install, you may need to run `pip install sglang[all]`\n"
                    "or try to use HuggingFace backend: --infer_backend huggingface"
                ) from e
        elif model_args.infer_backend == EngineName.KT:
            try:
                from .kt_engine import KTransformersEngine

                self.engine: BaseEngine = KTransformersEngine(model_args, data_args, finetuning_args, generating_args)
            except ImportError as e:
                raise ImportError(
                    "KTransformers not install, you may need to run `pip install ktransformers`\n"
                    "or try to use HuggingFace backend: --infer_backend huggingface"
                ) from e
        else:
            raise NotImplementedError(f"Unknown backend: {model_args.infer_backend}")

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        r"""Get a list of responses of the chat model."""
        task = asyncio.run_coroutine_threadsafe(
            self.achat(messages, system, tools, images, videos, audios, **input_kwargs), self._loop
        )
        return task.result()

    async def achat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        r"""Asynchronously get a list of responses of the chat model."""
        return await self.engine.chat(messages, system, tools, images, videos, audios, **input_kwargs)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        r"""Get the response token-by-token of the chat model."""
        generator = self.astream_chat(messages, system, tools, images, videos, audios, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        r"""Asynchronously get the response token-by-token of the chat model."""
        async for new_token in self.engine.stream_chat(
            messages, system, tools, images, videos, audios, **input_kwargs
        ):
            yield new_token

    def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        r"""Get a list of scores of the reward model."""
        task = asyncio.run_coroutine_threadsafe(self.aget_scores(batch_input, **input_kwargs), self._loop)
        return task.result()

    async def aget_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        r"""Asynchronously get a list of scores of the reward model."""
        return await self.engine.get_scores(batch_input, **input_kwargs)


def run_chat() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    chat_model = ChatModel()
    messages = []
    
    # Define system prompts based on question type
    greeting_system = "Respond to user greetings and introduce yourself as HAI Reach agent."
    rag_system = "You are Hai Indexer, an AI assistant that helps users find information from their indexed documents. Answer questions using the provided context when available, or use your general knowledge for questions not covered in the documents."
    
    # Use default_system from config if available, otherwise use rag_system as default
    default_system = getattr(chat_model.data_args, "default_system", None) or rag_system
    
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            torch_gc()
            print("History has been removed.")
            continue

        # Determine if it's a greeting or RAG question
        query_lower = query.strip().lower()
        is_greeting = any(
            query_lower.startswith(greeting) 
            for greeting in ["good morning", "good afternoon", "good evening", "greetings", "hi", "hello", "hey"]
        ) or query_lower in ["hi", "hello", "hey"]
        
        # Format message to match training format: instruction + "\n" + input
        if is_greeting:
            # For greetings: instruction + "\n" + "" (empty input becomes "\n\n")
            formatted_query = f"{query}\n\n"
            system = greeting_system
        else:
            # For RAG questions: instruction + "\n" + context
            # Note: In a real RAG system, you would retrieve context here
            # For now, we'll format it as the model expects but without context
            # The model should handle questions without context based on its training
            formatted_query = f"{query}\n\n"
            system = default_system
        
        messages.append({"role": "user", "content": formatted_query})
        print("Assistant: ", end="", flush=True)

        response = ""
        # Pass system prompt to stream_chat
        for new_text in chat_model.stream_chat(messages, system=system):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})
