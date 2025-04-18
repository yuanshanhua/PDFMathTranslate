import asyncio
import logging
import os
import re
import time
import unicodedata
from copy import copy
from string import Template
from typing import cast

import openai
from aiolimiter import AsyncLimiter
from openai.types.chat import ChatCompletionMessageParam

from .cache import TranslationCache
from .config import ConfigManager


logger = logging.getLogger(__name__)

_global_loop = None


def get_event_loop():
    global _global_loop
    if _global_loop is None:
        _global_loop = asyncio.new_event_loop()
    return _global_loop


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class BaseTranslator:
    name = "base"
    envs = {}
    lang_map: dict[str, str] = {}
    CustomPrompt = False

    def __init__(self, lang_in: str, lang_out: str, model: str, ignore_cache: bool):
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.model = model
        self.ignore_cache = ignore_cache

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
                "model": model,
            },
        )

    def set_envs(self, envs):
        # Detach from self.__class__.envs
        # Cannot use self.envs = copy(self.__class__.envs)
        # because if set_envs called twice, the second call will override the first call
        self.envs = copy(self.envs)
        if ConfigManager.get_translator_by_name(self.name):
            self.envs = ConfigManager.get_translator_by_name(self.name)
        needUpdate = False
        for key in self.envs:
            if key in os.environ:
                self.envs[key] = os.environ[key]
                needUpdate = True
        if needUpdate:
            ConfigManager.set_translator_by_name(self.name, self.envs)
        if envs is not None:
            for key in envs:
                self.envs[key] = envs[key]
            ConfigManager.set_translator_by_name(self.name, self.envs)

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    async def translate(self, text: str, ignore_cache: bool = False) -> str:
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                return cache

        translation = await self.do_translate(text)
        self.cache.set(text, translation)
        return translation

    async def do_translate(self, text: str) -> str:
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    def prompt(self, text: str, prompt_template: Template | None = None) -> list[ChatCompletionMessageParam]:
        try:
            return [
                {
                    "role": "user",
                    "content": cast(Template, prompt_template).safe_substitute(
                        {
                            "lang_in": self.lang_in,
                            "lang_out": self.lang_out,
                            "text": text,
                        }
                    ),
                }
            ]
        except AttributeError:  # `prompt_template` is None
            pass
        except Exception:
            logging.exception("Error parsing prompt, use the default prompt.")

        return [
            {
                "role": "user",
                "content": (
                    "You are a professional, authentic machine translation engine. "
                    "Only Output the translated text, do not include any other text."
                    "\n\n"
                    f"Translate the following markdown source text to {self.lang_out}. "
                    "Keep the formula notation {v*} unchanged. "
                    "Output translation directly without any additional text."
                    "\n\n"
                    f"Source Text: {text}"
                    "\n\n"
                    "Translated Text:"
                ),
            },
        ]

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, id: int):
        return f"<b{id}>"

    def get_rich_text_right_placeholder(self, id: int):
        return f"</b{id}>"

    def get_formular_placeholder(self, id: int):
        return self.get_rich_text_left_placeholder(id) + self.get_rich_text_right_placeholder(id)


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"
    envs = {
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_KEY": None,
        "OPENAI_MODEL": "gpt-4o-mini",
        "OPENAI_RATE": "5",  # QPS
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
        req_rate=5,
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["OPENAI_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.client = openai.AsyncOpenAI(
            base_url=base_url or self.envs["OPENAI_BASE_URL"],
            api_key=api_key or self.envs["OPENAI_API_KEY"],
        )
        self.limiter = AsyncLimiter(req_rate or self.envs["OPENAI_RATE"], 1)
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", 0)
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))
        think_filter_regex = r"^<think>.+?\n*(</think>|\n)*(</think>)\n*"
        self.add_cache_impact_parameters("think_filter_regex", think_filter_regex)
        self.think_filter_regex = re.compile(think_filter_regex, flags=re.DOTALL)

    async def do_translate(self, text) -> str:
        async with self.limiter:
            count = 1
            while True:
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        temperature=0,  # 随机采样可能会打断公式标记
                        messages=self.prompt(text, self.prompttext),
                    )
                    break
                except Exception as e:
                    logger.warning(f"request LLM API failed: {e}")
                    logger.warning(f"wait and retry in {count}s")
                    time.sleep(count)
                    count *= 2
        if not response.choices:
            if hasattr(response, "error"):
                raise ValueError("Error response from Service", response.error)
        _ = getattr(response.choices[0].message, "reasoning_content", None)
        content = response.choices[0].message.content.strip()
        content = self.think_filter_regex.sub("", content).strip()
        return content

    def get_formular_placeholder(self, id: int):
        return "{{v" + str(id) + "}}"

    def get_rich_text_left_placeholder(self, id: int):
        return self.get_formular_placeholder(id)

    def get_rich_text_right_placeholder(self, id: int):
        return self.get_formular_placeholder(id + 1)
