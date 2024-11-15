import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests.exceptions
from retry import retry
from together import Together

LOGGER = logging.getLogger(__name__)


@retry(exceptions=requests.exceptions.RequestException, tries=3, delay=2, backoff=2)
def generate_response(
    client: Together,
    model: str,
    messages: List[str],
    temperature: float = 0.8,
    max_tokens: int = 256,
    seed: int = 42,
    **kwargs,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=1,
        logits=1,
        seed=seed,
        **kwargs,
    )
    return response


class VFT:
    def __init__(
        self,
        model_a: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        model_b: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        system_prompt_a: str = "Do nothing.",
        system_prompt_b: str = "Do nothing.",
        num_turns: int = 30,
        num_rounds: int = 1,
        together_api: bool = True,
        together_api_key: str | None = None,
        model_a_kargs=None,
        model_b_kargs=None,
        seed: int = 42,
    ) -> None:
        if model_a_kargs is None:
            model_a_kargs = {}
        self.model_a_kargs = model_a_kargs
        if model_b_kargs is None:
            model_b_kargs = {}
        self.model_b_kargs = model_b_kargs
        if together_api:
            if together_api_key is None:
                raise ValueError(
                    "together_api_key must be provided if together_api is True"
                )
            self.client = Together(api_key=together_api_key)
        else:
            self.client = None
        self.model_a = model_a
        self.system_prompt_a = system_prompt_a
        self.system_prompt_b = system_prompt_b
        self.model_b = model_b
        self.current_turn = 0
        self.num_turns = num_turns
        self.num_rounds = num_rounds
        self.messages = []
        self.seed = seed

    def preprocess_response(self, response):
        # remove <execute_code> and </execute_code>
        return response

    def generate_model_response(self, model_dyad_name):
        if model_dyad_name in ["a", "A"]:
            model = self.model_a
            system_prompt = self.system_prompt_a
            kargs = self.model_a_kargs
        elif model_dyad_name in ["b", "B"]:
            model = self.model_b
            system_prompt = self.system_prompt_b
            kargs = self.model_b_kargs
        else:
            raise ValueError(f"Invalid model dyad name: {model_dyad_name}")
        # format into content messages
        system_message = {"role": "system", "content": system_prompt}
        messages = [system_message] + [
            {"role": "user", "content": message["message"].choices[0].message.content}
            for message in self.messages
        ]
        response = generate_response(
            client=self.client, model=model, messages=messages, seed=self.seed, **kargs
        )
        return response

    def step(self, model_dyad_name):
        message = self.generate_model_response(model_dyad_name)
        if model_dyad_name in ["a", "A"]:
            self.messages.append({"name": model_dyad_name, "message": message})
        elif model_dyad_name in ["b", "B"]:
            self.messages.append({"name": model_dyad_name, "message": message})
        else:
            raise ValueError(f"Invalid model dyad name: {model_dyad_name}")

    def get_results(self, round_name):
        results = []
        for i, message in enumerate(self.messages):
            if message["name"] in ["a", "A"]:
                model = self.model_a
            elif message["name"] in ["b", "B"]:
                model = self.model_b
            else:
                raise ValueError(f"Invalid model dyad name: {m['name']}")
            m = message["message"]
            content = m.choices[0].message.content
            seed = m.choices[0].seed
            tokens = m.choices[0].logprobs.tokens
            token_logprobs = m.choices[0].logprobs.token_logprobs
            token_ids = m.choices[0].logprobs.token_ids
            im_results = {
                "round": round_name,
                "model": model,
                "name": message["name"],
                "content": content,
                "seed": seed,
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "token_ids": token_ids,
                "index": i,
            }
            results.append(im_results)
        return pd.DataFrame(results)

    def save_results(self, path, round_name):
        results = self.get_results(round_name)
        results.to_csv(path, index=False)

    def start(self, n_rounds=1):
        for round_index in range(n_rounds):
            LOGGER.info(f"Starting round {round_index}")
            print(f"Starting round {round_index}")
            # reset messages
            self.current_turn = 0
            while self.current_turn < self.num_turns:
                print(f"Turn {self.current_turn}")
                if self.current_turn % 2 == 0:
                    self.step("a")
                else:
                    self.step("b")
                self.current_turn += 1
            # get results
            results = self.save_results(
                f"round_{round_index}.csv", round_name=round_index
            )
            # reset messages
            self.messages = []
            # update seed
            self.seed += 1
