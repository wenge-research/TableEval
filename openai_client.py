# -*- encoding: utf-8 -*-
from datetime import datetime
import logging
from openai import OpenAI
import os
from typing import Any, Optional, List
from utils import *
from inspect import signature

class OpenAIClient:
    """
    OpenAI-Compatible chat completions API
    """
    def __init__(
        self,
        model: str,
        config_file: str
    ):
        self.api_config = load_api_configuration(model, config_file)
        base_url = self.api_config.get("base_url")
        api_key = self.api_config.get("api_key")
    
        self.timeout = self.api_config.get('timeout', 600)
        self.max_retries = self.api_config.get('max_retries', 3)

        if base_url:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=self.max_retries,
                timeout=self.timeout
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                max_retries=self.max_retries,
                timeout=self.timeout
            )
        params = signature(self.client.chat.completions.create)
        self.params_supported = list(params.parameters.keys())

    def generate_response(
        self,
        messages: List[dict]
    ) -> Optional[str]:
        """
        Generate a response based on the incoming message and return the reply content; return None upon failure.
        """
        try:
            request_json = {'model': self.api_config.get("model_name"), 'messages': messages}
            for key, value in self.api_config.items():
                if key in self.params_supported and key not in request_json:
                    request_json[key] = value

            response = self.client.chat.completions.create(**request_json)
            response_dict= response.model_dump()
            choice = response_dict.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason")
            response_content = choice.get("message", {}).get("content")
            if response_content is not None:
                response_content = response_content.strip()

            if finish_reason != "stop":
                logging.warning(f"OpenAI API warning: token truncated")
            return response_content
        except Exception as e:
            logging.error(f"Model Request failed: {e}")
            return None



def main() -> None:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    model = "gpt-4o-2024-11-20"
    config_file = "./config/api_config.yaml"
    client = OpenAIClient(model=model, config_file=config_file)

    sample_messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': 'Hello!'}
    ]
    client.generate_response(sample_messages)

    
if __name__ == "__main__":
    main()
