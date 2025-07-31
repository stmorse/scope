"""Contains classes for querying Ollama large language models, mirroring Online_LLM."""

from typing import List
from agent.LLM import LLM
from ollama import Client as OllamaClient

class Ollama_LLM(LLM):
    def __init__(self, model_config, **kwargs):
        self.model_name = model_config["name"]
        self.tokenizer_has_system_prompt = True
        self.generation_config = model_config["generation_config"]
        self.system_prompt = model_config["sys_prompt"]
        # Use the specified base URL for the Ollama service
        # OllamaClient expects 'host', not 'base_url'
        self.client = OllamaClient(host=model_config.get("base_url", "http://ollama-brewster:80"))

    def generate(self, chat: List[dict], **kwargs) -> List[str]:
        print("generating responses in ollama...")
        print(chat)
        # Ollama expects a list of messages (like OpenAI)
        response = self.client.chat(
            model=self.model_name,
            messages=chat,
            options=self.generation_config,
            **kwargs
        )
        # Ollama returns a dict with a 'message' key for the last message
        # and possibly 'choices' for multi-response models (mimic OpenAI)
        if "choices" in response:
            output = [choice["message"]["content"] for choice in response["choices"]]
        elif "message" in response:
            output = [response["message"]["content"]]
        else:
            output = [str(response)]
        return output
