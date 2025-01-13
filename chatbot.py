from tool_service import convert_tool_call_into_chat_message, check_if_tool_call
from typing import Literal, List
from abc import ABC, abstractmethod
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
import json
import os
import torch
torch.cuda.empty_cache()


LLAMA_31_8 = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_31_70 = "meta-llama/Llama-3.1-70B-Instruct"
LLAMA_32 = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_33 = "meta-llama/Llama-3.3-70B-Instruct"


class AbstractChatBot(ABC):
    def __init__(self, setup_prompt: str = "You are a helpful assistant", tools: List = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Running on:', self.device)
        self.tools = tools
        self.setup(setup_prompt)

    def setup(self, system_instructions):
        self.conversation = []
        self.conversation.append(
            {"role": "system", "content": system_instructions})

    @abstractmethod
    def get_response_streamer(self, query: str, max_tokens: int = 1028, temperature: float = 0.6,
                              top_p: float = 0.9):
        """
        Returns a response streamer for generating chatbot responses.
        """
        pass

    def generate_chat_response(self, user_input: str, is_tool_response: bool = False):
        if is_tool_response is True or is_tool_response.strip() == "True":
            query = json.loads(user_input)
        else:
            query = {"role": "user", "content": user_input}
        streamer = self.get_response_streamer(query)
        generated_response = ""
        for response in streamer:
            generated_response += response
            yield response
        is_tool_call = check_if_tool_call(generated_response)
        field_name = "tool_calls" if is_tool_call else "content"
        if is_tool_call:
            print(generated_response)
            generated_response = convert_tool_call_into_chat_message(
                generated_response)
            print(generated_response)
        self.conversation.append(
            {"role": "assistant", field_name: generated_response})

    def clear_history(self, setup_prompt):
        self.conversation = []
        self.setup(setup_prompt)

    def chatbot(self, system_instructions: str = ""):
        """
        Generates a chatbot interface. Useful for texting the llm.
        """
        self.setup(system_instructions)
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break
            generated_response = ""
            for response in self.generate_chat_response(user_input):
                generated_response += response
                if (response.strip()):
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("User input:", user_input)
                    print("Llama response:", generated_response)
                    print('', end='', flush=True)
            print()


# ---------------------------------------------------------
class PandaChatBot(AbstractChatBot):
    def __init__(
            self,
            model_path: str,
            quantization: Literal["16bit", "8bit", "4bit"] = "16bit",
            setup_prompt: str = "You are a helpful assistant",
            tools=None
    ):
        super().__init__(setup_prompt=setup_prompt, tools=tools)
        # Quantitation
        quantization_config = None
        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_response_streamer(
            self, query, max_tokens=1028, temperature=0.6, top_p=0.9
    ):
        self.conversation.append(query)
        # Create tokenized prompt
        prompt = self.tokenizer.apply_chat_template(
            self.conversation,
            tools=self.tools,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        prompt.to(self.device)
        # Create text streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            timeout=10,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            inputs=prompt["input_ids"],
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            attention_mask=prompt["attention_mask"]
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer
