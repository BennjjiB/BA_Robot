import os
import json
from groq import Groq
from chatbot import AbstractChatBot


class GroqChatBot(AbstractChatBot):
    def __init__(self, model="llama3-groq-70b-8192-tool-use-preview", setup_prompt="", tools=[]):
        self.client = Groq(
            api_key="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",
        )
        self.model = model
        super().__init__(setup_prompt=setup_prompt, tools=tools)

    def get_response_streamer(
            self, query, max_tokens=1024, temperature=0.6, top_p=0.9
    ):
        self.conversation.append(query)
        response = self.client.chat.completions.create(
            messages=self.conversation,
            model=self.model,
            tools=self.tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=None,
            tool_choice="auto",
            stream=False,
        )
        return response.choices[0].message

    def generate_chat_response(self, user_input: str, is_tool_response: bool = False):
        if is_tool_response is True or is_tool_response.strip() == "True":
            query = json.loads(user_input)
            # replace ipython with groq tool role
            query["role"] = "tool"
        else:
            query = {"role": "user", "content": user_input}
        response_message = self.get_response_streamer(query)
        tool_calls = response_message.tool_calls
        if tool_calls:
            tools = [{"id": tool_call.id, "function_name": tool_call.function.name,
                      "arguments": tool_call.function.arguments}
                     for tool_call in tool_calls]
            self.conversation.append(response_message)
            for tool in tools:
                print(tool)
                yield f"<tool_call>{json.dumps(tool)}</tool_call>"
        else:
            yield response_message.content
            self.conversation.append(
                {"role": "assistant", "content": response_message.content}
            )
