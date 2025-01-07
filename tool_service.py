import json
import threading
from tool_definitions import available_tools
import re


class ToolService():
    def __init__(self, available_tools=available_tools):
        self.available_tools = available_tools

    def parse_and_execute_response(self, tools):
        parsed_tools = self.parse_tools(tools)
        threads = []
        if parsed_tools:
            for tool in parsed_tools:
                function_name = tool["function_name"]
                function_to_call = self.available_tools[function_name]
                function_args = tool["arguments"]
                if "id" in tool:
                    function_args["tool_id"] = tool["id"]
                threads.append(
                    self.start_tool_call(function_to_call, function_args)
                )
        return threads, parsed_tools

    def start_tool_call(self, function_to_call, function_args):
        thread = threading.Thread(target=function_to_call, args=(function_args,))
        thread.start()
        return thread

    def parse_tools(self, tools):
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        tool_call_match = re.findall(tool_call_pattern, tools, re.DOTALL)
        tool_calls = [convert_recursively(match.strip()) for match in tool_call_match]
        return tool_calls

    def get_tool_response_template(self, tool_response):
        dict = {
            "role": "ipython",
            "name": tool_response["name"],
            "content": tool_response["content"],
        }
        if "tool_call_id" in tool_response:
            dict['tool_call_id'] = tool_response["tool_call_id"]
        return json.dumps(dict)


def check_if_tool_call(chunk):
    return chunk.startswith("<tool_call>")


def convert_tool_call_into_chat_message(text):
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_call_match = re.findall(tool_call_pattern, text, re.DOTALL)
    tool_calls = [json.loads(match.strip()) for match in tool_call_match]
    return [{"type": "function", "function": tool_call} for tool_call in tool_calls]


def convert_recursively(data):
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            pass
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_recursively(value)
    elif isinstance(data, list):
        data = [convert_recursively(item) for item in data]
    return data
