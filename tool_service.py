import json
import threading
import re
from robot_interface import RobotInterface


class ToolService():
    def __init__(self, robot_interface: RobotInterface):
        self.available_tools = {
            "sort_all_bricks": robot_interface.sort_bricks,
            "get_collision_free_bricks": robot_interface.display_collision_free_bricks,
            "grab_brick": robot_interface.grab_brick
        }

    def parse_and_execute_response(self, tools):
        parsed_tools = self.parse_tools(tools)
        thread = None
        if parsed_tools:
            thread = threading.Thread(
                target=self.start_tool_calls(), args=(parsed_tools)
            )
            thread.start()
        return thread, parsed_tools

    def start_tool_calls(self, parsed_tools):
        for tool in parsed_tools:
            function_name = tool["function_name"]
            function_to_call = self.available_tools.get(function_name, None)
            if function_to_call is None:
                print(function_to_call, " is not a defined function")
                return
            function_args = tool["arguments"]
            if "id" in tool:
                function_args["tool_id"] = tool["id"]
            function_to_call(**function_args)

    def parse_tools(self, tools):
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        tool_call_match = re.findall(tool_call_pattern, tools, re.DOTALL)
        tool_calls = [convert_recursively(match.strip())
                      for match in tool_call_match]
        return tool_calls

    def get_tool_response_template(self, tool_response):
        dict = {
            "role": "tool",
            "name": tool_response["name"],
            "content": tool_response["content"],
        }
        if "tool_call_id" in tool_response:
            dict['tool_call_id'] = tool_response["tool_call_id"]
        return json.dumps(dict)


def check_if_tool_call(chunk):
    return chunk.startswith("<tool_call>")


def convert_recursively(data):
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            pass
    elif isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_recursively(value)
    elif isinstance(data, list):
        data = [convert_recursively(item) for item in data]
    return data
