from status_helper import update_status
import json
import time


def calculate(args):
    """
    Evaluate a mathematical expression

     Args:
        args (dict): Contains "expression" (str), the math expression to evaluate.
    """
    tool_id = args.get("tool_id", None)
    update_status("calculate", "Calculator is booting...", tool_id)
    try:
        result = eval(args.get("expression"))
        print(result)
        update_status("calculate", json.dumps({"result": result}), tool_id)
    except Exception as e:
        update_status("calculate", json.dumps({"error": f"Unexpected error: {str(e)}"}), id)


def sort_bricks(args):
    """
    Evaluate a mathematical expression

     Args:
        args (dict): Contains "expression" (str), the math expression to evaluate.
    """
    tool_id = args.get("tool_id", None)
    update_status("sort_bricks", "Starting to detect bricks", tool_id)
    time.sleep(1)
    update_status("sort_bricks", "Sorted all blue bricks", tool_id)
    time.sleep(2)
    update_status("sort_bricks", "Sorted all red bricks", tool_id)
    update_status("sort_bricks", "Finished sorting all bricks", tool_id)


calculate_definition = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Note the calculator first has to boot!",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    },
}

sort_bricks_definition = {
    "type": "function",
    "function": {
        "name": "sort_bricks",
        "description": "Sorts all bricks by color.",
        "parameters": {},
    },
}

tool_definitions = [calculate_definition, sort_bricks_definition]

available_tools = {
    "calculate": calculate,
    "sort_bricks": sort_bricks
}
