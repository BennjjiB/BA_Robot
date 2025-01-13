sort_bricks_definition = {
    "type": "function",
    "function": {
        "name": "sort_all_bricks",
        "description": "Sorts all bricks by color or size.",
        "parameters": {
            "type": "object",
            "properties": {
                "by_color": {
                    "type": "bool",
                    "description": "If true, bricks will be sorted by color."
                }
            },
            "required": [
                "by_color"
            ]
        }
    }
}

tool_definitions = [sort_bricks_definition]


