import queue

status_queue = queue.Queue()


def update_status(function_name, new_status, id=None):
    """Function to immediately update the status."""
    status = {
        "name": function_name,
        "content": new_status,
    }
    if id:
        status["tool_call_id"] = id
    status_queue.put(status)
