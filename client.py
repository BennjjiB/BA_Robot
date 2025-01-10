import requests
import queue
from status_helper import status_queue
from tool_service import ToolService, check_if_tool_call
import numpy as np
import base64
import json

class Client:
    def __init__(self, base_url: str, tool_service: ToolService = ToolService()) -> None:
        self.base_url = base_url
        self.tool_service = tool_service

    def clear_history(self):
        endpoint = f"{self.base_url}/clear"
        requests.get(endpoint)

    def send_prompt(self, prompt: str, tool_response=False):
        """
        Sends a GET request to the server and processes the response.

        Args:
            base_url (str): The server's base URL.
            prompt (str): The transcription prompt to be processed.
        """
        endpoint = f"{self.base_url}/get-response"
        params = {"prompt": prompt, "is_tool_response": tool_response}
        with requests.get(endpoint, params=params, stream=True, timeout=10) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                return
            for chunk in response.iter_content():
                r = chunk.decode(errors='replace')
                yield r
    
    def send_audio(self, audio_data, sample_rate):
        payload = {
            'audio_data': audio_data.tolist(),
            'sample_rate': sample_rate
        }
        endpoint = f"{self.base_url}/transcribe"
        response = requests.post(endpoint, json=payload).json()
        return response.get('transcription'), response.get('sendPrompt')

    def stream_transcription(self, transcript):
        self.transcript = transcript

    def handle_response(self, response):
        generated_response = ""
        for r in response:
            generated_response += r
            if not check_if_tool_call(generated_response):
                yield {"text": generated_response}
        if generated_response:
            yield {"finished": True}
        tool_threads, parsed_tools = self.tool_service.parse_and_execute_response(generated_response)
        if tool_threads:
            yield {"tool": parsed_tools}
            yield from self.handle_tool_status_changes(tool_threads)

    def handle_tool_status_changes(self, threads):
        while any(thread.is_alive() for thread in threads) or not status_queue.empty():
            try:
                new_status = status_queue.get(timeout=1)  # Wait for 1 second for an item
                response = self.send_prompt(
                    self.tool_service.get_tool_response_template(new_status), tool_response=True
                )
                yield from self.handle_response(response)
            except queue.Empty:
                if not any(thread.is_alive() for thread in threads):
                    break
                continue
        print("All threads are finished and the queue is empty. Exiting.")