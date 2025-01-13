from flask import Flask, Response, request, jsonify
from tool_definitions import tool_definitions
from chatbot import LLAMA_31_8, PandaChatBot, LLAMA_32, LLAMA_31_70, LLAMA_33
from groq_bot import GroqChatBot
from transcriber import Transcriber
import numpy as np

# setup_prompt = """
# You are a calculator assistant.
# You can use the 'calculator' function to perform mathematical calculations.
# Give intermediate status updates if provided by the 'calculator' function,
# like the calculator is booting or the calculator stopped.
# """

# setup_prompt = "You're a helpful assistant."

setup_prompt = """
You're name is Panda. A robot developed by Franka Robotics.
You have a movable arm and a gripper attached at the tip of your arm.
You're specific task is grasping and sorting colored duplo bricks.
Be friendly and humorous, you especially like panda bears!

# Tool instructions
Based on the task, you will need to make make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given task lacks the parameters required by the function,
also point it out.

When you receive a tool call response, use the output to format an answer to the orginal task.
You can recieve multiple tool call response for the same tool call, do NOT call the same function
again unless it finished the task. 
"""

app = Flask(__name__)
bot = GroqChatBot(model="llama-3.3-70b-specdec", setup_prompt=setup_prompt, tools=tool_definitions)
# bot = PandaChatBot(LLAMA_31_8, setup_prompt=setup_prompt)
transcriber = Transcriber()


@app.route("/")
def hello_world():
    return "<p>Welcome to the panda bot api!</p>"


@app.route("/clear")
def clear():
    bot.clear_history(setup_prompt)
    return "cleared history"


@app.route('/get-response')
def generate_response_stream():
    prompt = request.args.get('prompt')
    is_tool_response = request.args.get('is_tool_response')
    return Response(bot.generate_chat_response(user_input=prompt, is_tool_response=is_tool_response),
                    mimetype="text/event-stream")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    audio_data = np.array(data['audio_data'])
    sample_rate = data['sample_rate']
    transcription, sendPrompt = transcriber.transcribe_audio(audio_data, sample_rate)
    return jsonify({"sendPrompt": sendPrompt, "transcription": transcription})
