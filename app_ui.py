import gradio as gr
import chatbot_ui
from client import Client
from robot_interface import RobotInterface
import robot_ui

BASE_URL = "http://134.2.17.204:5000"
# BASE_URL = "http://127.0.0.1:5000"
client = Client(BASE_URL, None)
robot_interface = RobotInterface()


css = """
.message.pending {
    display: none;
}
"""

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
with gr.Blocks(title="Panda-Bot", css=css, fill_height=True, js=js_func) as demo:
    chatbot_ui.chatbot_ui(client)
    with gr.Accordion(label="Robot Controls", visible=True):
        robot_ui.robot_ui(robot_interface)


demo.launch(favicon_path="panda_avatar.png")
