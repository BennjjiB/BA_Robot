import gradio as gr
from gradio import ChatMessage

from foundation_pose.real_sense_reader import RealSenseReader
from client import Client
from transcriber import Transcriber
import numpy as np

BASE_URL = "http://134.2.17.204:5000"
# BASE_URL = "http://127.0.0.1:5000"
client = Client(BASE_URL, None)
transcriber = Transcriber()
webcam = RealSenseReader()

def interact_with_pandabot(prompt, messages):
    messages = messages if messages else []
    messages.append(ChatMessage(role="user", content=prompt))
    yield "", messages
    response = client.send_prompt(prompt)
    messages.append(ChatMessage(role="assistant", content=""))
    for chunk in client.handle_response(response):
        if chunk.get("text"):
            messages[-1] = ChatMessage(role="assistant", content=chunk["text"])
        elif chunk.get("finished"):
            messages.append(ChatMessage(role="assistant", content=""))
        elif chunk.get("tool"):
            messages.pop()
            for tool in chunk["tool"]:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=f"{tool}",
                        metadata={
                            "title": f"🛠️ Used tool {tool['function_name']}"}
                    )
                )
            messages.append(ChatMessage(role="assistant", content=""))
        yield "", messages


def capture_audio(new_chunk, transcript, messages):
    yield gr.skip(), gr.skip()
    if new_chunk:
        new_transcript, start_prompt = transcriber.transcribe_audio(
            new_chunk, transcript)
        if start_prompt:
            yield from interact_with_pandabot(new_transcript, messages)
        elif new_transcript:
            yield new_transcript, gr.skip()
    yield gr.skip(), gr.skip()


def clear_all():
    transcriber.reset()
    client.clear_history()
    return "", []

def get_images():
    color_image, depth_image, depth_colormap = webcam.capture_image()
    # flip images and adjust color bgr to rgb
    color_image = color_image[:, ::-1]
    color_image = color_image[...,::-1]
    
    depth_image = np.clip(depth_image, -1, 1)
    depth_image = depth_image[:, ::-1]
    depth_image = depth_image[...,::-1]

    depth_colormap = depth_colormap[:, ::-1]
    depth_colormap = depth_colormap[...,::-1]
    return color_image, depth_image, depth_colormap, None


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
    gr.Markdown("# Chat with Panda-Bot")
    chatbot = gr.Chatbot(
        value=[],
        type="messages",
        avatar_images=(
            None,
            "panda_avatar.png",
        ),
        layout="bubble",
        placeholder="Start speaking to interact with Panda-Bot",
        show_copy_all_button=True
    )
    text_input = gr.Textbox(
        lines=1,
        label="Chat Message",
        placeholder="Placeholder"
    )
    input_audio = gr.Audio(
        sources=["microphone"],
        label="Input Audio",
        waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
        streaming=True,
        type="numpy"
    )
    with gr.Row():
        clear = gr.Button("Clear", variant="secondary", size="lg")
        submit_button = gr.Button("Submit", variant="primary", size="lg")


    timer = gr.Timer(0.05)
    with gr.Column():
        with gr.Row():
            default_image = gr.Image(label="image")
            depth = gr.Image(label="depth image")
        with gr.Row():
            color_map = gr.Image(label="depth color map")
            yolo_image = gr.Image(label="yolo annotation")
    timer.tick(get_images, None, [default_image, depth, color_map, yolo_image])
    input_audio.stream(
        fn=capture_audio,
        inputs=[input_audio, text_input, chatbot],
        outputs=[text_input, chatbot]
    )

    submit_button.click(
        fn=interact_with_pandabot,
        inputs=[text_input, chatbot],
        outputs=[text_input, chatbot]
    )
    text_input.submit(interact_with_pandabot, [
                      text_input, chatbot], [text_input, chatbot])
    clear.click(fn=clear_all, inputs=[], outputs=[text_input, chatbot])

demo.launch()
