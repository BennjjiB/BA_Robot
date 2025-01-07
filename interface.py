import gradio as gr
from gradio import ChatMessage

from foundation_pose.PoseEstimationApp import PoseEstimatorApp
from foundation_pose.real_sense_reader import RealSenseReader
from client import Client
from transcriber import Transcriber
import numpy as np
from ultralytics import YOLO

BASE_URL = "http://134.2.17.204:5000"
# BASE_URL = "http://127.0.0.1:5000"
client = Client(BASE_URL, None)
transcriber = Transcriber()
webcam = RealSenseReader()
maskModel = YOLO('/home/panda3/Desktop/Robot_BA/best.pt')
robotInterface = PoseEstimatorApp(reader=webcam, maskModel=maskModel)


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
    registered_bricks = maskModel(color_image, iou=0.9, verbose=False)[0]
    annotated_frame = registered_bricks.plot()
    og_color_image = color_image.copy()
    og_depth_image = depth_image.copy()
    # adjust color bgr to rgb
    color_image = color_image[..., ::-1]

    depth_image = np.clip(depth_image, -1, 1)
    depth_image = depth_image[..., ::-1]

    depth_colormap = depth_colormap[..., ::-1]

    annotated_frame = annotated_frame[..., ::-1]
    
    return color_image, depth_image, depth_colormap, annotated_frame, (og_color_image, og_depth_image), registered_bricks

    # if not registered_bricks:
    #     self.start_estimate = False
    #     self.offset_red = 0
    #     self.offset_orange = 0
    #     self.offset_yellow = 0
    #     self.offset_green = 0
    #     self.offset_blue = 0
    #     self.offset_left = 0
    #     self.offset_right = 0
    # else:
    #     bricks = self.get_brick_poses(
    #         registered_bricks,
    #         registered_bricks.orig_img.shape,
    #         color_image,
    #         depth_image
    #     )


def get_3d_image(images, registered_bricks):
    bricks, image_3d = robotInterface.get_brick_poses(
        registered_bricks,
        registered_bricks.orig_img.shape,
        images[0],
        images[1]
    )
    return image_3d


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
    input_audio.stream(
        fn=capture_audio,
        inputs=[input_audio, text_input, chatbot],
        outputs=[text_input, chatbot]
    )

    with gr.Row():
        clear = gr.Button("Clear", variant="secondary", size="lg")
        submit_button = gr.Button("Submit", variant="primary", size="lg")

    submit_button.click(
        fn=interact_with_pandabot,
        inputs=[text_input, chatbot],
        outputs=[text_input, chatbot]
    )
    text_input.submit(interact_with_pandabot, [
                      text_input, chatbot], [text_input, chatbot])
    clear.click(fn=clear_all, inputs=[], outputs=[text_input, chatbot])

    # Webcam images
    timer = gr.Timer(0.05)
    with gr.Column():
        with gr.Row():
            default_image = gr.Image(label="image")
            depth = gr.Image(label="depth image")
        with gr.Row():
            color_map = gr.Image(label="depth color map")
            yolo_image = gr.Image(label="yolo annotation")
        image_3d = gr.Image(label="3D image")

    images = gr.State()
    registered_bricks = gr.State()
    timer.tick(get_images, None, [
               default_image, depth, color_map, yolo_image, images, registered_bricks])

    with gr.Row():
        sortColorButton = gr.Button(
            "Sort by color", variant="huggingface", size="lg")
        sortSizeButton = gr.Button(
            "Sort by size", variant="huggingface", size="lg")

    sortColorButton.click(lambda: gr.Timer(active=False), None, timer).then(fn=get_3d_image, inputs=[
                          images, registered_bricks], outputs=[image_3d])

    sortSizeButton.click(fn=lambda: True, inputs=[], outputs=[])

demo.launch()
