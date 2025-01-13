import gradio as gr
from robot_interface import RobotInterface


def robot_ui(robot_interface: RobotInterface):
    show_webcam = gr.State(False)
    image_3d = gr.State(None)

    with gr.Row():
        toggle_webcam_button = gr.Button(
            "Show / Hide webcam", variant="huggingface", size="lg")
        toggle_3d_image = gr.Button(
            "Show / Update 3d image", variant="huggingface", size="lg")
        toggle_webcam_button.click(lambda show: (not show, None), show_webcam, [
                                   show_webcam, image_3d])
        toggle_3d_image.click(fn=lambda: (False, robot_interface.get_3d_image(
        )), inputs=None, outputs=[show_webcam, image_3d])

    @gr.render(inputs=[show_webcam, image_3d])
    def webcam_view(show_webcam, image_3d):
        if show_webcam and image_3d is None:
            timer = gr.Timer(0.05)
            with gr.Column():
                with gr.Row():
                    default_image = gr.Image(label="image")
                    depth = gr.Image(label="depth image")
                with gr.Row():
                    color_map = gr.Image(label="depth color map")
                    yolo_image = gr.Image(label="yolo annotation")
            timer.tick(robot_interface.get_images, None, [
                default_image, depth, color_map, yolo_image])
        else:
            pass

    @gr.render(inputs=[show_webcam, image_3d])
    def image_3d_view(show_webcam, image_3d):
        if not show_webcam and image_3d is not None:
            gr.Image(image_3d, label="3D image")
        else:
            pass

    sort_option = gr.Dropdown(value="color", choices=["color", "size"], label="Sort by:",
                              info="Please select the sorting criteria", show_label=True)

    def start_sorting(option: str):
        registered_bricks, bricks, image = robot_interface.get_3d_bricks_and_image()
        yield False, image, gr.Button(
            "Stop sorting", variant="stop", size="lg")
        robot_interface.sort_bricks(
            registered_bricks, bricks, by_color=option == "color")

    sort_button = gr.Button(
        "Start sorting", variant="primary", size="lg")
    sort_button.click(fn=start_sorting, inputs=sort_option,
                      outputs=[show_webcam, image_3d, sort_button])
