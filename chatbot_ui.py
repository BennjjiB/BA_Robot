import gradio as gr
from gradio import ChatMessage
from client import Client

def chatbot_ui(client: Client):
    def interact_with_pandabot(prompt, messages):
        messages = messages if messages else []
        messages.append(ChatMessage(role="user", content=prompt))
        yield "", messages
        response = client.send_prompt(prompt)
        messages.append(ChatMessage(role="assistant", content=""))
        for chunk in client.handle_response(response):
            if chunk.get("text"):
                messages[-1] = ChatMessage(role="assistant", content=chunk.get("text"))
                yield "", messages
            elif chunk.get("add"):
                messages.append(ChatMessage(role="assistant", content=""))
            elif chunk.get("tool"):
                for tool in chunk["tool"]:
                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=f"{tool}",
                            metadata={"title": f"🛠️ Used tool {tool.get('function_name', '')}"}
                        )
                    )
                    yield "", messages


    def capture_audio(new_chunk, messages):
        if new_chunk:
            new_transcript, start_prompt = client.send_audio(new_chunk[1], new_chunk[0])
            if start_prompt:
                yield from interact_with_pandabot(new_transcript, messages)
            elif new_transcript:
                yield new_transcript, gr.skip()
        yield gr.skip(), gr.skip()

    def clear_all():
        client.clear_history()
        return "", []

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
        inputs=[input_audio, chatbot],
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
