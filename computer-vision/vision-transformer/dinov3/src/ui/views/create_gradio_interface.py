import gradio as gr
from ui.events.generate_similarity_images_from_image1_click import (
    generate_similarity_images_from_image1_click,
)


def create_gradio_interface() -> gr.Blocks:
    default_image1_url = "https://i.postimg.cc/gk84yJxy/507449262-d391060e-2ebe-42a0-befe-8bdbcd9eb5ef.png"
    default_image2_url = (
        "https://i.postimg.cc/TY15GH6n/istockphoto-1360719282-2048x2048.jpg"
    )

    with gr.Blocks(title="DINOv3 Similarity Visualization") as demo:
        gr.Markdown("# DINOv3 Interactive Similarity Visualization")
        gr.Markdown(
            "Upload two images and click on the first image to see similarity heatmaps.",
        )

        with gr.Row():
            with gr.Column():
                input_image1 = gr.Image(
                    label="Image 1 (Click to select a point)",
                    type="pil",
                    value=default_image1_url,
                )
            with gr.Column():
                input_image2 = gr.Image(
                    label="Image 2",
                    type="pil",
                    value=default_image2_url,
                )

        with gr.Row():
            with gr.Column():
                output_image1 = gr.Image(label="Image 1: Self-similarity")
            with gr.Column():
                output_image2 = gr.Image(label="Image 2: Cross-similarity")

        input_image1.select(
            fn=generate_similarity_images_from_image1_click,
            inputs=[input_image1, input_image2],
            outputs=[output_image1, output_image2],
        )

    return demo
