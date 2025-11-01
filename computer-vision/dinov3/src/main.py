import logging

from ui.views.create_gradio_interface import create_gradio_interface


def main() -> None:
    demo = create_gradio_interface()
    demo.launch(share=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
