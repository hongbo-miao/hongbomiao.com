import logging
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import lancedb
import lancedb.embeddings.imagebind
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table

logger = logging.getLogger(__name__)
EMBEDDINGS = get_registry().get("imagebind").create()
DATA_DIR = Path("data")

TEXT_LIST = ["A bird", "A dragon", "A car", "A guitar", "A witch", "Thunder"]
IMAGE_URLS = [
    "https://github.com/raghavdixit99/assets/assets/34462078/abf47cc4-d979-4aaa-83be-53a2115bf318",
    "https://github.com/raghavdixit99/assets/assets/34462078/93be928e-522b-4e37-889d-d4efd54b2112",
    "https://github.com/raghavdixit99/assets/assets/34462078/025deaff-632a-4829-a86c-3de6e326402f",
    "https://github.com/raghavdixit99/assets/assets/34462078/a20bff32-155c-4bad-acf1-97856c493099",
    "https://github.com/raghavdixit99/assets/assets/34462078/4f7dadd8-b38c-4c14-ac8a-5a2e74414f6a",
    "https://github.com/raghavdixit99/assets/assets/34462078/ac11eeab-7b2b-4db3-981b-d5fed08d9bc2",
]
AUDIO_URLS = [
    "https://github.com/raghavdixit99/assets/raw/main/bird_audio.wav",
    "https://github.com/raghavdixit99/assets/raw/main/dragon-growl-37570.wav",
    "https://github.com/raghavdixit99/assets/raw/main/car_audio.wav",
    "https://github.com/raghavdixit99/assets/raw/main/acoustic-guitar.wav",
    "https://github.com/raghavdixit99/assets/raw/main/witch.wav",
    "https://github.com/raghavdixit99/assets/raw/main/thunder-25689.wav",
]


class MultimodalSearchSchema(LanceModel):
    text: str
    image_path: str = EMBEDDINGS.SourceField()
    audio_path: str
    vector: Vector(EMBEDDINGS.ndims()) = EMBEDDINGS.VectorField()  # type: ignore[valid-type]


class ImageBindSearch:
    def __init__(self):
        self.table: Table | None = None

    @staticmethod
    def download_file(client: httpx.Client, url: str, is_audio: bool = True) -> Path:
        filename = url.split("/")[-1]
        if not is_audio:
            filename = f"{filename}.jpg"
        local_file_path = DATA_DIR / filename

        response = client.get(url)
        if response.status_code == 200:
            with open(local_file_path, "wb") as file:
                file.write(response.content)
                logger.info(f"Downloaded file: {local_file_path}")
            return local_file_path
        else:
            raise RuntimeError(f"Download failed: {response}")

    @staticmethod
    def download_all_files() -> tuple[list[Path], list[Path]]:
        with httpx.Client(follow_redirects=True) as client:
            audio_paths = [
                ImageBindSearch.download_file(client, url, True) for url in AUDIO_URLS
            ]
            image_paths = [
                ImageBindSearch.download_file(client, url, False) for url in IMAGE_URLS
            ]
            return audio_paths, image_paths

    def initialize_database(
        self,
        audio_paths: list[Path],
        image_paths: list[Path],
    ) -> None:
        inputs = [
            {"text": a, "audio_path": str(b), "image_path": str(c)}
            for a, b, c in zip(TEXT_LIST, audio_paths, image_paths)
        ]
        db = lancedb.connect("data/lancedb")
        self.table = db.create_table(
            "imagebind",
            schema=MultimodalSearchSchema,
            mode="overwrite",
        )
        self.table.add(inputs)

    def search_by_image(self, input_image: Any) -> tuple[str, Path]:
        if self.table is None:
            raise RuntimeError(
                "Database not initialized. Call initialize_database first.",
            )
        result = (
            self.table.search(input_image, vector_column_name="vector")
            .limit(1)
            .to_pydantic(MultimodalSearchSchema)[0]
        )
        return result.text, Path(result.audio_path)

    def search_by_text(self, input_text: str) -> tuple[Path, Path]:
        if self.table is None:
            raise RuntimeError(
                "Database not initialized. Call initialize_database first.",
            )
        result = (
            self.table.search(input_text, vector_column_name="vector")
            .limit(1)
            .to_pydantic(MultimodalSearchSchema)[0]
        )
        return Path(result.image_path), Path(result.audio_path)

    def search_by_audio(self, input_audio: Any) -> tuple[Path, str]:
        if self.table is None:
            raise RuntimeError(
                "Database not initialized. Call initialize_database first.",
            )
        result = (
            self.table.search(input_audio, vector_column_name="vector")
            .limit(1)
            .to_pydantic(MultimodalSearchSchema)[0]
        )
        return Path(result.image_path), result.text

    def create_gradio_interface(
        self,
        audio_paths: list[Path],
        image_paths: list[Path],
    ) -> gr.TabbedInterface:
        image_to_text_audio = gr.Interface(
            fn=self.search_by_image,
            inputs=gr.Image(type="filepath", value=image_paths[0]),
            outputs=[gr.Text(label="Output Text"), gr.Audio(label="Output Audio")],
            examples=image_paths,
            flagging_mode="never",
        )
        text_to_image_audio = gr.Interface(
            fn=self.search_by_text,
            inputs=gr.Textbox(label="Enter a prompt:"),
            outputs=[gr.Image(label="Output Image"), gr.Audio(label="Output Audio")],
            flagging_mode="never",
            examples=TEXT_LIST,
        )
        audio_to_image_text = gr.Interface(
            fn=self.search_by_audio,
            inputs=gr.Audio(type="filepath", value=audio_paths[0]),
            outputs=[gr.Image(label="Output Image"), gr.Text(label="Output Text")],
            examples=audio_paths,
            flagging_mode="never",
        )
        gradio_interface = gr.TabbedInterface(
            [image_to_text_audio, text_to_image_audio, audio_to_image_text],
            ["Image to Text/Audio", "Text to Image/Audio", "Audio to Image/Text"],
        )
        return gradio_interface


def main() -> None:
    # Download files
    audio_paths, image_paths = ImageBindSearch.download_all_files()

    # Initialize search engine
    search_engine = ImageBindSearch()
    search_engine.initialize_database(audio_paths, image_paths)

    # Create and launch interface
    gradio_interface = search_engine.create_gradio_interface(audio_paths, image_paths)
    gradio_interface.launch(share=False, allowed_paths=[str(DATA_DIR)])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
