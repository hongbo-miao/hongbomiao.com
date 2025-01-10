import gradio as gr
import httpx
import torch
from torchvision import transforms


def main() -> None:
    model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True).eval()

    # Download human-readable labels for ImageNet
    res = httpx.get(
        "https://raw.githubusercontent.com/gradio-app/mobilenet-example/master/labels.txt",
    )
    labels = res.text.split("\n")

    def predict(image_tensor: torch.Tensor) -> dict[str, float]:
        image_tensor = transforms.ToTensor()(image_tensor).unsqueeze(0)
        with torch.no_grad():
            prediction = torch.nn.functional.softmax(model(image_tensor)[0], dim=0)
            confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
        return confidences

    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
    ).launch()


if __name__ == "__main__":
    main()
