import torch
from torchvision import models


def main() -> None:
    model_name = "model.pt"

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    resnet50 = models.resnet50(pretrained=True)
    resnet50 = resnet50.eval()
    resnet50.to(device)

    resnet50_jit = torch.jit.script(resnet50)
    resnet50_jit.save(model_name)


if __name__ == "__main__":
    main()
