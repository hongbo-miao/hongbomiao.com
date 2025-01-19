import torch
from torchvision import models


def main() -> None:
    model_name = "model.onnx"
    resnet50 = models.resnet50(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    resnet50 = resnet50.eval()

    torch.onnx.export(
        resnet50,
        dummy_input,
        model_name,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


if __name__ == "__main__":
    main()
