import logging
import random
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

class_colors = {}


def generate_random_color() -> tuple[int, int, int]:
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b


def main(model_path: Path, image_path: Path) -> None:
    model = YOLO(model_path)
    image = cv2.imread(str(image_path))
    res = model(image)[0]
    detections = sv.Detections.from_ultralytics(res)
    detections = detections[detections.confidence > 0.3]
    class_names = model.names

    for detection in detections:
        print(detection)
        xyxy, _, _, class_id, _, _ = detection

        x0, y0, x1, y1 = xyxy
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)

        label = class_names[class_id]

        if class_id not in class_colors:
            class_colors[class_id] = generate_random_color()

        color = class_colors[class_id]

        cv2.rectangle(image, (x0, y0), (x1, y1), color, 5)
        cv2.putText(image, label, (x0, y0 - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

    cv2.imshow("Detections", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir_path = Path("data")
    external_model_path = data_dir_path / Path("yolov8x.pt")
    external_image_path = data_dir_path / Path("image.jpg")
    main(external_model_path, external_image_path)
