from PIL import Image, ImageDraw


def create_altitude_image(
    simulated_altitude: float,
    image_size: tuple[int, int] = (256, 256),
) -> Image.Image:
    sky_blue = int(200 + simulated_altitude * 5)
    sky_blue = min(255, sky_blue)
    ground_height = int(image_size[1] * (1 - simulated_altitude / 15.0))
    ground_height = max(50, min(image_size[1] - 50, ground_height))

    image = Image.new("RGB", image_size, color=(135, 206, sky_blue))

    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [0, ground_height, image_size[0], image_size[1]],
        fill=(34, 139, 34),
    )

    return image
