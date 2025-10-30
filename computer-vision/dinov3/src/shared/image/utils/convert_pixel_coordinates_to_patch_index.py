def convert_pixel_coordinates_to_patch_index(
    x: int,
    y: int,
    patch_size: int,
    column_count: int,
) -> int:
    column = int(x // patch_size)
    row = int(y // patch_size)
    return row * column_count + column
