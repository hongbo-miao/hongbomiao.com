def convert_pixel_coordinates_to_patch_index(
    x: int,
    y: int,
    patch_size: int,
    column_count: int,
) -> int:
    """
    Convert pixel coordinates (x, y) to flattened patch index.

    Math:
        Given pixel coordinate (x, y) and patch size P:
            col = floor(x / P)
            row = floor(y / P)
            index = row * N_col + col

    """
    # col = floor(x / P)
    column = int(x // patch_size)
    # row = floor(y / P)
    row = int(y // patch_size)
    # index = row * N_col + col
    return row * column_count + column
