import math


def compute_item(i: int) -> float:
    float_i = float(i)
    value = float_i**2
    for _ in range(1000):
        value = abs(math.sin(value) * math.cos(value)) + math.sqrt(float_i)
    return value
