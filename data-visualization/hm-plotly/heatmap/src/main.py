from pathlib import Path

import plotly.express as px


def main() -> None:
    z = [
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [1, 0.8, 0.6, 0.4, 0.2],
        [0.2, 0, 0.5, 0.7, 0.9],
        [0.9, 0.8, 0.4, 0.2, 0],
        [0.3, 0.4, 0.5, 0.7, 1],
    ]
    fig = px.imshow(z, text_auto=True, aspect="auto")
    fig.show()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "heatmap.html"
    fig.write_html(str(output_path))


if __name__ == "__main__":
    main()
