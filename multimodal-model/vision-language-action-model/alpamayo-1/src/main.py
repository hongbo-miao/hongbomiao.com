import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[AlpamayoR1, helper.AutoProcessor]:
    logger.info(f"Loading model: {model_name}")
    model = AlpamayoR1.from_pretrained(model_name, dtype=dtype).to(device)
    processor = helper.get_processor(model.tokenizer)
    logger.info("Model loaded successfully")
    return model, processor


def prepare_model_inputs(
    clip_id: str,
    processor: helper.AutoProcessor,
    device: torch.device,
) -> tuple[dict, dict]:
    logger.info(f"Loading data for clip: {clip_id}")
    data = load_physical_aiavdataset(clip_id)

    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    logger.info(f"Sequence length: {inputs.input_ids.shape}")

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)

    return model_inputs, data


def run_inference(
    model: AlpamayoR1,
    model_inputs: dict,
    dtype: torch.dtype,
    top_p: float,
    temperature: float,
    trajectory_sample_count: int,
    max_generation_length: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    logger.info("Running model inference")
    torch.cuda.manual_seed_all(42)

    with torch.autocast("cuda", dtype=dtype):
        predicted_xyz, predicted_rotation, extra = (
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=copy.deepcopy(model_inputs),
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=trajectory_sample_count,
                max_generation_length=max_generation_length,
                return_extra=True,
            )
        )

    logger.info(f"Chain-of-Causation (per trajectory): {extra['cot'][0]}")
    return predicted_xyz, predicted_rotation, extra


def rotate_90_counter_clockwise(xy: np.ndarray) -> np.ndarray:
    return np.stack([-xy[1], xy[0]], axis=0)


def visualize_trajectories(
    predicted_xyz: torch.Tensor,
    data: dict,
    output_path: Path,
) -> None:
    logger.info("Visualizing trajectories")

    for i in range(predicted_xyz.shape[2]):
        predicted_xy = predicted_xyz.cpu()[0, 0, i, :, :2].T.numpy()
        predicted_xy_rotated = rotate_90_counter_clockwise(predicted_xy)
        plt.plot(*predicted_xy_rotated, "o-", label=f"Predicted Trajectory #{i + 1}")

    ground_truth_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    ground_truth_xy_rotated = rotate_90_counter_clockwise(ground_truth_xy)
    plt.plot(*ground_truth_xy_rotated, "r-", label="Ground Truth Trajectory")

    plt.ylabel("y coordinate (meters)")
    plt.xlabel("x coordinate (meters)")
    plt.legend(loc="best")
    plt.axis("equal")
    plt.savefig(output_path)
    logger.info(f"Saved trajectory visualization to {output_path}")


def calculate_min_ade(predicted_xyz: torch.Tensor, data: dict) -> float:
    predicted_xy = predicted_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    ground_truth_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    difference = np.linalg.norm(predicted_xy - ground_truth_xy[None, ...], axis=1).mean(
        -1,
    )
    return float(difference.min())


def main() -> None:
    model_name = "nvidia/Alpamayo-R1-10B"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    trajectory_output_path = Path("output/trajectory.png")
    clip_ids_parquet_url = "https://raw.githubusercontent.com/NVlabs/alpamayo/main/notebooks/clip_ids.parquet"
    clip_id_index = 774

    model, processor = load_model(model_name, device, dtype)

    clip_ids = pl.read_parquet(clip_ids_parquet_url)["clip_id"].to_list()
    clip_id = clip_ids[clip_id_index]

    model_inputs, data = prepare_model_inputs(clip_id, processor, device)

    predicted_xyz, _, _ = run_inference(
        model=model,
        model_inputs=model_inputs,
        dtype=dtype,
        top_p=0.98,
        temperature=0.6,
        trajectory_sample_count=1,
        max_generation_length=256,
    )

    visualize_trajectories(predicted_xyz, data, output_path=trajectory_output_path)

    min_ade = calculate_min_ade(predicted_xyz, data)
    logger.info(f"minADE: {min_ade} meters")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()
