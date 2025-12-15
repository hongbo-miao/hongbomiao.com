import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn, optim
from torchdiffeq import odeint, odeint_adjoint

logger = logging.getLogger(__name__)

METHOD = "dopri5"
DATA_SIZE = 1000
BATCH_TIME = 10
BATCH_SIZE = 20
TOTAL_ITERATIONS = 100
TEST_FREQUENCY = 20
USE_ADJOINT = False
RNG = np.random.default_rng(seed=42)


class LambdaDynamics(nn.Module):
    def __init__(self, system_matrix: Tensor) -> None:
        super().__init__()
        self.register_buffer("system_matrix", system_matrix)

    def forward(self, time_value: Tensor, state_value: Tensor) -> Tensor:
        del time_value
        return torch.mm(state_value**3, self.system_matrix)


class ODEFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )
        self._initialize_parameters()

    def forward(self, time_value: Tensor, state_value: Tensor) -> Tensor:
        del time_value
        return self.network(state_value**3)

    def _initialize_parameters(self) -> None:
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.constant_(module.bias, val=0.0)


def select_ode_solver(is_adjoint: bool) -> Callable[..., Tensor]:
    return odeint_adjoint if is_adjoint else odeint


def compute_true_trajectory(
    solver: Callable[..., Tensor],
    system_matrix: Tensor,
    initial_state: Tensor,
    time_points: Tensor,
    method: str,
    solver_options: dict[str, torch.dtype],
) -> Tensor:
    true_dynamics = LambdaDynamics(system_matrix)
    with torch.no_grad():
        return solver(
            true_dynamics,
            initial_state,
            time_points,
            method=method,
            options=solver_options,
        )


def get_training_batch(
    trajectory: Tensor,
    time_points: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    valid_range = DATA_SIZE - BATCH_TIME
    if valid_range <= 0:
        message = "DATA_SIZE must be greater than BATCH_TIME."
        raise ValueError(message)
    indices = RNG.choice(
        np.arange(valid_range, dtype=np.int64),
        size=BATCH_SIZE,
        replace=False,
    )
    start_positions = torch.from_numpy(indices).to(trajectory.device)
    batch_initial_state = trajectory[start_positions]
    batch_time = time_points[:BATCH_TIME]
    batch_states = torch.stack(
        [trajectory[start_positions + offset] for offset in range(BATCH_TIME)],
        dim=0,
    )
    return batch_initial_state, batch_time, batch_states


def save_prediction_plot(
    time_points: Tensor,
    true_trajectory: Tensor,
    predicted_trajectory: Tensor,
    learned_function: nn.Module,
    iteration_index: int,
    output_directory: Path,
) -> None:
    time_np = time_points.cpu().numpy()
    true_np = true_trajectory.cpu().numpy()[:, 0]
    predicted_np = predicted_trajectory.cpu().numpy()[:, 0]

    figure = plt.figure(figsize=(12, 4), facecolor="white")
    trajectory_axis = figure.add_subplot(131, frameon=False)
    phase_axis = figure.add_subplot(132, frameon=False)
    vector_field_axis = figure.add_subplot(133, frameon=False)

    trajectory_axis.set_title("Trajectories")
    trajectory_axis.set_xlabel("t")
    trajectory_axis.set_ylabel("x, y")
    trajectory_axis.plot(time_np, true_np[:, 0], "g-", label="True x")
    trajectory_axis.plot(time_np, true_np[:, 1], "g--", label="True y")
    trajectory_axis.plot(time_np, predicted_np[:, 0], "b-", label="Pred x")
    trajectory_axis.plot(time_np, predicted_np[:, 1], "b--", label="Pred y")
    trajectory_axis.set_xlim(time_np.min(), time_np.max())
    trajectory_axis.set_ylim(-2, 2)
    trajectory_axis.legend()

    phase_axis.set_title("Phase Portrait")
    phase_axis.set_xlabel("x")
    phase_axis.set_ylabel("y")
    phase_axis.plot(true_np[:, 0], true_np[:, 1], "g-")
    phase_axis.plot(predicted_np[:, 0], predicted_np[:, 1], "b--")
    phase_axis.set_xlim(-2, 2)
    phase_axis.set_ylim(-2, 2)

    vector_field_axis.set_title("Learned Vector Field")
    vector_field_axis.set_xlabel("x")
    vector_field_axis.set_ylabel("y")
    y_grid, x_grid = np.mgrid[-2:2:21j, -2:2:21j]  # type: ignore[misc]
    grid_tensor = torch.tensor(
        np.stack([x_grid, y_grid], -1).reshape(21 * 21, 2),
        dtype=true_trajectory.dtype,
        device=true_trajectory.device,
    )
    with torch.no_grad():
        field = (
            learned_function(torch.tensor(0.0, device=grid_tensor.device), grid_tensor)
            .cpu()
            .numpy()
        )
    magnitude = np.linalg.norm(field, axis=1, keepdims=True)
    magnitude = np.where(magnitude == 0.0, 1.0, magnitude)
    normalized_field = (field / magnitude).reshape(21, 21, 2)
    vector_field_axis.streamplot(
        x_grid,
        y_grid,
        normalized_field[:, :, 0],
        normalized_field[:, :, 1],
        color="black",
    )
    vector_field_axis.set_xlim(-2, 2)
    vector_field_axis.set_ylim(-2, 2)

    figure.tight_layout()
    plt.savefig(output_directory / f"{iteration_index:03d}.png")
    plt.close(figure)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ode_solver = select_ode_solver(USE_ADJOINT)
    device = resolve_device()

    dtype = torch.float32
    true_initial_state = torch.tensor([[2.0, 0.0]], dtype=dtype, device=device)
    time_points = torch.linspace(
        0.0,
        25.0,
        steps=DATA_SIZE,
        dtype=dtype,
        device=device,
    )
    system_matrix = torch.tensor(
        [
            [-0.1, 2.0],
            [-2.0, -0.1],
        ],
        dtype=dtype,
        device=device,
    )

    solver_options = {"dtype": true_initial_state.dtype}
    true_trajectory = compute_true_trajectory(
        solver=ode_solver,
        system_matrix=system_matrix,
        initial_state=true_initial_state,
        time_points=time_points,
        method=METHOD,
        solver_options=solver_options,
    )

    output_directory = Path("output")
    output_directory.mkdir(parents=True, exist_ok=True)

    learned_function = ODEFunction().to(device)
    optimizer = optim.RMSprop(learned_function.parameters(), lr=1e-3)
    snapshot_index = 0

    for iteration in range(1, TOTAL_ITERATIONS + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_training_batch(true_trajectory, time_points)
        predicted_batch = ode_solver(
            learned_function,
            batch_y0,
            batch_t,
            method=METHOD,
            options={"dtype": batch_y0.dtype},
        )
        loss = torch.mean(torch.abs(predicted_batch - batch_y))
        loss.backward()
        optimizer.step()

        if iteration % TEST_FREQUENCY == 0:
            with torch.no_grad():
                predicted_trajectory = ode_solver(
                    learned_function,
                    true_initial_state,
                    time_points,
                    method=METHOD,
                    options=solver_options,
                )
                evaluation_loss = torch.mean(
                    torch.abs(predicted_trajectory - true_trajectory),
                )
                logger.info(
                    "Iter %04d | Total Loss %.6f",
                    iteration,
                    evaluation_loss.item(),
                )
                save_prediction_plot(
                    time_points=time_points,
                    true_trajectory=true_trajectory,
                    predicted_trajectory=predicted_trajectory,
                    learned_function=learned_function,
                    iteration_index=snapshot_index,
                    output_directory=output_directory,
                )
                snapshot_index += 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
