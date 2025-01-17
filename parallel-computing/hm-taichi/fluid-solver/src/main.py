# https://github.com/hietwll/LBM_Taichi

import logging
import sys

import matplotlib as mpl
import numpy as np
import taichi as ti
from matplotlib import cm

logger = logging.getLogger(__name__)
ti.init(arch=ti.gpu)


@ti.data_oriented
class LatticeBoltzmannSolver:
    """
    A GPU-accelerated Lattice Boltzmann Method (LBM) solver for 2D fluid dynamics.

    This solver implements the D2Q9 lattice model to simulate fluid flows in two dimensions.
    It uses the Taichi programming language for GPU acceleration, enabling efficient
    parallel computation of fluid dynamics.

    The D2Q9 model uses 9 discrete velocities at each lattice point:

    * 1 rest velocity (center)
    * 4 primary directions (north, south, east, west)
    * 4 diagonal directions (northeast, northwest, southeast, southwest)

    :param case_name: Identifier for the simulation case (e.g., "Karman Vortex Street")
    :param grid_size_x: Number of grid points along the x-axis
    :param grid_size_y: Number of grid points along the y-axis
    :param kinematic_viscosity: Fluid's kinematic viscosity, determines the Reynolds number
    :param boundary_types: Boundary conditions for [left, top, right, bottom] walls.
                         0: Dirichlet boundary (fixed velocity),
                         1: Neumann boundary (zero gradient)
    :param boundary_values: Velocity values for Dirichlet boundaries
    :param has_cylinder: Flag to include a cylindrical obstacle (0: no, 1: yes), defaults to 0
    :param cylinder_params: Cylinder parameters [center_x, center_y, radius], defaults to [0.0, 0.0, 0.0]
    """

    def __init__(
        self,
        case_name: str,
        grid_size_x: int,
        grid_size_y: int,
        kinematic_viscosity: float,
        boundary_types: list[int],
        boundary_values: list[list[float]],
        has_cylinder: int = 0,
        cylinder_params: list[float] | None = None,
    ) -> None:
        if cylinder_params is None:
            cylinder_params = [0.0, 0.0, 0.0]
        self.case_name = case_name

        # Set up the computational grid
        # Note: We use lattice units where dx = dy = dt = 1.0
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y

        # Initialize fluid properties
        self.kinematic_viscosity = kinematic_viscosity
        self.relaxation_time = (
            3.0 * kinematic_viscosity + 0.5
        )  # Relaxation time for collision operator
        self.inverse_relaxation_time = 1.0 / self.relaxation_time

        # Initialize field variables for simulation
        self.density = ti.field(float, shape=(grid_size_x, grid_size_y))
        self.velocity = ti.Vector.field(2, float, shape=(grid_size_x, grid_size_y))
        self.obstacle_mask = ti.field(float, shape=(grid_size_x, grid_size_y))
        self.distribution_curr = ti.Vector.field(
            9,
            float,
            shape=(grid_size_x, grid_size_y),
        )
        self.distribution_next = ti.Vector.field(
            9,
            float,
            shape=(grid_size_x, grid_size_y),
        )

        # Define D2Q9 lattice weights for equilibrium distribution
        self.weights = (
            ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        )

        # Define D2Q9 lattice velocities
        # Each row represents a discrete velocity vector:
        # [0]: Rest particle (0, 0)
        # [1-4]: Primary directions
        # [5-8]: Diagonal directions
        self.lattice_velocities = ti.types.matrix(9, 2, int)(
            [0, 0],  # Center (rest)
            [1, 0],  # Right (east)
            [0, 1],  # Up (north)
            [-1, 0],  # Left (west)
            [0, -1],  # Down (south)
            [1, 1],  # Up-Right (northeast)
            [-1, 1],  # Up-Left (northwest)
            [-1, -1],  # Down-Left (southwest)
            [1, -1],  # Down-Right (southeast)
        )

        # Set up boundary conditions
        self.boundary_types = ti.field(int, 4)
        self.boundary_types.from_numpy(np.array(boundary_types, dtype=np.int32))
        self.boundary_values = ti.Vector.field(2, float, shape=4)
        self.boundary_values.from_numpy(np.array(boundary_values, dtype=np.float32))

        # Configure cylindrical obstacle if present
        self.has_cylinder = has_cylinder
        self.cylinder_params = ti.math.vec3(cylinder_params)

    @ti.func
    def calculate_equilibrium_distribution(self, i: int, j: int):  # noqa: ANN201
        """
        Calculate equilibrium distribution function at grid point (i, j).

        This function computes the equilibrium state of the particle distribution
        based on local density and velocity, following the Maxwell-Boltzmann distribution.

        :param i: Grid point x-coordinate
        :param j: Grid point y-coordinate
        :return: Equilibrium distribution for all directions at point (i, j)
        """
        velocity_projection = self.lattice_velocities @ self.velocity[i, j]
        velocity_magnitude_squared = ti.math.dot(
            self.velocity[i, j],
            self.velocity[i, j],
        )
        return (
            self.weights
            * self.density[i, j]
            * (
                1
                + 3 * velocity_projection
                + 4.5 * velocity_projection * velocity_projection
                - 1.5 * velocity_magnitude_squared
            )
        )

    @ti.kernel
    def initialize_simulation_domain(self):  # noqa: ANN201
        """
        Initialize the simulation domain.

        Sets up initial conditions:

        * Zero velocity everywhere
        * Unit density
        * Equilibrium distribution
        * Cylindrical obstacle (if enabled)
        """
        self.velocity.fill(0)
        self.density.fill(1)
        self.obstacle_mask.fill(0)
        for i, j in self.density:
            self.distribution_curr[i, j] = self.distribution_next[i, j] = (
                self.calculate_equilibrium_distribution(i, j)
            )
            if (
                self.has_cylinder == 1
                and (i - self.cylinder_params[0]) ** 2
                + (j - self.cylinder_params[1]) ** 2
                <= self.cylinder_params[2] ** 2
            ):
                self.obstacle_mask[i, j] = 1.0

    @ti.kernel
    def perform_collision_and_streaming(self):  # noqa: ANN201
        """
        Perform collision and streaming steps of the LBM algorithm.

        This is the core of the LBM simulation, consisting of two steps:

        1. Collision: Particles at each lattice point collide and redistribute
        2. Streaming: Particles move to neighboring lattice points
        """
        for i, j in ti.ndrange((1, self.grid_size_x - 1), (1, self.grid_size_y - 1)):
            for k in ti.static(range(9)):
                # Find the source point for streaming
                source_x = i - self.lattice_velocities[k, 0]
                source_y = j - self.lattice_velocities[k, 1]

                # Calculate equilibrium distribution
                equilibrium_dist = self.calculate_equilibrium_distribution(
                    source_x,
                    source_y,
                )

                # Perform collision and streaming in one step
                self.distribution_next[i, j][k] = (
                    1 - self.inverse_relaxation_time
                ) * self.distribution_curr[source_x, source_y][k] + equilibrium_dist[
                    k
                ] * self.inverse_relaxation_time

    @ti.kernel
    def update_macroscopic_variables(self):  # noqa: ANN201
        """
        Update macroscopic variables (density and velocity) from distribution functions.

        After streaming, we need to:

        1. Calculate new density and velocity at each point
        2. Update distribution functions for the next iteration
        """
        for i, j in ti.ndrange((1, self.grid_size_x - 1), (1, self.grid_size_y - 1)):
            self.density[i, j] = 0
            self.velocity[i, j] = 0, 0

            # Sum up contributions from all directions
            for k in ti.static(range(9)):
                # Copy new distribution to current for next iteration
                self.distribution_curr[i, j][k] = self.distribution_next[i, j][k]

                # Calculate macroscopic variables
                self.density[i, j] += self.distribution_next[i, j][k]
                self.velocity[i, j] += (
                    ti.math.vec2(
                        self.lattice_velocities[k, 0],
                        self.lattice_velocities[k, 1],
                    )
                    * self.distribution_next[i, j][k]
                )

            # Normalize velocity by density
            self.velocity[i, j] /= self.density[i, j]

    @ti.kernel
    def apply_boundary_conditions(self):  # noqa: ANN201
        """
        Apply boundary conditions to the simulation domain.

        Handles three types of boundaries:

        1. Left and right walls
        2. Top and bottom walls
        3. Cylindrical obstacle (if present)
        """
        # Apply boundary conditions to left and right walls
        for j in range(1, self.grid_size_y - 1):
            # Left wall: Apply specified boundary condition
            self.apply_boundary_condition_at_node(1, 0, 0, j, 1, j)

            # Right wall: Apply specified boundary condition
            self.apply_boundary_condition_at_node(
                1,
                2,
                self.grid_size_x - 1,
                j,
                self.grid_size_x - 2,
                j,
            )

        # Apply boundary conditions to top and bottom walls
        for i in range(self.grid_size_x):
            # Top wall: Apply specified boundary condition
            self.apply_boundary_condition_at_node(
                1,
                1,
                i,
                self.grid_size_y - 1,
                i,
                self.grid_size_y - 2,
            )

            # Bottom wall: Apply specified boundary condition
            self.apply_boundary_condition_at_node(1, 3, i, 0, i, 1)

        # Handle cylindrical obstacle boundary
        for i, j in ti.ndrange(self.grid_size_x, self.grid_size_y):
            if self.has_cylinder == 1 and self.obstacle_mask[i, j] == 1:
                # Enforce no-slip condition on cylinder surface
                self.velocity[i, j] = 0, 0

                # Find nearest fluid node for boundary treatment
                neighbor_x = 0
                neighbor_y = 0
                neighbor_x = i + 1 if i >= self.cylinder_params[0] else i - 1
                neighbor_y = j + 1 if j >= self.cylinder_params[1] else j - 1

                # Apply bounce-back boundary condition
                self.apply_boundary_condition_at_node(
                    0,
                    0,
                    i,
                    j,
                    neighbor_x,
                    neighbor_y,
                )

    @ti.func
    def apply_boundary_condition_at_node(
        self,
        is_outer_boundary: int,
        direction: int,
        boundary_x: int,
        boundary_y: int,
        neighbor_x: int,
        neighbor_y: int,
    ) -> None:
        """
        Apply boundary condition at a specific boundary node.

        :param is_outer_boundary: Flag for outer boundary (1) or internal boundary (0)
        :param direction: Direction index (0: left, 1: top, 2: right, 3: bottom)
        :param boundary_x: X-coordinate of boundary node
        :param boundary_y: Y-coordinate of boundary node
        :param neighbor_x: X-coordinate of neighboring fluid node
        :param neighbor_y: Y-coordinate of neighboring fluid node
        """
        if is_outer_boundary == 1:
            if self.boundary_types[direction] == 0:
                # Dirichlet boundary: Set specified velocity
                self.velocity[boundary_x, boundary_y] = self.boundary_values[direction]
            elif self.boundary_types[direction] == 1:
                # Neumann boundary: Copy velocity from neighbor
                self.velocity[boundary_x, boundary_y] = self.velocity[
                    neighbor_x,
                    neighbor_y,
                ]

        # Update density and distribution function at boundary
        self.density[boundary_x, boundary_y] = self.density[neighbor_x, neighbor_y]
        self.distribution_curr[boundary_x, boundary_y] = (
            self.calculate_equilibrium_distribution(boundary_x, boundary_y)
            - self.calculate_equilibrium_distribution(neighbor_x, neighbor_y)
            + self.distribution_curr[neighbor_x, neighbor_y]
        )

    def run_simulation(self) -> None:
        """
        Run the main simulation loop.

        Creates a GUI window and iteratively:

        1. Performs multiple LBM steps per frame
        2. Visualizes the flow field using vorticity
        3. Continues until user closes the window
        """
        gui = ti.GUI(self.case_name, (self.grid_size_x, 2 * self.grid_size_y))
        self.initialize_simulation_domain()
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            for _ in range(10):
                self.perform_collision_and_streaming()
                self.update_macroscopic_variables()
                self.apply_boundary_conditions()

            # Visualization Section
            # Calculate vorticity field from velocity gradients
            vel = self.velocity.to_numpy()
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5

            # Define custom colormap for visualization
            # Yellow -> Orange -> Black -> Green -> Cyan
            # Used for visualizing vorticity (rotation) in the fluid
            colors = [
                (1, 1, 0),  # Yellow
                (0.953, 0.490, 0.016),  # Orange
                (0, 0, 0),  # Black
                (0.176, 0.976, 0.529),  # Green
                (0, 1, 1),  # Cyan
            ]
            my_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "my_cmap",
                colors,
            )
            vor_img = cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=-0.02, vmax=0.02),
                cmap=my_cmap,
            ).to_rgba(vor)
            vel_img = cm.plasma(vel_mag / 0.15)
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    flow_case = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    if flow_case == 0:  # von Karman vortex street: Re = U*D/niu = 200
        lbm = LatticeBoltzmannSolver(
            "Karman Vortex Street",
            801,
            201,
            0.01,
            [0, 0, 1, 0],
            [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            1,
            [160.0, 100.0, 20.0],
        )
        lbm.run_simulation()
    elif flow_case == 1:  # lid-driven cavity flow: Re = U*L/niu = 1000
        lbm = LatticeBoltzmannSolver(
            "Lid-driven Cavity Flow",
            256,
            256,
            0.0255,
            [0, 0, 0, 0],
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )
        lbm.run_simulation()
    else:
        logger.error(
            "Invalid flow case ! Please choose from 0 (Karman Vortex Street) and 1 (Lid-driven Cavity Flow).",
        )
