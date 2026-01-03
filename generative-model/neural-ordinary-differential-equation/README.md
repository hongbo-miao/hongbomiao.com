# Neural Ordinary Differential Equation (Neural ODE)

Based on the paper [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) by Chen et al. (NeurIPS 2018).

## Core Idea

Neural ODEs replace discrete layer-by-layer transformations with continuous dynamics. Instead of stacking layers, we parameterize the derivative of the hidden state with a neural network and solve the resulting ODE. This allows the network to learn complex temporal dynamics while providing benefits like constant memory cost and adaptive computation.

## Intuition

| Traditional Neural Networks                                 | Neural ODEs                                              |
| ----------------------------------------------------------- | -------------------------------------------------------- |
| Discrete layers (layer 1, layer 2, ...)                     | Continuous depth                                         |
| Fixed number of transformations                             | Adaptive computation (solver chooses steps)              |
| Memory scales with depth                                    | Constant memory via adjoint method                       |
| ResNet: $\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t)$ | ODE: $\frac{d\mathbf{h}}{dt} = f(\mathbf{h}, t, \theta)$ |

The key insight: ResNet's skip connections are actually Euler discretizations of an ODE. Why not solve the ODE directly?

## The Math

### The True System (What We Want to Learn)

This experiment learns dynamics governed by a cubic nonlinear system:

```math
\frac{d\mathbf{x}}{dt} = \mathbf{x}^{\odot 3} \cdot \mathbf{A}
```

where:

- $\mathbf{x}(t) = [x_1(t), x_2(t)]$ is the 2D state vector
- $\mathbf{x}^{\odot 3}$ denotes elementwise cube (Hadamard power)
- $\mathbf{A}$ is the system matrix:

```math
\mathbf{A} = \begin{bmatrix} -0.1 & 2.0 \\ -2.0 & -0.1 \end{bmatrix}
```

| Matrix Component       | Effect                  |
| ---------------------- | ----------------------- |
| Diagonal $-0.1$        | Damping (spiral inward) |
| Off-diagonal $\pm 2.0$ | Rotation (oscillation)  |

### The Neural ODE Approximation

We approximate the unknown dynamics with a neural network:

```math
\frac{d\mathbf{x}}{dt} = f_\theta(\mathbf{x}^{\odot 3})
```

where $f_\theta$ is a simple MLP:

```math
f_\theta(\mathbf{z}) = \mathbf{W}_2 \cdot \tanh(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2
```

| Layer  | Dimensions | Purpose                      |
| ------ | ---------- | ---------------------------- |
| Input  | 2          | State vector (after cubing)  |
| Hidden | 50         | Nonlinear feature extraction |
| Output | 2          | Predicted derivative         |

### Solving the ODE

Given initial condition $\mathbf{x}(t_0)$, we solve:

```math
\mathbf{x}(t_1) = \mathbf{x}(t_0) + \int_{t_0}^{t_1} f_\theta(\mathbf{x}(t)^{\odot 3}) \, dt
```

The code uses `torchdiffeq.odeint` with the **Dormand-Prince (dopri5)** adaptive solver, which automatically chooses step sizes to maintain accuracy.

### Training with Backpropagation

Two approaches for computing gradients:

| Method   | Memory                     | Speed  | When to Use                       |
| -------- | -------------------------- | ------ | --------------------------------- |
| Standard | $O(L)$ - stores all states | Faster | Small problems                    |
| Adjoint  | $O(1)$ - constant          | Slower | Large problems, long trajectories |

The **adjoint method** solves a backward ODE instead of storing intermediate states:

```math
\frac{d\mathbf{a}}{dt} = -\mathbf{a}^T \frac{\partial f}{\partial \mathbf{x}}, \quad \mathbf{a}(t_1) = \frac{\partial L}{\partial \mathbf{x}(t_1)}
```

where $\mathbf{a}(t)$ is the adjoint state (gradient w.r.t. hidden state at time $t$).

## Implementation

### Code-to-Math Mapping

| Step            | Formula                                                         | Code                                           |
| --------------- | --------------------------------------------------------------- | ---------------------------------------------- |
| True dynamics   | $\frac{d\mathbf{x}}{dt} = \mathbf{x}^{\odot 3} \mathbf{A}$      | `torch.mm(state_value**3, self.system_matrix)` |
| Neural dynamics | $\frac{d\mathbf{x}}{dt} = f_\theta(\mathbf{x}^{\odot 3})$       | `self.network(state_value**3)`                 |
| Solve ODE       | $\mathbf{x}(T) = \text{ODESolve}(\mathbf{x}_0, f, t)$           | `odeint(func, y0, t, method="dopri5")`         |
| Loss            | $L = \frac{1}{N}\sum_i \|\hat{\mathbf{x}}_i - \mathbf{x}_i\|_1$ | `torch.mean(torch.abs(predicted - true))`      |

### Training Configuration

| Parameter          | Value        | Purpose                             |
| ------------------ | ------------ | ----------------------------------- |
| `DATA_SIZE`        | 1000         | Number of time points in trajectory |
| `BATCH_TIME`       | 10           | Trajectory segment length per batch |
| `BATCH_SIZE`       | 20           | Number of trajectory segments       |
| `TOTAL_ITERATIONS` | 100          | Training iterations                 |
| Time interval      | $[0, 25]$    | Integration range                   |
| Initial state      | $[2.0, 0.0]$ | Starting point                      |

### Training Process

```math
\begin{aligned}
\text{1. Sample batch:} \quad & \{(\mathbf{x}_{s_i}, t_{s_i:s_i+T})\}_{i=1}^B \\
\text{2. Forward pass:} \quad & \hat{\mathbf{x}} = \text{ODESolve}(\mathbf{x}_{s_i}, f_\theta, t_{s_i:s_i+T}) \\
\text{3. Compute loss:} \quad & L = \frac{1}{BT}\sum_{i,t} |\hat{\mathbf{x}}_{i,t} - \mathbf{x}_{i,t}| \\
\text{4. Update:} \quad & \theta \leftarrow \theta - \eta \nabla_\theta L
\end{aligned}
```

## Visualization Outputs

The code generates three plots at each checkpoint:

| Plot           | Shows                           | Interpretation                                        |
| -------------- | ------------------------------- | ----------------------------------------------------- |
| Trajectories   | $x(t), y(t)$ vs $t$             | How well predicted curves match true curves over time |
| Phase Portrait | $y$ vs $x$                      | The spiral attractor structure                        |
| Vector Field   | $\frac{d\mathbf{x}}{dt}$ arrows | Learned dynamics across state space                   |

## Comparison: Neural ODE vs Traditional Approaches

| Property          | Discrete NN             | Neural ODE                        |
| ----------------- | ----------------------- | --------------------------------- |
| Depth             | Fixed layers            | Continuous (solver-adaptive)      |
| Memory (training) | $O(\text{layers})$      | $O(1)$ with adjoint               |
| Time series       | Requires fixed sampling | Handles irregular times naturally |
| Invertibility     | Generally not           | Guaranteed (solve backward)       |
| Physics modeling  | Approximates            | Native continuous dynamics        |

## Why Cubic Nonlinearity?

The choice of $\mathbf{x}^{\odot 3}$ creates interesting dynamics:

- **Near origin**: $|\mathbf{x}|$ small means $|\mathbf{x}^3|$ very small, slow dynamics
- **Far from origin**: Large state leads to strong nonlinear effects
- **Sign preservation**: Cubing preserves sign unlike squaring

This creates a "soft" bounded system - trajectories spiral but don't explode.
