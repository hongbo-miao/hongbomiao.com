# Quadcopter PD Controller

## Control Formulas

### Position Control (Outer Loop)

**Command Acceleration:**

```math
\ddot{r}_{cmd} = \ddot{r}_{des} + K_d (\dot{r}_{des} - \dot{r}) + K_p (r_{des} - r)
```

where:

- $r = [x, y, z]^T$ is the position
- $K_p$ and $K_d$ are the proportional and derivative gains

### Thrust Control

**Total Thrust:**

```math
F = m (g + \ddot{z}_{cmd})
```

### Attitude Control

**Desired Roll and Pitch:**

```math
\phi_{des} = \frac{1}{g} (\ddot{x}_{cmd} \sin\psi_{des} - \ddot{y}_{cmd} \cos\psi_{des})
```

```math
\theta_{des} = \frac{1}{g} (\ddot{x}_{cmd} \cos\psi_{des} + \ddot{y}_{cmd} \sin\psi_{des})
```

**Control Moment (Inner Loop):**

```math
M = K_{p,ang} (\Phi_{des} - \Phi) + K_{d,ang} (\omega_{des} - \omega)
```

where:

- $\Phi = [\phi, \theta, \psi]^T$ is the Euler angles (roll, pitch, yaw)
- $\omega = [p, q, r]^T$ is the angular velocity

### Equations of Motion

**Translational Dynamics:**

```math
\ddot{r} = \frac{1}{m} R \begin{bmatrix} 0 \\ 0 \\ F \end{bmatrix} - \begin{bmatrix} 0 \\ 0 \\ g \end{bmatrix}
```

where $R$ is the rotation matrix from body to world frame.

**Rotational Dynamics:**

```math
I \dot{\omega} = M - \omega \times (I \omega)
```

where $I$ is the inertia matrix.
