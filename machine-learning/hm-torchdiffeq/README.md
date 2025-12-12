# hm-torchdiffeq

This experiment learns the dynamics of a two dimensional state vector

```math
\mathbf{x}(t)=
\begin{bmatrix}
x_1(t) & x_2(t)
\end{bmatrix},
```

whose time derivative is governed by the cubic interaction with the constant system matrix $\mathbf{A}$:

```math
\frac{\mathrm{d}\mathbf{x}(t)}{\mathrm{d}t}
= \bigl(\mathbf{x}(t)^{\odot 3}\bigr)\mathbf{A}, \quad
\mathbf{A} =
\begin{bmatrix}
-0.1 & 2.0\\
-2.0 & -0.1
\end{bmatrix},
```

where $\mathbf{x}(t)^{\odot 3}$ denotes the elementwise cube of the state vector. The trajectory is initialized at
$\mathbf{x}(0) = [2.0,\, 0.0]$ and integrated over a uniform grid of $1000$ time points in the interval $[0, 25]$ using the Dormand–Prince method with the dopri5 scheme provided by `torchdiffeq`.
