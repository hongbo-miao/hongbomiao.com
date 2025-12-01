# Camera-Radar-Lidar Fusion

## Camera-Radar-Lidar Fusion

### Image-Space Association

Each camera detection $j$ is represented by a 2D bounding box
$[x_{1,j}, y_{1,j}, x_{2,j}, y_{2,j}]$ in image coordinates. Radar and lidar
detections provide projected image points $(u_k, v_k)$.

For a radar detection $k$, the distance to box $j$ is

1. If the point lies inside the bounding box,

    ```math
    x_{1,j} \le u_k \le x_{2,j}, \quad y_{1,j} \le v_k \le y_{2,j},
    ```

    then the distance is set to

    ```math
    d_{jk}^{\text{radar}} = 0
    ```

2. Otherwise, we clamp the point to the closest point on the box edges

    ```math
    u_k^{\text{closest}} = \text{clamp}(u_k, x_{1,j}, x_{2,j}), \\
    v_k^{\text{closest}} = \text{clamp}(v_k, y_{1,j}, y_{2,j}),
    ```

    and compute the Euclidean distance in pixels

    ```math
    d_{jk}^{\text{radar}} = \sqrt{\bigl(u_k - u_k^{\text{closest}}\bigr)^2 + \bigl(v_k - v_k^{\text{closest}}\bigr)^2}
    ```

The same formula is used for lidar detections, giving distances $d_{jk}^{\text{lidar}}$ from lidar image points to camera bounding boxes.

Only pairs with distance below a configured threshold $d_{\text{max}}$ (environment variable `ASSOCIATION_DISTANCE_THRESHOLD_PIXELS`) are considered:

```math
d_{jk} \le d_{\text{max}}
```

Among all candidate camera-radar (or camera-lidar) pairs that pass this threshold, the implementation sorts by distance and greedily assigns each camera detection and each radar/lidar detection to at most one partner (one-to-one nearest-neighbor matching in image space).

### Variance-Weighted Distance Fusion

When both lidar and radar distances are available for a fused track, the fused distance is computed using inverse-variance weighting. Let

```math
d_{\text{lidar}}, \quad d_{\text{radar}}
```

be the measured distances, and let

```math
\sigma_{\text{lidar}}, \quad \sigma_{\text{radar}}
```

be their standard deviations (for example, $\sigma_{\text{lidar}} = 0.02\,
\text{m}$, $\sigma_{\text{radar}} = 0.5\,\text{m}$). The variances are

```math
\sigma_{\text{lidar}}^2, \quad \sigma_{\text{radar}}^2,
```

and the corresponding inverse-variance weights are

```math
w_{\text{lidar}} = \frac{1}{\sigma_{\text{lidar}}^2}, \quad
w_{\text{radar}} = \frac{1}{\sigma_{\text{radar}}^2}
```

The fused distance is then

```math
d_{\text{fused}} =
\frac{w_{\text{lidar}}\, d_{\text{lidar}} + w_{\text{radar}} \,
      d_{\text{radar}}}{w_{\text{lidar}} + w_{\text{radar}}}
```

If only one sensor provides a distance, the fused distance defaults to that sensor's measurement (lidar-only or radar-only). If neither is available, the fallback distance is set to zero.

### Fusion Confidence

For each fused track, the fusion confidence is computed from the camera detector confidence. Let

```math
c_{\text{cam}}
```

be the camera detection confidence, and let

```math
\alpha = \text{CAMERA\_CONFIDENCE\_WEIGHT}, \quad
\beta = \text{FUSION\_BASE\_CONFIDENCE}
```

be configuration parameters. The fusion confidence is

```math
c_{\text{fused}} = \alpha\, c_{\text{cam}} + \beta
```

This scalar confidence is stored in each `FusedTrack` alongside the fused distance.

## Occupancy Grid

We represent the occupancy of each voxel $i$ with a probability

```math
p_i = P(\text{occupied}_i)
```

and a corresponding log-odds value

```math
\ell_i = \ln \frac{p_i}{1 - p_i}
```

The inverse relation from log-odds back to probability is

```math
p_i = \frac{1}{1 + e^{-\ell_i}}
```

### Measurement Update (Bayesian Log-Odds)

For each measurement, we use a measurement probability $p_z$ derived from configuration (for example, `OCCUPANCY_OCCUPIED_PROBABILITY_GIVEN_OCCUPIED_EVIDENCE` for occupied evidence and `OCCUPANCY_OCCUPIED_PROBABILITY_GIVEN_FREE_EVIDENCE` for free-space evidence). The corresponding measurement log-odds is

```math
\ell_z = \ln \frac{p_z}{1 - p_z}
```

Given the previous log-odds $\ell_i^{t-1}$, the updated log-odds after incorporating the measurement is

```math
\ell_i^{t} = \ell_i^{t-1} + \ell_z
```

The updated occupancy probability is obtained via the logistic function

```math
p_i^{t} = \frac{1}{1 + e^{-\ell_i^{t}}}
```

### Voxel State Classification

Given occupancy thresholds $p_{\text{occ}}$ and $p_{\text{free}}$, a voxel state is classified as

```math
\text{state}_i^{t} =
\begin{cases}
\text{Occupied} & \text{if } p_i^{t} \ge p_{\text{occ}}, \\
\text{Free}     & \text{if } p_i^{t} \le p_{\text{free}}, \\
\text{Unknown}  & \text{otherwise.}
\end{cases}
```

### Temporal Decay

To keep the grid focused on the recent environment, log-odds are decayed toward zero (corresponding to $p_i = 0.5$, an unknown state). With decay rate $\lambda$ (configured by `OCCUPANCY_DECAY_RATE`), the per-step update is

```math
\ell_i^{t} =
\begin{cases}
\max(0, \ell_i^{t-1} - \lambda) & \text{if } \ell_i^{t-1} > 0, \\
\min(0, \ell_i^{t-1} + \lambda) & \text{if } \ell_i^{t-1} < 0, \\
0                                 & \text{otherwise.}
\end{cases}
```

The corresponding probability \(p_i^{t}\) is recomputed from \(\ell_i^{t}\) using the same logistic relation above.
