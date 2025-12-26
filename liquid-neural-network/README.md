# Liquid Neural Network

Based on the paper [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439) by Hasani et al. (2020).

## Core Idea

Liquid Neural Networks (LNNs) are continuous-time recurrent neural networks inspired by the nervous system of C. elegans (a small worm with only 302 neurons). Unlike traditional neural networks where parameters are "frozen" after training, LNN parameters dynamically adjust during inference based on the flow of input data, making them behave like "liquid" with fluidity and adaptability.

## Intuition

| Traditional RNN (like a flipbook) | LNN (like a physics simulator) |
| --- | --- |
| Time is discrete (step 1, step 2, ...) | Time flows continuously |
| 1 second or 10 seconds between frames - same logic | Perceives "how much time has passed" |
| Fixed response speed | Adjusts neuron response speed dynamically |

## The Math

### The Continuous ODE

LNNs model hidden state as a differential equation:

```math
\frac{d\mathbf{h}}{dt} = -\frac{1}{\tau} \odot \mathbf{h} + \frac{1}{\tau} \odot f(\mathbf{x}, \mathbf{h})
```

- **Damping term** $-\frac{1}{\tau} \odot \mathbf{h}$: State decays toward zero without input
- **Input term** $\frac{1}{\tau} \odot f(\mathbf{x}, \mathbf{h})$: New information drives the state

### Adaptive Time Constants

The key innovation - $\tau$ is computed dynamically from input and state:

```math
\tau = \sigma(\mathbf{W}_\tau [\mathbf{x}; \mathbf{h}] + \mathbf{b}_\tau)
```

| Environment | $\tau$ | Neuron Behavior |
| --- | --- | --- |
| Changes drastically | Small | "Agile" - quickly captures new changes |
| Stable | Large | "Contemplative" - retains memory |

This is why it's called "liquid" - the dynamics flow and adapt continuously.

### Discretization

Computers cannot run continuous ODEs directly (discrete machines, finite precision), so we discretize using the **exponential Euler method** (see [derivation](#derivation-of-exponential-euler-method)):

```math
\mathbf{h}_{t+1} = \alpha \cdot \mathbf{h}_t + (1 - \alpha) \cdot \tilde{\mathbf{h}}_t
```

where:

- $\alpha = e^{-\Delta t / \tau}$ is the **decay factor**
- $\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_t + \mathbf{b})$ is the **candidate activation**

This is a weighted average: $\alpha$ controls how much old state vs new candidate.

| Parameter | When Large | When Small |
| --- | --- | --- |
| $\Delta t$ (time interval) | $\alpha \to 0$, big update | $\alpha \to 1$, keep old state |
| $\tau$ (time constant) | $\alpha \to 1$, slow adaptation | $\alpha \to 0$, fast adaptation |

## Implementation

The LTC cell forward pass in `main.py`:

| Step | Formula | Code |
| --- | --- | --- |
| 1 | $[\mathbf{x}; \mathbf{h}]$ | `combined = torch.cat([input_tensor, hidden_state], dim=-1)` |
| 2 | $\tau = \sigma(\mathbf{W}\_\tau [\mathbf{x}; \mathbf{h}] + \mathbf{b}\_\tau)$ | `time_constant = self.sigmoid(self.time_constant_weight(combined))` |
| 3 | $\tilde{\mathbf{h}} = \tanh(\mathbf{W}\_x \mathbf{x} + \mathbf{W}\_h \mathbf{h} + \mathbf{b})$ | `input_activation = self.tanh(self.input_weight(x) + self.hidden_weight(h))` |
| 4 | $\alpha = e^{-\Delta t / \tau}$ | `decay_factor = torch.exp(-time_delta / (time_constant + 1e-8))` |
| 5 | $\mathbf{h}_{t+1} = \alpha \mathbf{h}_t + (1 - \alpha) \tilde{\mathbf{h}}_t$ | `new_hidden_state = decay_factor * h + (1 - decay_factor) * input_activation` |

## Liquid Neural Network Architecture

```math
\begin{aligned}
\mathbf{h}_0 &= \mathbf{0} \\
\mathbf{h}_t &= \text{LTC}(\mathbf{x}_t, \mathbf{h}_{t-1}, \Delta t) \quad \text{for } t = 1, \ldots, T \\
\mathbf{y}_t &= \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o
\end{aligned}
```

## Comparison with Traditional RNN/LSTM

| Property | Traditional RNN/LSTM | Liquid Neural Network |
| --- | --- | --- |
| Time view | Discrete stepping | Continuous flow |
| Adaptability | Weights fixed, only state changes | Response speed ($\tau$) changes with input |
| Model size | Many parameters | Minimal (hundreds of neurons for complex tasks) |
| Strengths | Text generation, regular time-series | Irregular sampling, robot control, autonomous driving |

## Advantages

- **Non-uniform sampling**: Naturally handles irregular time intervals (ECG, sensors)
- **Interpretability**: Fewer neurons, easier to observe specific functions
- **Causality**: Less misled by background distractors in autonomous driving
- **Efficiency**: Lower inference overhead than Transformers

---

## Appendix

### Derivation of Exponential Euler Method

How to derive the discrete update rule from the continuous ODE.

Starting from:

```math
\frac{d\mathbf{h}}{dt} = -\frac{1}{\tau} \mathbf{h} + \frac{1}{\tau} f(\mathbf{x}, \mathbf{h})
```

Rewrite as:

```math
\frac{d\mathbf{h}}{dt} = \frac{1}{\tau} (f - \mathbf{h})
```

**Key assumption**: During a small time step $\Delta t$, the input term $f$ is constant.

#### Step 1: Separation of variables

Move terms with $\mathbf{h}$ to left, terms with $t$ to right:

```math
\frac{d\mathbf{h}}{f - \mathbf{h}} = \frac{1}{\tau} dt
```

#### Step 2: Set integration bounds

Integrate from current time $t$ to next time $t + \Delta t$, with state changing from $\mathbf{h}(t)$ to $\mathbf{h}(t + \Delta t)$:

```math
\int_{\mathbf{h}(t)}^{\mathbf{h}(t+\Delta t)} \frac{1}{f - \mathbf{h}} d\mathbf{h} = \int_{t}^{t+\Delta t} \frac{1}{\tau} dt
```

#### Step 3: Evaluate integrals

Left side uses $\int \frac{1}{a-x} dx = -\ln|a-x|$:

```math
[-\ln(f - \mathbf{h})]_{\mathbf{h}(t)}^{\mathbf{h}(t+\Delta t)} = \frac{\Delta t}{\tau}
```

Apply bounds:

```math
-\ln(f - \mathbf{h}(t+\Delta t)) + \ln(f - \mathbf{h}(t)) = \frac{\Delta t}{\tau}
```

Using log properties:

```math
\ln \left( \frac{f - \mathbf{h}(t)}{f - \mathbf{h}(t+\Delta t)} \right) = \frac{\Delta t}{\tau}
```

#### Step 4: Solve for $\mathbf{h}(t + \Delta t)$

Take exponential of both sides:

```math
\frac{f - \mathbf{h}(t)}{f - \mathbf{h}(t+\Delta t)} = e^{\Delta t / \tau}
```

Invert:

```math
f - \mathbf{h}(t+\Delta t) = (f - \mathbf{h}(t)) \cdot e^{-\Delta t / \tau}
```

Solve for $\mathbf{h}(t+\Delta t)$:

```math
\mathbf{h}(t+\Delta t) = f - (f - \mathbf{h}(t)) \cdot e^{-\Delta t / \tau}
```

Expand and rearrange:

```math
\mathbf{h}(t + \Delta t) = \mathbf{h}(t) \cdot e^{-\Delta t / \tau} + f \cdot (1 - e^{-\Delta t / \tau})
```

Let $\alpha = e^{-\Delta t / \tau}$ and $\tilde{\mathbf{h}} = f(\mathbf{x}, \mathbf{h})$:

```math
\mathbf{h}_{t+1} = \alpha \cdot \mathbf{h}_t + (1 - \alpha) \cdot \tilde{\mathbf{h}}_t
```

This is the **exponential Euler method** - more stable than simple Euler because it uses the exact solution of the linear decay part.
