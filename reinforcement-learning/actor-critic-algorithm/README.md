# Actor-Critic Algorithm

Based on the foundational work in policy gradient methods, particularly [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) by Konda and Tsitsiklis (1999).

## Core Idea

Actor-Critic combines two neural networks working together:

| Component | Role | Output |
| --- | --- | --- |
| **Actor** | Learns *what to do* (policy) | Probability distribution over actions |
| **Critic** | Learns *how good* states are (value) | Scalar value estimate |

The actor proposes actions, the critic evaluates them, and both learn from the feedback. This is analogous to a student (actor) and teacher (critic) - the student tries actions, and the teacher provides feedback on how good they were.

## Intuition

| REINFORCE (no critic) | Actor-Critic |
| --- | --- |
| "This episode got 100 reward, so all actions were good" | "Action at step 5 was 20 points better than expected" |
| Uses total episode return for all actions | Evaluates each action relative to baseline |
| High variance, slow learning | Lower variance, faster learning |

The key insight: **don't just ask "was the reward good?"** - ask **"was the reward better than expected?"**

## The Math

### Value Function (Critic)

The state value function estimates expected future rewards:

```math
V(s) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]
```

where $\gamma \in [0, 1]$ is the discount factor that makes future rewards worth less than immediate ones.

### Policy (Actor)

The policy $\pi_\theta(a|s)$ outputs a probability distribution over actions given a state:

```math
\pi_\theta(a|s) = P(\text{action} = a \mid \text{state} = s)
```

For discrete actions, this is typically a softmax over network outputs:

```math
\pi_\theta(a|s) = \frac{e^{z_a}}{\sum_{a'} e^{z_{a'}}}
```

### Return and Advantage

**Return** $G_t$ is the discounted sum of future rewards from time $t$:

```math
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
```

**Advantage** $A_t$ measures how much better an action was compared to the baseline (expected value):

```math
A_t = G_t - V(s_t)
```

| Advantage Value | Meaning | Effect on Policy |
| --- | --- | --- |
| $A_t > 0$ | Action was better than expected | Increase probability |
| $A_t < 0$ | Action was worse than expected | Decrease probability |
| $A_t = 0$ | Action matched expectation | No change |

### Policy Gradient Theorem

The policy gradient tells us how to update the actor to maximize expected return:

```math
\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)\right]
```

Intuition: Increase log-probability of actions proportional to how good they were (advantage).

### Loss Functions

**Actor (Policy) Loss** - maximize expected advantage-weighted log probabilities:

```math
L_{\text{actor}} = -\frac{1}{T}\sum_{t=1}^{T} \log \pi_\theta(a_t|s_t) \cdot A_t
```

(Negative because we minimize loss but want to maximize return)

**Critic (Value) Loss** - minimize prediction error:

```math
L_{\text{critic}} = \frac{1}{T}\sum_{t=1}^{T} \left(V_\phi(s_t) - G_t\right)^2
```

**Entropy Bonus** - encourages exploration:

```math
H(\pi) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)
```

**Combined Loss**:

```math
L_{\text{total}} = L_{\text{actor}} + c_v \cdot L_{\text{critic}} - c_e \cdot H(\pi)
```

where $c_v$ is the value loss coefficient and $c_e$ is the entropy coefficient.

## Why Use Advantage Instead of Raw Returns?

Consider two episodes:

| Episode | Actions | Returns | Raw Gradient Signal |
| --- | --- | --- | --- |
| A | Same actions | 100 | "All good!" |
| B | Same actions | 500 | "All great!" |

Both episodes reinforce the same behavior, but one contributes 5x more gradient. This causes **high variance**.

With advantage:

| Episode | $V(s)$ | $G_t$ | Advantage $A_t$ |
| --- | --- | --- | --- |
| A | 90 | 100 | +10 (slightly better than expected) |
| B | 490 | 500 | +10 (slightly better than expected) |

Now both contribute equally, centered around "expected performance."

## Implementation

The actor-critic update in `actor_critic_agent.py`:

| Step | Formula | Code |
| --- | --- | --- |
| 1 | $V(s_t)$ | `values = self.critic(states).squeeze()` |
| 2 | $A_t = G_t - V(s_t)$ | `advantages = returns - values.detach()` |
| 3 | Normalize $A_t$ | `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)` |
| 4 | $L_{\text{actor}} = -\mathbb{E}[\log \pi \cdot A]$ | `policy_loss = -(log_probabilities * advantages).mean()` |
| 5 | $L_{\text{critic}} = \text{MSE}(V, G)$ | `value_loss = F.mse_loss(values, returns)` |
| 6 | $H = \mathbb{E}[H(\pi(s))]$ | `entropy = entropies.mean()` |
| 7 | $L = L_{\text{actor}} + c_v L_{\text{critic}} - c_e H$ | `total_loss = policy_loss + c_v * value_loss - c_e * entropy` |

## Actor-Critic Architecture

```math
\begin{aligned}
\text{Actor: } & \pi_\theta(a|s) = \text{softmax}(\text{MLP}_\theta(s)) \\
\text{Critic: } & V_\phi(s) = \text{MLP}_\phi(s) \\
\text{Action: } & a \sim \pi_\theta(\cdot|s) \\
\text{Update: } & \theta \leftarrow \theta - \alpha \nabla_\theta L_{\text{total}}
\end{aligned}
```

## Comparison with Other Methods

| Method | Pros | Cons |
| --- | --- | --- |
| **REINFORCE** | Simple, unbiased | High variance, slow |
| **Actor-Critic (this)** | Lower variance, online learning | Biased (depends on critic accuracy) |
| **PPO** | Stable, widely used | More complex, requires clipping |
| **DQN** | Works well for discrete actions | No continuous actions, off-policy only |

## Hyperparameters

| Parameter | Symbol | Typical Value | Effect |
| --- | --- | --- | --- |
| Learning rate | $\alpha$ | 0.001-0.01 | Too high = unstable, too low = slow |
| Discount factor | $\gamma$ | 0.99 | Higher = considers more future rewards |
| Entropy coefficient | $c_e$ | 0.01 | Higher = more exploration |
| Value loss coefficient | $c_v$ | 0.5 | Balances actor vs critic learning |

## Advantages

- **Lower variance**: Critic provides baseline, reducing gradient noise
- **Online learning**: Can update every step (not just end of episode)
- **Continuous actions**: Easily extends to continuous action spaces
- **Foundation**: Basis for advanced methods (A3C, PPO, SAC)
