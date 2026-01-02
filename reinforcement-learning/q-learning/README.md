# Q-Learning

A model-free reinforcement learning algorithm that learns the optimal action-selection policy by iteratively updating Q-values based on rewards and future value estimates.

## Core Idea

Q-Learning learns a **Q-function** (also called action-value function) that estimates the expected cumulative reward of taking an action $a$ in state $s$ and then following the optimal policy. The "Q" stands for "quality" - how good is it to take a particular action in a particular state.

Unlike model-based methods, Q-Learning doesn't need to know the environment's dynamics (transition probabilities). It learns purely from experience through trial and error.

## Intuition

| Concept                  | Analogy                                                        |
| ------------------------ | -------------------------------------------------------------- |
| Q-table                  | A cheat sheet mapping "situation + action" to "expected score" |
| Learning                 | Updating the cheat sheet after each attempt                    |
| Exploration ($\epsilon$) | Sometimes trying random moves to discover better strategies    |
| Exploitation             | Using the cheat sheet to pick the best-known action            |

Think of it like learning to navigate a maze:

- You don't know where the exit is initially
- Each time you reach the exit, you get a reward
- Over time, you learn which turns lead to the exit faster
- The Q-table remembers: "at this intersection, turning right is better"

## The Math

### The Q-Function

The optimal Q-function satisfies the **Bellman optimality equation**:

```math
Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a')\right]
```

- $Q^*(s, a)$ - optimal Q-value for state $s$ and action $a$ (scalar)
- $\mathbb{E}$ - expectation over next state $s'$ and reward $r$ (requires knowing $P(s'|s,a)$ to compute exactly; Q-learning avoids this by sampling)
- $Q$ - Q-table storing all Q-values ($|S| \times |A|$ matrix)
- $s$ - state index ($s \in \{0, 1, ..., |S|-1\}$)
- $a$ - action index ($a \in \{0, 1, ..., |A|-1\}$)
- $r$ - immediate reward received (scalar)
- $\gamma$ - discount factor, how much we value future rewards ($\gamma \in [0, 1]$)
- $\alpha$ - learning rate, how much to adjust Q-value ($\alpha \in (0, 1]$)
- $s'$ - next state after taking action $a$
- $\max_{a'} Q^*(s', a')$ - best possible future value from next state (scalar)

### The Update Rule

Q-Learning iteratively approximates $Q^*$ using this update:

```math
Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
```

Breaking it down:

| Term                                       | Meaning                                                                 |
| ------------------------------------------ | ----------------------------------------------------------------------- |
| $Q(s, a)$                                  | Current estimate of value                                               |
| $r + \gamma \max_{a'} Q(s', a')$           | **Temporal difference (TD) target** - what we think the value should be |
| $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ | **Temporal difference (TD) error** - how wrong we were                  |
| $\alpha$                                   | Learning rate - how much to adjust                                      |

The update moves $Q(s, a)$ toward the target by a fraction $\alpha$ of the error.

### Relationship: Bellman Equation vs Q-Learning

The **Bellman equation** defines _what_ the optimal Q-function must satisfy. **Q-learning** is _how_ we find it.

| Bellman Equation (the goal)                                           | Q-Learning Update (the method)                                                             |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| $Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a')\right]$ | $Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$ |
| Fixed-point condition                                                 | Iterative approximation                                                                    |
| Requires knowing $P(s' \mid s, a)$                                    | Only needs samples                                                                         |

Rearranging the Bellman equation:

```math
Q^*(s, a) - \left[r + \gamma \max_{a'} Q^*(s', a')\right] = 0
```

At the optimal solution, the difference (temporal difference error) is zero. Q-learning pushes $Q$ toward satisfying this by reducing the temporal difference error with each update.

### Temporal Difference (TD) Learning

Q-Learning is a form of **Temporal difference (TD) learning** - it updates estimates based on other estimates (bootstrapping):

```math
\underbrace{Q(s,a)}_{\text{old estimate}} \leftarrow \underbrace{Q(s,a)}_{\text{old estimate}} + \alpha \left[\underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{TD target}} - \underbrace{Q(s,a)}_{\text{old estimate}}\right]
```

This is powerful because we don't need to wait until the episode ends to learn - we update after every step.

### Exploration vs Exploitation: $\epsilon$-Greedy Policy

To balance learning new things vs using what we know:

```math
a = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon
\end{cases}
```

| $\epsilon$ value | Behavior                                         |
| ---------------- | ------------------------------------------------ |
| High (e.g., 0.5) | More exploration, slower convergence             |
| Low (e.g., 0.1)  | More exploitation, may miss better strategies    |
| Decaying         | Start high, decrease over time (common practice) |

## Implementation

The Q-Learning update in `main.py`:

| Step | Formula                                | Code                                                      |
| ---- | -------------------------------------- | --------------------------------------------------------- |
| 1    | Select action using $\epsilon$-greedy  | `if random() < epsilon: random_action else: argmax(Q[s])` |
| 2    | Take action, observe $s'$, $r$         | `next_state, reward, is_done = env.step(action)`          |
| 3    | Find best next action                  | `next_action = argmax(Q[s'])`                             |
| 4    | Compute temporal difference (TD)target | `target = r + gamma * Q[s', a']`                          |
| 5    | Update Q-value                         | `Q[s, a] += alpha * (target - Q[s, a])`                   |

## Example Environment

The implementation uses a simple 1D grid world:

```text
[0] - [1] - [2] - [3*]
```

- **States**: 0, 1, 2, 3 (4 positions)
- **Actions**: 0 (left), 1 (right)
- **Goal**: Reach state 3 (reward = 1.0)
- **Optimal policy**: Always go right

After training, the Q-table shows higher values for "right" in each state.

## Hyperparameters

| Parameter        | Symbol     | Typical Range | Effect                                |
| ---------------- | ---------- | ------------- | ------------------------------------- |
| Learning rate    | $\alpha$   | 0.1 - 0.8     | Higher = faster learning, less stable |
| Discount factor  | $\gamma$   | 0.9 - 0.99    | Higher = values future rewards more   |
| Exploration rate | $\epsilon$ | 0.05 - 0.3    | Higher = more random exploration      |
| Episodes         | -          | 100 - 10000+  | More = better convergence             |

## Convergence Guarantee

Q-Learning converges to the optimal Q-function $Q^*$ under these conditions:

1. All state-action pairs are visited infinitely often
2. Learning rate $\alpha$ satisfies: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$
3. Rewards are bounded

In practice, a fixed small $\alpha$ with $\epsilon$-greedy exploration usually works well.

## Comparison with Other Methods

| Property    | Q-Learning             | SARSA                   | Policy Gradient         |
| ----------- | ---------------------- | ----------------------- | ----------------------- |
| Type        | Off-policy             | On-policy               | On-policy               |
| Updates     | Uses max Q (greedy)    | Uses actual next action | Updates policy directly |
| Exploration | Separate from learning | Affects learning        | Built into policy       |
| Convergence | To optimal Q\*         | To Q for current policy | Local optimum           |
| Best for    | Discrete spaces        | Safer learning          | Continuous actions      |

## Limitations

- **Discrete spaces only**: Q-table doesn't scale to continuous states/actions
- **Curse of dimensionality**: Table size = $|S| \times |A|$
- **No generalization**: Similar states are treated independently

These limitations are addressed by **Deep Q-Networks (DQN)**, which replace the Q-table with a neural network.
