import torch
import torch.nn.functional as F  # noqa: N812
from models.expert import Expert
from torch import nn


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) Layer.

    Instead of one large network, MoE uses multiple expert networks.
    A router decides which top-k experts process each token.
    Only selected experts are activated (sparse computation).

    Math:
        G(x) = Softmax(W_g * x)                   # Router probabilities
        TopK = {(i, g_i) : i in argmax_k G(x)}    # Select top-k experts
        w_i = g_i / sum(g_j for j in TopK)        # Normalize weights
        y = sum(w_i * E_i(x) for i in TopK)       # Weighted expert outputs

    Example dimensions (batch=16, seq=10, dim=32, experts=4, top_k=2):
        input:              [16, 10, 32]
        input_flat:         [160, 32]       (16*10=160 tokens)
        router_logits:      [160, 4]        (160 tokens, 4 experts)
        router_probs:       [160, 4]        (probabilities sum to 1 per token)
        top_k_weights:      [160, 2]        (top-2 weights per token)
        top_k_indices:      [160, 2]        (which 2 experts selected)
        output:             [16, 10, 32]
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        expert_count: int,
        top_k: int,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.expert_count = expert_count  # N: number of experts
        self.top_k = top_k  # k: number of experts to activate per token
        self.load_balance_weight = load_balance_weight  # alpha: balance loss weight

        # Create N expert networks
        self.experts = nn.ModuleList(
            [
                Expert(input_dimension, hidden_dimension, output_dimension)
                for _ in range(expert_count)
            ],
        )

        # Router: W_g matrix that computes expert selection probabilities
        # W_g shape: [input_dimension, expert_count]
        self.router_weight = nn.Linear(input_dimension, expert_count)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, dimension = input_tensor.shape
        # input_tensor: [batch_size, sequence_length, dimension]
        # Example:      [16, 10, 32]

        # Flatten to process all tokens together
        # input_flat: [batch_size * sequence_length, dimension] = [token_count, dimension]
        # Example:    [160, 32]
        input_flat = input_tensor.view(-1, dimension)
        token_count = input_flat.shape[0]  # Example: 160

        # ============ Step 1: Compute router probabilities ============
        # g = W_g * x
        # router_logits: [token_count, expert_count]
        # Example:       [160, 4]
        router_logits = self.router_weight(input_flat)

        # G(x) = Softmax(g) -> probability distribution over experts
        # router_probabilities: [token_count, expert_count]
        # Example:              [160, 4]
        # Each row sums to 1.0, e.g., [0.1, 0.6, 0.25, 0.05]
        router_probabilities = F.softmax(router_logits, dim=-1)

        # ============ Step 2: Select top-k experts ============
        # TopK(G(x), k) = {(i, g_i) : i in argmax_k G(x)}
        # top_k_weights: [token_count, top_k] - the probability values
        # top_k_indices: [token_count, top_k] - which experts were selected
        # Example:       [160, 2]
        # If probs = [0.1, 0.6, 0.25, 0.05], then:
        #   top_k_weights = [0.6, 0.25]
        #   top_k_indices = [1, 2]
        top_k_weights, top_k_indices = torch.topk(
            router_probabilities,
            self.top_k,
            dim=-1,
        )

        # Normalize weights: w_i = g_i / sum(g_j for j in TopK)
        # top_k_weights.sum(dim=-1, keepdim=True): [token_count, 1]
        # Example: [0.6 + 0.25] = [0.85] -> keepdim makes it [[0.85]]
        # After normalization: [0.6/0.85, 0.25/0.85] = [0.71, 0.29]
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Initialize output tensor
        # output_flat: [token_count, output_dimension]
        # Example:     [160, 32]
        output_flat = torch.zeros(
            token_count,
            self.experts[0].linear_2.out_features,
            device=input_tensor.device,
        )

        # ============ Step 3: Compute weighted sum of expert outputs ============
        # y = sum(w_i * E_i(x) for i in TopK)
        for expert_index, expert in enumerate(self.experts):
            # Find which tokens selected this expert in their top-k
            # (top_k_indices == expert_index): [token_count, top_k] of bool
            # Example for expert_index=1: [[False, True], [True, False], ...]
            # .any(dim=-1): [token_count] - True if expert is in top-k for that token
            # expert_mask: [token_count] of bool
            # Example:     [160] with ~80 True values if balanced
            #              (token_count * top_k / expert_count = 160 * 2 / 4 = 80)
            expert_mask = (top_k_indices == expert_index).any(dim=-1)

            if expert_mask.any():
                # Get tokens that selected this expert
                # expert_input: [num_selected_tokens, dimension]
                # Example:      [~80, 32]
                expert_input = input_flat[expert_mask]

                # E_i(x): run expert network
                # expert_output: [num_selected_tokens, output_dimension]
                # Example:       [~80, 32]
                expert_output = expert(expert_input)

                # Get the weight w_i for this expert (from normalized top_k_weights)
                # weight_mask: [num_selected_tokens, top_k] of bool
                # Example:     [~80, 2]
                weight_mask = top_k_indices[expert_mask] == expert_index
                expert_weights = torch.where(
                    weight_mask,
                    top_k_weights[expert_mask],
                    torch.zeros_like(top_k_weights[expert_mask]),
                )
                # Extract the weight from whichever top-k position this expert is in
                # Token A and Token B are two example tokens (from 160 total) that selected expert 1
                # They have expert 1 at different positions in their top-k:
                #   Token A: top_k_indices=[1,2], weights=[0.71,0.29] -> expert 1 at position 0
                #   Token B: top_k_indices=[3,1], weights=[0.65,0.35] -> expert 1 at position 1
                # torch.where keeps weight where mask is True, else 0.0:
                #   Token A: mask=[True,False]  -> [0.71, 0.0]  -> sum -> 0.71
                #   Token B: mask=[False,True]  -> [0.0, 0.35]  -> sum -> 0.35
                # Combined expert_weights after sum: [[0.71], [0.35]] shape [2, 1]
                # expert_weights: [num_selected_tokens, top_k] -> sum -> [num_selected_tokens, 1]
                # Example:        [~80, 2] -> [~80, 1]
                expert_weights = expert_weights.sum(dim=-1, keepdim=True)

                # y += w_i * E_i(x)
                # expert_weights: [~80, 1], expert_output: [~80, 32]
                # Broadcasting: [~80, 1] * [~80, 32] = [~80, 32]
                output_flat[expert_mask] += expert_weights * expert_output

        # Reshape back to original batch structure
        # output: [batch_size, sequence_length, output_dimension]
        # Example: [16, 10, 32]
        output = output_flat.view(batch_size, sequence_length, -1)

        # Compute auxiliary load balancing loss
        load_balance_loss = self._compute_load_balance_loss(
            router_probabilities,
            top_k_indices,
            token_count,
        )

        return output, load_balance_loss

    def _compute_load_balance_loss(
        self,
        router_probabilities: torch.Tensor,
        top_k_indices: torch.Tensor,
        token_count: int,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert utilization.

        Without this loss, the router might collapse to using only 1-2 experts.

        Math:
            L_balance = N * sum(f_i * p_i for i in 1..N)

        where:
            N = number of experts
            f_i = fraction of tokens routed to expert i
            p_i = mean routing probability to expert i

        Example dimensions (token_count=160, expert_count=4, top_k=2):
            router_probabilities:     shape [token_count, expert_count] = [160, 4]
            top_k_indices:            shape [token_count, top_k] = [160, 2]
            expert_usage_fraction:    shape [expert_count] = [4] e.g., [0.25, 0.25, 0.25, 0.25] if balanced
            mean_routing_probability: shape [expert_count] = [4] e.g., [0.25, 0.25, 0.25, 0.25] if balanced
                                        (mean across all 160 tokens, not a single token's e.g., [0.1, 0.6, 0.25, 0.05])

        Perfect balance (4 experts): f_i = 0.25, p_i = 0.25 -> L = 4 * 4 * 0.0625 = 1.0
        Complete collapse (1 expert): f_i = [1,0,0,0], p_i = [1,0,0,0] -> L = 4 * 1 = 4.0

        Total loss: L_total = L_task + alpha * L_balance
        When balanced, L_balance ~= 1.0, so final loss ~= alpha (e.g., 0.01)
        """
        # f_i: fraction of tokens actually routed to expert i (hard selection result)
        # Based on top-k hard decision: was expert selected or not (0 or 1)
        # expert_usage_fraction: [expert_count]
        # Example:               [4] with values like [0.25, 0.25, 0.25, 0.25] if balanced
        expert_usage_fraction = torch.zeros(
            self.expert_count,
            device=router_probabilities.device,
        )
        for expert_index in range(self.expert_count):
            # Count how many times expert_index appears in top_k_indices
            # Divide by total selections (token_count * top_k)
            # Example: 160 tokens * 2 top_k = 320 total selections
            #          If expert 0 selected 80 times: f_0 = 80/320 = 0.25
            expert_usage_fraction[expert_index] = (
                top_k_indices == expert_index
            ).float().sum() / (token_count * self.top_k)

        # p_i: mean routing probability to expert i (soft probability from router)
        # Based on softmax soft probability: continuous value 0.0 ~ 1.0
        # This is what the router "wanted" before top-k hard selection
        # router_probabilities: [token_count, expert_count] = [160, 4]
        # mean(dim=0): average across tokens -> [expert_count] = [4]
        # Example: [0.25, 0.25, 0.25, 0.25] if balanced
        mean_routing_probability = router_probabilities.mean(dim=0)

        # L_balance = N * sum(f_i * p_i)
        # Multiply f_i * p_i to detect true imbalance:
        #   - f_i high, p_i high -> expert overused (penalize)
        #   - f_i low, p_i low -> expert underused (ok, balanced)
        # expert_usage_fraction * mean_routing_probability: [4] * [4] = [4]
        # .sum(): scalar
        # Example (balanced): 4 * (0.25*0.25 + 0.25*0.25 + 0.25*0.25 + 0.25*0.25)
        #                   = 4 * 0.25 = 1.0
        # Lower is better, but minimum is 1.0 (perfect balance)
        # Maximum is N (e.g., 4.0 when collapsed to 1 expert)
        load_balance_loss = (
            self.expert_count * (expert_usage_fraction * mean_routing_probability).sum()
        )

        # Return alpha * L_balance
        # Example: 0.01 * 1.0 = 0.01 (balanced, good)
        #          0.01 * 4.0 = 0.04 (collapsed, bad)
        return self.load_balance_weight * load_balance_loss
