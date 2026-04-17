"""ANFIS model implemented with PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class AnfisForwardResult:
    predictions: Tensor
    membership_values: Tensor
    firing_strengths: Tensor
    normalized_strengths: Tensor
    rule_inputs: Tensor


class GaussianMembershipLayer(nn.Module):
    """Gaussian membership functions for each input dimension."""

    def __init__(self, input_dim: int, num_memberships: int) -> None:
        super().__init__()
        centers = torch.linspace(0.25, 0.75, steps=num_memberships).repeat(input_dim, 1)
        sigmas = torch.full((input_dim, num_memberships), 0.2)
        self.centers = nn.Parameter(centers)
        self.raw_sigmas = nn.Parameter(sigmas.log())

    @property
    def sigmas(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_sigmas) + 1e-4

    def forward(self, inputs: Tensor) -> Tensor:
        expanded_inputs = inputs.unsqueeze(-1)
        centers = self.centers.unsqueeze(0)
        sigmas = self.sigmas.unsqueeze(0)
        exponent = -0.5 * ((expanded_inputs - centers) / sigmas) ** 2
        return torch.exp(exponent)


class ANFIS(nn.Module):
    """Sugeno first-order ANFIS with Gaussian membership functions."""

    def __init__(self, input_dim: int, num_memberships: int = 2) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_memberships <= 0:
            raise ValueError("num_memberships must be positive")

        self.input_dim = input_dim
        self.num_memberships = num_memberships
        self.membership_layer = GaussianMembershipLayer(input_dim, num_memberships)

        rule_indices = list(product(range(num_memberships), repeat=input_dim))
        self.register_buffer("rule_index_map", torch.tensor(rule_indices, dtype=torch.long))
        self.num_rules = len(rule_indices)

        # Each rule learns first-order Sugeno coefficients [x1, ..., xn, bias].
        self.consequents = nn.Parameter(torch.zeros(self.num_rules, input_dim + 1))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_with_details(inputs).predictions

    def forward_with_details(self, inputs: Tensor) -> AnfisForwardResult:
        if inputs.ndim != 2:
            raise ValueError("ANFIS expects a 2D tensor of shape [batch, features]")
        if inputs.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {inputs.shape[1]}")

        membership_values = self.membership_layer(inputs)
        gathered = self._gather_rule_memberships(membership_values)
        firing_strengths = torch.prod(gathered, dim=-1)
        normalized_strengths = firing_strengths / firing_strengths.sum(dim=1, keepdim=True).clamp_min(1e-8)

        inputs_with_bias = torch.cat(
            [inputs, torch.ones(inputs.shape[0], 1, device=inputs.device, dtype=inputs.dtype)],
            dim=1,
        )
        rule_inputs = inputs_with_bias.unsqueeze(1).expand(-1, self.num_rules, -1)
        consequent_outputs = (rule_inputs * self.consequents.unsqueeze(0)).sum(dim=-1)
        predictions = (normalized_strengths * consequent_outputs).sum(dim=1, keepdim=True)

        return AnfisForwardResult(
            predictions=predictions,
            membership_values=membership_values,
            firing_strengths=firing_strengths,
            normalized_strengths=normalized_strengths,
            rule_inputs=rule_inputs,
        )

    def solve_consequents(self, inputs: Tensor, targets: Tensor) -> None:
        """Update consequent parameters via least squares while premise stays fixed."""
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        details = self.forward_with_details(inputs)
        design = details.normalized_strengths.unsqueeze(-1) * details.rule_inputs
        design = design.reshape(inputs.shape[0], -1)

        solution = torch.linalg.lstsq(design, targets).solution
        if solution.numel() == 0:
            return

        solution = solution[: self.num_rules * (self.input_dim + 1)]
        self.consequents.data.copy_(solution.reshape(self.num_rules, self.input_dim + 1))

    def _gather_rule_memberships(self, membership_values: Tensor) -> Tensor:
        transposed = membership_values.transpose(0, 1)
        gathered = []
        for dim_idx in range(self.input_dim):
            membership_for_dim = transposed[dim_idx]
            indices = self.rule_index_map[:, dim_idx]
            gathered.append(membership_for_dim[:, indices])
        return torch.stack(gathered, dim=-1)
