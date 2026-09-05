
"""Dropout-placement ablation model for the cylinder psi-p PINN.

This file intentionally does NOT replace pinn_model.py.

Supported placement modes map the reviewer's requested configurations to
specific hidden-layer dropout locations:

    none        : no dropout
    input       : after the first hidden activation ("near the input")
    middle      : after the central hidden activations
    output      : after the last hidden activation ("near the output")
    alternating : after hidden layers 1,3,5,... in zero-based indexing
    all         : after every hidden activation

Dropout layers contain no trainable parameters, so all placement variants use
the same trainable backbone parameter count.
"""

from __future__ import annotations

import torch.nn as nn

from pinn_model import PINN_Net


VALID_DROPOUT_PLACEMENTS = (
    "none",
    "input",
    "middle",
    "output",
    "alternating",
    "all",
)


def selected_hidden_layers(n_hidden: int, placement: str):
    placement = str(placement).lower()

    if placement not in VALID_DROPOUT_PLACEMENTS:
        raise ValueError(
            f"Unknown dropout placement '{placement}'. "
            f"Choose from {VALID_DROPOUT_PLACEMENTS}."
        )

    if n_hidden < 1:
        raise ValueError("Network must contain at least one hidden layer.")

    if placement == "none":
        return []

    if placement == "input":
        return [0]

    if placement == "output":
        return [n_hidden - 1]

    if placement == "middle":
        if n_hidden == 1:
            return [0]
        if n_hidden % 2 == 0:
            return [n_hidden // 2 - 1, n_hidden // 2]
        return [n_hidden // 2]

    if placement == "alternating":
        return list(range(0, n_hidden, 2))

    if placement == "all":
        return list(range(n_hidden))

    raise AssertionError("Unreachable placement branch.")


class PlacementPINNNet(PINN_Net):
    """Psi-p PINN with explicit dropout placement control."""

    def __init__(
        self,
        layer_mat,
        dropout_rate: float = 0.0,
        dropout_placement: str = "all",
    ):
        # Initialize inherited methods/attributes without inserting legacy dropout.
        super().__init__(layer_mat, dropout_rate=0.0)

        self.dropout_rate = float(dropout_rate)
        self.dropout_placement = str(dropout_placement).lower()

        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError("dropout_rate must satisfy 0 <= p < 1.")

        n_hidden = len(self.layer_mat) - 2
        selected = set(
            selected_hidden_layers(
                n_hidden=n_hidden,
                placement=self.dropout_placement,
            )
        )

        # A zero dropout rate is equivalent to no stochastic masking, but the
        # requested placement is still recorded for traceability.
        layers = []

        for hidden_idx in range(n_hidden):
            layers.append(
                nn.Linear(
                    self.layer_mat[hidden_idx],
                    self.layer_mat[hidden_idx + 1],
                )
            )
            layers.append(nn.Tanh())

            if (
                self.dropout_rate > 0.0
                and hidden_idx in selected
            ):
                layers.append(nn.Dropout(p=self.dropout_rate))

        layers.append(
            nn.Linear(
                self.layer_mat[-2],
                self.layer_mat[-1],
            )
        )

        self.base = nn.Sequential(*layers)
        self.init_weights()

        self.dropout_hidden_layer_indices = sorted(selected)

    def dropout_configuration(self):
        return {
            "dropout_rate": self.dropout_rate,
            "dropout_placement": self.dropout_placement,
            "dropout_hidden_layer_indices": list(
                self.dropout_hidden_layer_indices
            ),
            "hidden_layer_count": len(self.layer_mat) - 2,
        }
