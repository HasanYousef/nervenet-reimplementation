from collections.abc import Sequence

import torch
from torch import Tensor, nn

from nervenet.policies.defaults import DEFAULT_HIDDEN_SIZE


class ActuatorDecoder(nn.Module):
    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=1,
        )

    def forward(
        self,
        hidden_states: Tensor,
        actuator_node_indices: Sequence[int],
    ) -> Tensor:
        node_indices = torch.tensor(
            actuator_node_indices,
            dtype=torch.long,
            device=hidden_states.device,
        )

        actuator_hidden_states = hidden_states.index_select(
            dim=-2,
            index=node_indices,
        )

        return self.linear(actuator_hidden_states).squeeze(-1)
