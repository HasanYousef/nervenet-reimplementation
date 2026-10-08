from collections.abc import Sequence

import torch
from torch import Tensor, nn

from nervenet.graphs import BodyNodeType
from nervenet.policies.defaults import DEFAULT_HIDDEN_SIZE


class NodeStateUpdater(nn.Module):
    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.grus = nn.ModuleDict(
            {
                node_type.value: nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                )
                for node_type in BodyNodeType
            }
        )

    def forward(
        self,
        hidden_states: Tensor,
        messages: Tensor,
        node_types: Sequence[BodyNodeType],
    ) -> Tensor:
        if hidden_states.shape != messages.shape:
            raise ValueError("hidden_states and messages must have the same shape")

        if hidden_states.ndim < 2:
            raise ValueError("hidden_states must include node and hidden dimensions")

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden state width must match hidden_size")

        if len(node_types) != hidden_states.shape[-2]:
            raise ValueError("node_types must contain one type per node")

        if any(not isinstance(node_type, BodyNodeType) for node_type in node_types):
            raise TypeError("every node type must be a BodyNodeType")

        updated_states = torch.empty_like(hidden_states)

        for node_type in BodyNodeType:
            node_indices = [
                index
                for index, current_type in enumerate(node_types)
                if current_type is node_type
            ]

            if not node_indices:
                continue

            index_tensor = torch.tensor(
                node_indices,
                dtype=torch.long,
                device=hidden_states.device,
            )
            type_hidden_states = hidden_states.index_select(-2, index_tensor)
            type_messages = messages.index_select(-2, index_tensor)
            type_shape = type_hidden_states.shape

            type_updated_states = self.grus[node_type.value](
                type_messages.reshape(-1, self.hidden_size),
                type_hidden_states.reshape(-1, self.hidden_size),
            ).reshape(type_shape)

            updated_states = updated_states.index_copy(
                -2,
                index_tensor,
                type_updated_states,
            )

        return updated_states
