from collections.abc import Sequence

from torch import Tensor, nn

from nervenet.policies.message_aggregation import aggregate_messages
from nervenet.policies.state_updater import NodeStateUpdater


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()

        self.state_updater = NodeStateUpdater(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        routes: Sequence[tuple[int, int]],
    ) -> Tensor:
        messages = aggregate_messages(hidden_states, routes)

        return self.state_updater(hidden_states, messages)
