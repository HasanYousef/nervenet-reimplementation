from collections.abc import Sequence

from torch import Tensor, nn

from nervenet.graphs import BodyNodeType
from nervenet.policies.defaults import DEFAULT_HIDDEN_SIZE
from nervenet.policies.message_aggregation import aggregate_messages
from nervenet.policies.message_network import MessageNetwork
from nervenet.policies.state_updater import NodeStateUpdater


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        super().__init__()

        self.message_network = MessageNetwork(hidden_size)
        self.state_updater = NodeStateUpdater(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        routes: Sequence[tuple[int, int]],
        node_types: Sequence[BodyNodeType],
    ) -> Tensor:
        outgoing_messages = self.message_network(hidden_states)
        messages = aggregate_messages(outgoing_messages, routes)

        return self.state_updater(hidden_states, messages, node_types)
