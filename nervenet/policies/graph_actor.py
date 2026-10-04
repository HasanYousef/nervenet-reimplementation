from collections.abc import Sequence

from torch import Tensor, nn

from nervenet.policies.actuator_decoder import ActuatorDecoder
from nervenet.policies.graph_processor import GraphProcessor
from nervenet.policies.input_encoder import NodeInputEncoder


class GraphActor(nn.Module):
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        message_passing_steps: int = 2,
    ) -> None:
        super().__init__()

        self.encoder = NodeInputEncoder(input_size, hidden_size)
        self.processor = GraphProcessor(
            hidden_size,
            message_passing_steps,
        )
        self.decoder = ActuatorDecoder(hidden_size)

    def forward(
        self,
        node_observations: Tensor,
        routes: Sequence[tuple[int, int]],
        actuator_node_indices: Sequence[int],
    ) -> Tensor:
        hidden_states = self.encoder(node_observations)
        hidden_states = self.processor(hidden_states, routes)

        return self.decoder(
            hidden_states,
            actuator_node_indices,
        )