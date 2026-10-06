from collections.abc import Sequence

from torch import Tensor, nn

from nervenet.graphs import GRAPH_OBSERVATION_WIDTH
from nervenet.policies.actuator_decoder import ActuatorDecoder
from nervenet.policies.defaults import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MESSAGE_PASSING_STEPS,
)
from nervenet.policies.graph_processor import GraphProcessor
from nervenet.policies.input_encoder import NodeInputEncoder


class GraphActor(nn.Module):
    def __init__(
        self,
        input_size: int = GRAPH_OBSERVATION_WIDTH,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        message_passing_steps: int = DEFAULT_MESSAGE_PASSING_STEPS,
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
