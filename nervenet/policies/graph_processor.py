from collections.abc import Sequence

from torch import Tensor, nn

from nervenet.policies.message_passing import MessagePassingLayer


class GraphProcessor(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        message_passing_steps: int = 2,
    ) -> None:
        super().__init__()

        if message_passing_steps <= 0:
            raise ValueError("message_passing_steps must be positive")

        self.message_passing_steps = message_passing_steps
        self.layer = MessagePassingLayer(hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        routes: Sequence[tuple[int, int]],
    ) -> Tensor:
        for _ in range(self.message_passing_steps):
            hidden_states = self.layer(hidden_states, routes)

        return hidden_states
