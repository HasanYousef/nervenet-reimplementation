from torch import Tensor, nn

from nervenet.policies.defaults import DEFAULT_HIDDEN_SIZE


class NodeStateUpdater(nn.Module):
    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        super().__init__()

        self.gru = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )

    def forward(
        self,
        hidden_states: Tensor,
        messages: Tensor,
    ) -> Tensor:
        original_shape = hidden_states.shape

        flat_hidden_states = hidden_states.reshape(
            -1,
            original_shape[-1],
        )
        flat_messages = messages.reshape(
            -1,
            original_shape[-1],
        )

        updated_states = self.gru(
            flat_messages,
            flat_hidden_states,
        )

        return updated_states.reshape(original_shape)
