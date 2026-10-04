from torch import Tensor, nn


class NodeStateUpdater(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
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
        return self.gru(messages, hidden_states)
