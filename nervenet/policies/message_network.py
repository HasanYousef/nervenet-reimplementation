from torch import Tensor, nn


class MessageNetwork(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.network(hidden_states)
