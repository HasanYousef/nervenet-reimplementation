from torch import Tensor, nn

from nervenet.policies.defaults import DEFAULT_HIDDEN_SIZE


class MessageNetwork(nn.Module):
    def __init__(self, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.network(hidden_states)
