from collections.abc import Sequence

import torch
from torch import Tensor


def aggregate_messages(
    hidden_states: Tensor,
    routes: Sequence[tuple[int, int]],
) -> Tensor:
    messages = torch.zeros_like(hidden_states)
    message_counts = torch.zeros(
        (hidden_states.shape[-2], 1),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for sender, receiver in routes:
        messages[..., receiver, :] += hidden_states[..., sender, :]
        message_counts[receiver] += 1

    return messages / message_counts.clamp_min(1)
