import unittest

import torch

from nervenet.policies import aggregate_messages


class MessageAggregationTest(unittest.TestCase):
    def test_messages_are_averaged_by_receiver(self) -> None:
        hidden_states = torch.tensor(
            [
                [2.0, 4.0],
                [5.0, 9.0],
                [10.0, 14.0],
            ]
        )

        routes = [
            (0, 1),
            (2, 1),
            (1, 0),
        ]

        result = aggregate_messages(hidden_states, routes)

        expected = torch.tensor(
            [
                [5.0, 9.0],  # Node 0 receives from node 1
                [6.0, 9.0],  # Node 1 averages nodes 0 and 2
                [0.0, 0.0],  # Node 2 receives nothing
            ]
        )

        torch.testing.assert_close(result, expected)

    def test_batched_graphs_are_aggregated_independently(self) -> None:
        hidden_states = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ],
            ]
        )

        routes = [
            (0, 1),
            (1, 0),
        ]

        result = aggregate_messages(hidden_states, routes)

        expected = torch.tensor(
            [
                [
                    [3.0, 4.0],
                    [1.0, 2.0],
                ],
                [
                    [30.0, 40.0],
                    [10.0, 20.0],
                ],
            ]
        )

        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
