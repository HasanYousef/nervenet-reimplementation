import unittest

from nervenet.policies import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MESSAGE_PASSING_STEPS,
    GraphActor,
    GraphProcessor,
)


class PolicyDefaultsTest(unittest.TestCase):
    def test_graph_actor_uses_shared_hidden_size(self) -> None:
        actor = GraphActor()

        self.assertEqual(
            actor.encoder.linear.out_features,
            DEFAULT_HIDDEN_SIZE,
        )
        self.assertEqual(
            actor.decoder.linear.in_features,
            DEFAULT_HIDDEN_SIZE,
        )

    def test_graph_components_use_shared_propagation_depth(self) -> None:
        processor = GraphProcessor()
        actor = GraphActor()

        self.assertEqual(
            processor.message_passing_steps,
            DEFAULT_MESSAGE_PASSING_STEPS,
        )
        self.assertEqual(
            actor.processor.message_passing_steps,
            DEFAULT_MESSAGE_PASSING_STEPS,
        )


if __name__ == "__main__":
    unittest.main()
