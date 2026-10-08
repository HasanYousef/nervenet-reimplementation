import torch
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


class DirectActorCriticPolicy(ActorCriticPolicy):
    action_dimension_error = "Actor output size must match the action-space size"

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError("Policy requires a continuous Box action space")

        if self.mlp_extractor.latent_dim_pi != self.action_dist.action_dim:
            raise ValueError(self.action_dimension_error)

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.log_std = nn.Parameter(
            torch.full(
                (self.action_dist.action_dim,),
                self.log_std_init,
            )
        )

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
