"""PPO actor isolation: frozen decisions for levels 1-5, trainable actor for 6-10.

The gate uses only the existing observed difficulty, with frozen VecNormalize.
No planner, absolute coordinates, or hidden goals are available at inference.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor


class SplitActor(nn.Module):
    def __init__(self, actor, head, threshold):
        super().__init__()
        self.frozen = nn.Sequential(deepcopy(actor), deepcopy(head))
        self.specialist = nn.Sequential(deepcopy(actor), deepcopy(head))
        self.frozen.requires_grad_(False)
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    def forward(self, features):
        return torch.where(features[:, -1:] > self.threshold,
                           self.specialist(features), self.frozen(features))


class SpecialistPolicy(ActorCriticPolicy):
    def __init__(self, *args, level_threshold=0.55, **kwargs):
        self.level_threshold = float(level_threshold)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if not isinstance(self.features_extractor, FlattenExtractor):
            raise ValueError("Specialist requires unchanged flattened planetary observations")
        self.mlp_extractor.policy_net = SplitActor(
            self.mlp_extractor.policy_net, self.action_net, self.level_threshold)
        self.mlp_extractor.latent_dim_pi = self.action_space.n
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "level_threshold": self.level_threshold}


def specialize(model, normalizer):
    if normalizer.training:
        raise ValueError("Specialist requires frozen observation normalization")
    if isinstance(model.policy, SpecialistPolicy):
        return False
    if model.observation_space.shape != (132,) or model.action_space.n != 7:
        raise ValueError("Specialist requires planetary history_steps=4 / Discrete(7)")
    source = model.policy
    threshold = (0.55 - normalizer.obs_rms.mean[-1]) / np.sqrt(normalizer.obs_rms.var[-1] + normalizer.epsilon)
    # The clipped normalized values for levels five and six must remain distinct.
    values = [(x-normalizer.obs_rms.mean[-1])/np.sqrt(normalizer.obs_rms.var[-1]+normalizer.epsilon) for x in (0.5,0.6)]
    if not np.clip(values[0], -normalizer.clip_obs, normalizer.clip_obs) < threshold < np.clip(values[1], -normalizer.clip_obs, normalizer.clip_obs):
        raise ValueError("Normalization clips the scenario gate")
    params = source._get_constructor_parameters()
    params.update(lr_schedule=model.lr_schedule, level_threshold=float(threshold))
    target = SpecialistPolicy(**params).to(model.device)
    actor = target.mlp_extractor.policy_net
    for branch in (actor.frozen, actor.specialist):
        branch[0].load_state_dict(source.mlp_extractor.policy_net.state_dict())
        branch[1].load_state_dict(source.action_net.state_dict())
    target.mlp_extractor.value_net.load_state_dict(source.mlp_extractor.value_net.state_dict())
    target.value_net.load_state_dict(source.value_net.state_dict())
    model.policy = target
    model.policy_class = SpecialistPolicy
    model.policy_kwargs = {**model.policy_kwargs, "level_threshold": float(threshold)}
    # New optimizer intentionally starts fresh; source optimizer has a different topology.
    return True


class ScenarioActor(nn.Module):
    """Independent small actors; a buffer freezes approved levels without changing optimizer shape."""
    def __init__(self, actor, head, boundaries):
        super().__init__()
        self.actors = nn.ModuleList([nn.Sequential(deepcopy(actor), deepcopy(head)) for _ in range(10)])
        self.register_buffer("boundaries", torch.tensor(boundaries, dtype=torch.float32))
        self.register_buffer("frozen_levels", torch.tensor([True]*5+[False]*5))
        self.action_count = head.out_features

    def forward(self, features):
        levels = torch.bucketize(features[:, -1].contiguous(), self.boundaries)
        output = features.new_zeros((len(features), self.action_count))
        for level, actor in enumerate(self.actors):
            mask = levels == level
            if mask.any():
                if self.frozen_levels[level]:
                    with torch.no_grad():
                        logits = actor(features[mask])
                else:
                    logits = actor(features[mask])
                output[mask] = logits
        return output


class ScenarioPolicy(ActorCriticPolicy):
    def __init__(self, *args, level_boundaries=None, **kwargs):
        self.level_boundaries = list(level_boundaries if level_boundaries is not None else np.arange(.15, 1, .1))
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if not isinstance(self.features_extractor, FlattenExtractor):
            raise ValueError("ScenarioPolicy requires flattened observations")
        self.mlp_extractor.policy_net = ScenarioActor(self.mlp_extractor.policy_net, self.action_net, self.level_boundaries)
        self.mlp_extractor.latent_dim_pi = self.action_space.n
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "level_boundaries": self.level_boundaries}


def specialize_by_scenario(model, normalizer):
    if normalizer.training or not normalizer.norm_obs:
        raise ValueError("ScenarioPolicy requires frozen observation normalization")
    if isinstance(model.policy, ScenarioPolicy):
        return False
    if isinstance(model.policy, SpecialistPolicy):
        raise ValueError("Convert the original PPO parent, not an earlier split-actor pilot")
    if model.observation_space.shape != (132,) or model.action_space.n != 7:
        raise ValueError("ScenarioPolicy requires history_steps=4 / Discrete(7)")
    source = model.policy
    scale = np.sqrt(normalizer.obs_rms.var[-1] + normalizer.epsilon)
    boundaries = ((np.arange(.15, 1, .1) - normalizer.obs_rms.mean[-1]) / scale).tolist()
    levels = np.clip((np.arange(1,11)/10-normalizer.obs_rms.mean[-1])/scale, -normalizer.clip_obs, normalizer.clip_obs)
    if any(not levels[i] < boundaries[i] < levels[i+1] for i in range(9)):
        raise ValueError("Normalization clips scenario boundaries")
    params = source._get_constructor_parameters()
    params.update(lr_schedule=model.lr_schedule, level_boundaries=boundaries)
    target = ScenarioPolicy(**params).to(model.device)
    for actor in target.mlp_extractor.policy_net.actors:
        actor[0].load_state_dict(source.mlp_extractor.policy_net.state_dict())
        actor[1].load_state_dict(source.action_net.state_dict())
    target.mlp_extractor.value_net.load_state_dict(source.mlp_extractor.value_net.state_dict())
    target.value_net.load_state_dict(source.value_net.state_dict())
    model.policy = target
    model.policy_class = ScenarioPolicy
    model.policy_kwargs = {**model.policy_kwargs, "level_boundaries": boundaries}
    return True


def freeze_mastered(model, levels):
    if isinstance(model.policy, ScenarioPolicy):
        actor = model.policy.mlp_extractor.policy_net
        with torch.no_grad():
            for level in levels:
                actor.frozen_levels[int(level)-1] = True
