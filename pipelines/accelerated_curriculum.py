"""Bounded, per-scenario imitation bootstrap followed by unchanged mastery gates.

Demonstrator labels are training-only. Evaluations always start at the base and
use only the learned policy. Interaction and supervised update counters are separate.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from pipelines.specialist_policy import ScenarioPolicy
from robot_expedition.planetary.campaign_curriculum import CampaignCurriculum
from robot_expedition.planetary.campaign import catalog_hash, SCENARIO_IDS


def stage_indices(stages, rng, size=128):
    groups = [np.flatnonzero(stages == stage) for stage in np.unique(stages)]
    chosen = rng.integers(len(groups), size=size)
    return np.asarray([rng.choice(groups[group]) for group in chosen])


class AcceleratedCurriculumCallback(BaseCallback):
    def __init__(self, config, model_dir, parent, guard):
        super().__init__()
        self.config, self.directory, self.parent, self.guard = config, model_dir, parent, guard
        self.settings = config["accelerated_curriculum"]
        self.record = {"phase": "preparing", "updated_at": time.time()}
        self.finished = False

    def publish(self, **values):
        from pipelines.train import save_json
        self.record.update(values, updated_at=time.time())
        save_json(self.directory / "acceleration.json", self.record)

    def check_stop(self):
        stop = os.environ.get("EXPEDITION_STOP_FILE")
        if stop and Path(stop).exists():
            raise KeyboardInterrupt("Accelerated training stopped; approved checkpoint preserved")

    def evaluate(self, level, count, phase, seed):
        from pipelines.train import _LivePPOPolicy, save_json
        from pipelines.planetary_evaluate import evaluate_planetary
        self.publish(phase=phase, evaluated_level=level, evaluation={"completed":0,"total":count,"successes":0})
        report = evaluate_planetary(_LivePPOPolicy(self.model, self.training_env),
            {**self.config["environment"], "record_episodes":False}, episodes=count,
            seed_start=seed, scenario_level=level,
            abort_below_success_rate=.85 if count >= 50 else None,
            on_progress=lambda value: self.publish(evaluation=value))
        report.update(training_step=int(self.model.num_timesteps),
                      policy_version=f"{self.directory.name}:imitation:{self.record.get('updates',0)}")
        save_json(self.directory / f"accelerated-{phase}-{level}.json", report)
        return report

    def demonstration(self, level):
        from scripts.build_stage_bank import build
        practice = self.directory / "practice"
        practice.mkdir(exist_ok=True)
        bank = practice / f"scenario_{level:02d}.json"
        data_path = practice / f"scenario_{level:02d}.npz"
        observations, actions, stages = [], [], []
        for index in range(int(self.settings.get("demonstrations",4))):
            self.check_stop()
            self.publish(phase="demonstration", demonstration=index+1)
            scratch = practice / "demonstration-scratch.npz"
            build(bank, scratch, level=level, seed=700001+index)
            with np.load(scratch) as data:
                observations.append(data["observations"]); actions.append(data["actions"]); stages.append(data["stages"])
            scratch.unlink()
        arrays = dict(observations=np.concatenate(observations), actions=np.concatenate(actions), stages=np.concatenate(stages))
        np.savez_compressed(data_path, **arrays, catalog_hash=catalog_hash())
        self.publish(stages=[{"stage":int(stage), "examples":int(sum(arrays["stages"] == stage))}
                             for stage in np.unique(arrays["stages"])])
        return arrays

    def _on_training_start(self):
        from pipelines.train import save_json, sha256_file
        if not isinstance(self.model.policy, ScenarioPolicy) or self.guard is None:
            self.publish(phase="ppo", note="Waiting for mastered-level scenario policy")
            return
        save_json(self.directory / "progress.json", {"total_timesteps":int(self.model.num_timesteps), "updates":int(self.model._n_updates)})
        history_path = self.directory / "acceleration-history.json"
        old_history = self.parent / history_path.name if self.parent else None
        history = json.loads(old_history.read_text()) if old_history and old_history.exists() else {}
        manager = CampaignCurriculum(self.config["environment"]["curriculum_state"],
                                     **self.config["environment"].get("curriculum_settings",{}))
        while len(manager.mastered) < len(SCENARIO_IDS):
            level = manager.current_level
            entry = history.get(str(level), {"attempts":0})
            if entry["attempts"] >= int(self.settings.get("max_attempts_per_level",2)):
                self.publish(phase="ppo", level=level, note="Bounded imitation attempts used; continuing reinforcement learning")
                break
            entry["attempts"] += 1; history[str(level)] = entry
            save_json(history_path, history)
            self.publish(level=level, updates=0, max_updates=int(self.settings.get("max_updates",2000)),
                         phase="demonstration", evaluation=None, stages=[])
            self.check_stop()
            try:
                data = self.demonstration(level)
            except (RuntimeError, ValueError, KeyError) as error:
                entry["error"] = str(error)[:400]
                save_json(history_path, history)
                self.publish(phase="ppo", note="Demonstration rejected; continuing PPO", error=entry["error"])
                break
            actor = self.model.policy.mlp_extractor.policy_net.actors[level-1]
            self.model.policy.mlp_extractor.policy_net.frozen_levels[level-1] = False
            x = torch.tensor(self.training_env.normalize_obs(data["observations"]), device=self.model.device)
            y = torch.tensor(data["actions"], device=self.model.device)
            rng = np.random.default_rng(4100+level+entry["attempts"])
            optimizer = torch.optim.Adam(actor.parameters(), lr=float(self.settings.get("learning_rate",.001)))
            report = self.evaluate(level, 3, "diagnostic", 750000+level*100)
            def score(r):
                m=r["metrics"]
                return (m["success_rate"],m["samples_collected_mean"],-m["collisions_per_episode"],-m["steps_mean"])
            best_score, best = score(report), deepcopy(actor.state_dict())
            chosen_updates = 0
            for update in range(1, int(self.settings.get("max_updates",2000))+1):
                self.check_stop()
                indices = torch.tensor(stage_indices(data["stages"],rng),device=self.model.device)
                loss = torch.nn.functional.cross_entropy(actor(x[indices]),y[indices])
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(),.5); optimizer.step()
                if update % 50 == 0:
                    with torch.no_grad():
                        predicted = actor(x).argmax(dim=1)
                        stages = [{"stage":int(s), "examples":int(sum(data["stages"]==s)),
                                   "imitation_accuracy":float((predicted[torch.tensor(data["stages"]==s,device=x.device)] == y[torch.tensor(data["stages"]==s,device=x.device)]).float().mean())}
                                  for s in np.unique(data["stages"])]
                    self.publish(phase="imitation",updates=update, loss=float(loss.detach()), stages=stages)
                if update % int(self.settings.get("diagnostic_every",500)) == 0:
                    report = self.evaluate(level,3,"diagnostic",750000+level*100)
                    if score(report) > best_score:
                        best_score, best, chosen_updates = score(report), deepcopy(actor.state_dict()), update
                    self.publish(diagnostic=report["metrics"])
                    if report["metrics"]["success_rate"] == 1 and report["metrics"]["collisions_per_episode"] <= 2:
                        break
            actor.load_state_dict(best)
            entry.update(selected_updates=chosen_updates, diagnostic_score=best_score, stages=self.record.get("stages",[]))
            # Discard PPO momentum from the pre-imitation policy for this actor only.
            for parameter in actor.parameters():
                self.model.policy.optimizer.state.pop(parameter,None)
            report = self.evaluate(level,50,"mastery",910000)
            passed = manager.evaluate_scenario_mastery(report)
            if passed:
                for previous in manager.mastered:
                    retention = self.evaluate(previous,50,"retention",920000)
                    if not manager.evaluate_scenario_mastery(retention):
                        passed=False
                        break
            entry["passed"] = passed
            save_json(history_path,history)
            if not passed:
                self.publish(phase="ppo",note="Full gate not passed; continue practice and PPO", level=level)
                break
            manager.advance_scenario()
            self.guard.protect_mastery(manager.mastered)
            checkpoint=self.directory / f"mastered-scenario-{level:02d}-{self.model.num_timesteps}"
            checkpoint.mkdir(exist_ok=True)
            self.model.save(checkpoint / "ppo_model.zip")
            self.training_env.save(checkpoint / "vecnormalize.pkl")
            torch.save(self.model.policy.optimizer.state_dict(),checkpoint / "optimizer.pt")
            save_json(checkpoint / "checkpoint_state.json",{"total_timesteps":int(self.model.num_timesteps)})
            save_json(checkpoint / "curriculum_state.json",manager.snapshot())
            save_json(checkpoint / "sha256.json",{p.name:sha256_file(p) for p in checkpoint.iterdir() if p.is_file() and p.name != "sha256.json"})
            # One representative learned-policy replay, not every training transition.
            from pipelines.planetary_evaluate import evaluate_planetary
            from pipelines.train import _LivePPOPolicy
            self.publish(phase="replay", level=level)
            successful_seed = next(case["seed"] for case in report["cases"] if case["success"])
            if self.config["environment"].get("record_episodes"):
                replay = evaluate_planetary(_LivePPOPolicy(self.model,self.training_env),
                    self.config["environment"],episodes=1,seed_start=successful_seed,scenario_level=level)
                save_json(self.directory / f"accelerated-replay-{level}.json",replay)
            self.publish(phase="approved", mastered=manager.mastered, level=level)
        save_json(history_path,history)
        self.finished=len(manager.mastered)==len(SCENARIO_IDS)
        if self.finished:
            self.publish(phase="all_scenarios_passed",mastered=manager.mastered)

    def _on_step(self):
        return not self.finished
