import numpy as np
from presp.evaluator import Evaluator
from presp.prescriptor import Prescriptor
import torch
from torch.utils.data import DataLoader

from building_predictor import train_model, BuildingDS
from sustaingym.envs.building import BuildingEnv, ParameterGenerator


class BuildingEvaluator(Evaluator):
    """
    Implementation of Evaluator that uses the BuildingEnv from SustainGym.
    """
    def __init__(self, outcomes: list[str], epochs: int, batch_size: int, device: str, n_repeats: int, keep_pct: float):
        super().__init__(outcomes)
        params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson')
        self.env = BuildingEnv(params)

        self.predictor = None
        self.ds = None

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.n_repeats = n_repeats
        self.keep_pct = keep_pct

    def collect_data(self, candidate: Prescriptor, n_repeats: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collects data from the environment using a candidate policy.
        Providing a candidate = None just collects randomly sampled actions.
        Returns:
            Context: NxC
            Actions: NxA
            Outcomes: N
        """
        num_hours = 24
        contexts = []
        actions = []
        outcomes = []
        for _ in range(n_repeats):
            obs, _ = self.env.reset()
            for _ in range(num_hours):
                contexts.append(obs)
                if not candidate:
                    action = self.env.action_space.sample()
                else:
                    obs = torch.tensor(obs, device=candidate.device, dtype=torch.float32)
                    action = candidate.forward(obs).cpu().numpy()
                actions.append(action)
                obs, reward, _, _, _ = self.env.step(action)
                outcomes.append(reward)

        contexts, actions, outcomes = np.array(contexts), np.array(actions), np.array(outcomes)
        return contexts, actions, outcomes

    def update_predictor(self, elites: list[Prescriptor]):
        # Get data from elites
        all_contexts, all_actions, all_outcomes = [], [], []
        for elite in elites:
            contexts, actions, outcomes = self.collect_data(elite, self.n_repeats)
            all_contexts.append(contexts)
            all_actions.append(actions)
            all_outcomes.append(outcomes)
        all_contexts, all_actions, all_outcomes = np.concatenate(all_contexts), np.concatenate(all_actions), np.concatenate(all_outcomes)

        # Merge with old dataset
        if self.ds is not None:
            self.ds.merge(all_contexts, all_actions, all_outcomes, self.keep_pct)
        else:
            self.ds = BuildingDS(all_contexts, all_actions, all_outcomes)

        self.predictor = train_model(self.ds, epochs=self.epochs, batch_size=self.batch_size, device=self.device)

    def evaluate_candidate(self, candidate: Prescriptor) -> np.ndarray:
        """
        Uses predictor to evaluate a candidate.
        """
        dataloader = DataLoader(self.ds, batch_size=self.batch_size, shuffle=False)
        outcomes = []
        self.predictor.to(candidate.device)
        with torch.no_grad():
            for context, _, _ in dataloader:
                context = context.to(candidate.device)
                action = candidate.forward(context)
                outcome = self.predictor(torch.cat([context, action], dim=1))
                outcomes.append(outcome)

        result = np.array([-1 * torch.sum(torch.cat(outcomes)).item()])
        return result


class BuildingValidator(Evaluator):
    """
    Validation evaluator for the BuildingEnv. Doesn't use a predictor just directly checks a candidate against the
    environment. Doesn't update itself.
    """
    def __init__(self):
        params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson')
        self.env = BuildingEnv(params)

    def update_predictor(self, elites: list[Prescriptor]):
        pass

    def evaluate_candidate(self, candidate: Prescriptor) -> np.ndarray:
        num_hours = 24
        obs, _ = self.env.reset(seed=123)
        total_reward = 0
        for _ in range(num_hours):
            obs = torch.tensor(obs, device=candidate.device, dtype=torch.float32)
            action = candidate.forward(obs).cpu().numpy()
            obs, reward, _, _, _ = self.env.step(action)
            total_reward += reward
        return np.array([-1 * total_reward])