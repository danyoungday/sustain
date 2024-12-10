from abc import ABC, abstractmethod

import numpy as np
from sustaingym.envs.building import BuildingEnv, ParameterGenerator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from presp.predictor import train_model, BuildingDS, BuildingPredictor
from presp.prescriptor import Prescriptor

class Evaluator(ABC):
    """
    Abstract class responsible for evaluating a population of prescriptors as well as updating itself with new data.
    """
    @abstractmethod
    def update_predictor(self, elites: list[Prescriptor]):
        """
        Trains a predictor by collecting training data from the elites.
        """

    def evaluate_population(self, population: list[Prescriptor], force=False, verbose=1) -> np.ndarray:
        """
        Evaluates an entire population of prescriptors.
        Doesn't evaluate prescriptors that already have metrics unless force is True.
        """
        iterator = population if verbose < 1 else tqdm(population, leave=False)
        for candidate in iterator:
            if candidate.metrics is None or force:
                candidate.metrics = self.evaluate_candidate(candidate)

    @abstractmethod
    def evaluate_candidate(self, candidate: Prescriptor) -> np.ndarray:
        """
        Evaluates a single candidate prescriptor.
        """


class BuildingEvaluator(Evaluator):
    """
    Implementation of Evaluator that uses the BuildingEnv from SustainGym.
    """
    def __init__(self, epochs: int, batch_size: int, device: str):
        params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson')
        self.env = BuildingEnv(params)

        self.predictor = None
        self.ds = None

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def collect_data(self, candidate: Prescriptor=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collects data from the environment using a candidate policy.
        Returns:
            Context: NxC
            Actions: NxA
            Outcomes: N
        """
        num_hours = 24
        obs, _ = self.env.reset(seed=123)
        contexts = []
        actions = []
        outcomes = []
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
        all_contexts, all_actions, all_outcomes = [], [], []
        for elite in elites:
            contexts, actions, outcomes = self.collect_data(elite)
            all_contexts.append(contexts)
            all_actions.append(actions)
            all_outcomes.append(outcomes)
        all_contexts, all_actions, all_outcomes = np.concatenate(all_contexts), np.concatenate(all_actions), np.concatenate(all_outcomes)
        ds = BuildingDS(all_contexts, all_actions, all_outcomes)
        self.predictor = train_model(ds, epochs=self.epochs, batch_size=self.batch_size, device=self.device)
        self.ds = ds

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
    def __init__(self):
        params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson')
        self.env = BuildingEnv(params)
        self.predictor = None
        self.ds = None

    def update_predictor(self, elites: list[Prescriptor]):
        pass

    def evaluate_candidate(self, candidate: Prescriptor) -> np.ndarray:
        num_hours = 24
        obs, _ = self.env.reset(seed=123)
        total_reward = 0
        for _ in range(num_hours):
            obs = torch.tensor(obs, device=candidate.device, dtype=torch.float32)
            action = candidate.forward(obs).cpu().numpy()
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
        return np.array([-1 * total_reward])