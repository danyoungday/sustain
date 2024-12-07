from abc import ABC, abstractmethod

import numpy as np
from sustaingym.envs.building import BuildingEnv, ParameterGenerator
import torch
from tqdm import tqdm

from presp.prescriptor import Prescriptor

class Evaluator(ABC):
    """
    Abstract class responsible for evaluating a population of prescriptors as well as updating itself with new data.
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
    def __init__(self):
        params = ParameterGenerator(
            building='OfficeSmall', weather='Hot_Dry', location='Tucson')
        self.env = BuildingEnv(params)

    def evaluate_candidate(self, candidate: Prescriptor) -> np.ndarray:
        num_hours = 24
        obs, _ = self.env.reset(seed=123)
        rewards = []
        for _ in range(num_hours):
            obs = torch.tensor(obs, device=candidate.device, dtype=torch.float32)
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)

        return np.array([-1 * sum(rewards)])