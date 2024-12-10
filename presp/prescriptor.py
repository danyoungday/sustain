from abc import ABC, abstractmethod
import copy
from pathlib import Path

import torch


class Prescriptor(ABC):
    """
    Prescriptor class that goes from context to actions.
    """
    def __init__(self):
        self.cand_id = ""
        self.metrics = None
        self.rank = None
        self.distance = None

    @abstractmethod
    def forward(self, context):
        """
        Generates actions from context.
        TODO: Is there a nicer way to have this be extensible?
        """

    @abstractmethod
    def save(self, path: Path):
        """
        Save the prescriptor to file.
        """


class BuildingPrescriptor(Prescriptor):
    """
    Prescriptor for the building environment.
    """
    def __init__(self, model_params: dict, device="cpu"):
        super().__init__()
        self.model_params = {k: v for k, v in model_params.items()}
        self.model = torch.nn.Sequential(
            torch.nn.Linear(model_params["in_size"], model_params["hidden_size"]),
            torch.nn.Tanh(),
            torch.nn.Linear(model_params["hidden_size"], model_params["out_size"])
        )
        self.model.eval()
        self.model.to(device)
        self.device = device

    def forward(self, context):
        """
        Gets sigmoided output between 0 and 1, then multiply by 2 and subtract 1 to get between -1 and 1.
        """
        with torch.no_grad():
            outputs = self.model(context)
            actions = torch.sigmoid(outputs) * 2 - 1
            return actions

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)


class PrescriptorFactory(ABC):
    """
    Abstract class in charge of creating prescriptors.
    Implementations should store details used to create prescriptors.
    """
    @abstractmethod
    def random_init(self) -> Prescriptor:
        """
        Creates a randomly initialized prescriptor model.
        """

    @abstractmethod
    def crossover(self, parents: list[Prescriptor], mutation_rate: float, mutation_factor: float) -> list[Prescriptor]:
        """
        Crosses over N parents to make N children. Mutates the N children.
        """

    @abstractmethod
    def load(self, path: Path) -> Prescriptor:
        """
        Load a prescriptor from file.
        """


class BuildingPrescriptorFactory(PrescriptorFactory):
    """
    Prescriptor factory for the BuildingPrescriptor class.
    Handles model parameters and device for PyTorch.
    """
    def __init__(self, model_params: dict, device="cpu"):
        self.model_params = model_params
        self.device = device

    def random_init(self) -> Prescriptor:
        """
        Creates a prescriptor and randomly orthogonally intializes the weights.
        """
        candidate = BuildingPrescriptor(self.model_params, device=self.device)
        # Orthogonal initialization
        for layer in candidate.model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.01)

        return candidate

    def crossover(self, parents: list[Prescriptor], mutation_rate: float, mutation_factor: float) -> list[Prescriptor]:
        """
        Crossover two parents to create a child.
        Take a random 50/50 choice of either parent's weights.
        Then mutates the child.
        NOTE: The child is returned in a list to fit the abstract crossover method.
        """
        child = BuildingPrescriptor(self.model_params, self.device)
        parent1, parent2 = parents[0], parents[1]
        child.model = copy.deepcopy(parent1.model)
        for child_param, parent2_param in zip(child.model.parameters(), parent2.model.parameters()):
            mask = torch.rand(size=child_param.data.shape, device=self.device) < 0.5
            child_param.data[mask] = parent2_param.data[mask]
        self.mutate_(child, mutation_rate, mutation_factor)
        return [child]

    def mutate_(self, candidate: BuildingPrescriptor, mutation_rate: float, mutation_factor: float):
        """
        Mutates a prescriptor in-place with gaussian percent noise.
        """
        with torch.no_grad():
            for param in candidate.model.parameters():
                mutate_mask = torch.rand(param.shape, device=param.device) < mutation_rate
                noise = torch.normal(0, mutation_factor, param[mutate_mask].shape, device=param.device, dtype=param.dtype)
                param[mutate_mask] += noise * param[mutate_mask]

    def load(self, path: Path) -> Prescriptor:
        """
        Loads torch model from file.
        """
        candidate = BuildingPrescriptor(self.model_params, device=self.device)
        candidate.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return candidate