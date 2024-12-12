import unittest

import numpy as np

from building_predictor import BuildingPredictor, BuildingDS, train_model


class TestPredictor(unittest.TestCase):
    def test_train_model(self):
        # Create dummy data
        contexts = np.random.rand(100, 10)
        actions = np.random.rand(100, 6)
        outcomes = np.sin(contexts.sum(axis=1) + actions.sum(axis=1))

        ds = BuildingDS(contexts, actions, outcomes)

        model = train_model(ds, epochs=1000, batch_size=16, device="cuda:0")
