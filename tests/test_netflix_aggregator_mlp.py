"""
Test Suite for Aggregator Main Module

This suite includes tests for various functionalities of the aggregator module, ensuring correctness of:
- Network participant discovery
- Parameter extraction and averaging
- Vocabulary creation from TV series data
"""

import shutil
import unittest
from pathlib import Path

import joblib
import numpy as np

from syftbox_netflix.aggregator.pets.fedavg_mlp import (
    extract_number,
    get_users_mlp_parameters,
    mlp_fedavg,
    weighted_average,
)

APP_NAME = "mock_api"
DATA_DIR = "test_sandbox/aggregator/data"
SHARED_FOLDER = "test_sandbox/this_client/api_data/netflix_data"


class TestAggregatorMain_MLP(unittest.TestCase):
    """
    Test cases for verifying functionalities of aggregator module.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by creating a sandbox directory.
        """
        cls.base_path = Path(DATA_DIR)
        if cls.base_path.exists():
            shutil.rmtree(cls.base_path)
        cls.base_path.mkdir(parents=True, exist_ok=True)

        cls.shared_path = Path(SHARED_FOLDER)
        if cls.shared_path.exists():
            shutil.rmtree(cls.shared_path)
        cls.shared_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the sandbox directory after tests are complete.
        """
        if cls.base_path.exists():
            shutil.rmtree(cls.base_path)

        if cls.shared_path.exists():
            shutil.rmtree(cls.shared_path)

    def setUp(self):
        """
        Ensure a clean state before each test by clearing the sandbox directory.
        """
        for item in self.base_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def test_extract_number(self):
        """
        Test extraction of numeric suffixes from filenames.
        Expected: Correct number is extracted; -1 for invalid filenames.
        """
        self.assertEqual(extract_number("netflix_mlp_weights_100.joblib"), 100)
        self.assertEqual(extract_number("invalid_file_name.joblib"), -1)

    def test_get_users_mlp_parameters(self):
        """
        Test extraction of MLP parameters (weights and biases) from directories.
        Expected: Highest numbered files for each peer are selected.
        """

        peers = ["user1", "user2"]

        # Create test directories and files
        user1_dir = self.base_path / "user1" / "app_data" / APP_NAME
        user2_dir = self.base_path / "user2" / "app_data" / APP_NAME
        user1_dir.mkdir(parents=True, exist_ok=True)
        user2_dir.mkdir(parents=True, exist_ok=True)

        (user1_dir / "netflix_mlp_weights_100.joblib").touch()
        (user1_dir / "netflix_mlp_weights_200.joblib").touch()
        (user1_dir / "netflix_mlp_bias_100.joblib").touch()
        (user1_dir / "netflix_mlp_bias_200.joblib").touch()
        (user2_dir / "netflix_mlp_weights_150.joblib").touch()
        (user2_dir / "netflix_mlp_bias_150.joblib").touch()

        weights, biases = get_users_mlp_parameters(self.base_path, APP_NAME, peers)

        expected_weights = [
            user1_dir / "netflix_mlp_weights_200.joblib",
            user2_dir / "netflix_mlp_weights_150.joblib",
        ]
        expected_biases = [
            user1_dir / "netflix_mlp_bias_200.joblib",
            user2_dir / "netflix_mlp_bias_150.joblib",
        ]

        self.assertEqual(weights, expected_weights)
        self.assertEqual(biases, expected_biases)

    def test_weighted_average(self):
        """
        Test computation of weighted averages for MLP parameters.
        Expected: Weighted average is correctly calculated.
        """
        parameters = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        samples = [2, 3]
        result = weighted_average(parameters, samples)
        expected = np.array([2.8, 3.8, 4.8])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mlp_fedavg(self):
        """
        Test federated averaging (FedAvg) for MLP parameters.
        Expected: Weighted average for both weights and biases is correctly computed.
        """
        # Create mock weight and bias files
        weights = [
            self.base_path / "user1_weights.joblib",
            self.base_path / "user2_weights.joblib",
        ]
        biases = [
            self.base_path / "user1_biases.joblib",
            self.base_path / "user2_biases.joblib",
        ]

        weight_data_user1 = [np.array([[1, 2], [3, 4]])]
        weight_data_user2 = [np.array([[5, 6], [7, 8]])]
        bias_data_user1 = [np.array([1, 2])]
        bias_data_user2 = [np.array([3, 4])]

        joblib.dump(weight_data_user1, weights[0])
        joblib.dump(weight_data_user2, weights[1])
        joblib.dump(bias_data_user1, biases[0])
        joblib.dump(bias_data_user2, biases[1])

        fedavg_weights, fedavg_biases = mlp_fedavg(weights, biases)

        expected_weights = [np.array([[3, 4], [5, 6]])]
        expected_biases = [np.array([2, 3])]

        np.testing.assert_array_equal(fedavg_weights[0], expected_weights[0])
        np.testing.assert_array_equal(fedavg_biases[0], expected_biases[0])


if __name__ == "__main__":
    unittest.main()
