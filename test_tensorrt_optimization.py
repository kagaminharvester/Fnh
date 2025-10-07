import unittest
import os
import logging
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add project root to Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from detection.cd.stage_1_cd import _get_optimized_model_path

class TestTensorRTOptimization(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and dummy files for testing."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pt")
        self.engine_path = os.path.join(self.temp_dir, "test_model.engine")

        # Create a dummy .pt file to prevent YOLO from trying to download it
        with open(self.model_path, 'w') as f:
            f.write('dummy content')

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    @patch('detection.cd.stage_1_cd.torch.cuda.is_available', return_value=False)
    def test_no_cuda_device(self, mock_is_available):
        """Test that the .pt model is returned when no CUDA device is available."""
        result_path = _get_optimized_model_path(self.model_path, self.logger)
        self.assertEqual(result_path, self.model_path)

    @patch('detection.cd.stage_1_cd.torch.cuda.is_available', return_value=True)
    @patch('detection.cd.stage_1_cd.torch.cuda.get_device_capability', return_value=(6, 0))
    def test_incompatible_gpu(self, mock_get_capability, mock_is_available):
        """Test that the .pt model is returned for incompatible GPUs."""
        result_path = _get_optimized_model_path(self.model_path, self.logger)
        self.assertEqual(result_path, self.model_path)

    @patch('detection.cd.stage_1_cd.torch.cuda.is_available', return_value=True)
    @patch('detection.cd.stage_1_cd.torch.cuda.get_device_capability', return_value=(8, 6))
    def test_engine_already_exists(self, mock_get_capability, mock_is_available):
        """Test that an existing .engine file is returned."""
        with open(self.engine_path, 'w') as f:
            f.write('dummy engine')

        result_path = _get_optimized_model_path(self.model_path, self.logger)
        self.assertEqual(result_path, self.engine_path)

    @patch('detection.cd.stage_1_cd.constants')
    @patch('detection.cd.stage_1_cd.YOLO')
    @patch('detection.cd.stage_1_cd.torch.cuda.is_available', return_value=True)
    @patch('detection.cd.stage_1_cd.torch.cuda.get_device_capability', return_value=(8, 6))
    def test_engine_generation(self, mock_get_capability, mock_is_available, mock_yolo, mock_constants):
        """Test that a TensorRT engine is generated if it doesn't exist."""
        mock_constants.DEVICE = 'cpu'
        mock_model_instance = MagicMock()

        def create_engine_file(*args, **kwargs):
            with open(self.engine_path, 'w') as f:
                f.write('dummy engine')

        mock_model_instance.export.side_effect = create_engine_file
        mock_yolo.return_value = mock_model_instance

        result_path = _get_optimized_model_path(self.model_path, self.logger)

        mock_yolo.assert_called_with(self.model_path)
        mock_model_instance.export.assert_called_with(format="engine", device='cpu')

        self.assertEqual(result_path, self.engine_path)
        self.assertTrue(os.path.exists(self.engine_path))

if __name__ == '__main__':
    unittest.main()