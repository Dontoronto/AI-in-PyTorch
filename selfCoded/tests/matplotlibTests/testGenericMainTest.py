import unittest
from unittest.mock import patch
import numpy as np

# Assuming plot_model_comparison is part of a class named ModelVisualizer
from .genericTableBelowSubplotsTest import plot_model_comparison

class TestModelComparisonPlotting(unittest.TestCase):

    @patch('your_module.plt')
    def test_empty_inputs(self, mock_plt):
        """Test the method with empty input lists."""
        with self.assertRaises(IndexError):
            plot_model_comparison([], [], np.array([]), [], [])

    @patch('your_module.plt')
    def test_single_image_single_model(self, mock_plt):
        """Test the method with a single image and a single model result."""
        plot_model_comparison([np.random.rand(10, 10)], [[np.random.rand(10, 10)]], np.array([[1]]), ['Row 1'], ['Col 1'])
        # Verify that plot creation methods are called
        self.assertTrue(mock_plt.subplots.called)
        self.assertTrue(mock_plt.show.called)

    @patch('your_module.plt')
    def test_multiple_images_multiple_models(self, mock_plt):
        """Test the method with multiple images and multiple model results."""
        plot_model_comparison([np.random.rand(10, 10) for _ in range(3)],
                                              [[np.random.rand(10, 10) for _ in range(2)] for _ in range(3)],
                                              np.array([[1, 2], [3, 4], [5, 6]]),
                                              ['Row 1', 'Row 2', 'Row 3'],
                                              ['Col 1', 'Col 2'])
        # Verify that plot creation methods are called
        self.assertTrue(mock_plt.subplots.called)
        self.assertTrue(mock_plt.show.called)

    @patch('your_module.plt')
    def test_invalid_table_data_dimensions(self, mock_plt):
        """Test the method with table data dimensions that don't match the specified rows and columns."""
        with self.assertRaises(ValueError):
            plot_model_comparison([np.random.rand(10, 10)], [[np.random.rand(10, 10)]], np.array([[1, 2], [3, 4]]), ['Row 1'], ['Col 1'])

if __name__ == '__main__':
    unittest.main()

#%%

#%%
