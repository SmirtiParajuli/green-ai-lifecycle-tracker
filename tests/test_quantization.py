"""
Tests for the model quantization functionality.
"""

import unittest
import torch
from src.models import SimpleCNN
from optimisation.quantization import quantize_model

class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.model = SimpleCNN()
    
    def test_model_quantization(self):
        # Test that model can be quantized
        quantized_model = quantize_model(self.model)
        self.assertIsNotNone(quantized_model)
        
        # Test that quantized model is smaller
        original_size = sum(p.numel() for p in self.model.parameters())
        quantized_size = sum(p.numel() for p in quantized_model.parameters())
        self.assertLess(quantized_size, original_size)

if __name__ == '__main__':
    unittest.main()