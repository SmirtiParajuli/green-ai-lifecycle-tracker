"""
Tests for the energy tracking functionality.
"""

import unittest
import torch
from src.tracker import EnergyTracker

class TestEnergyTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = EnergyTracker()
    
    def test_start_tracking(self):
        # Test that tracking can be started
        self.tracker.start()
        self.assertTrue(self.tracker.is_tracking)
    
    def test_stop_tracking(self):
        # Test that tracking can be stopped
        self.tracker.start()
        measurements = self.tracker.stop()
        self.assertFalse(self.tracker.is_tracking)
        self.assertIsNotNone(measurements)

if __name__ == '__main__':
    unittest.main()