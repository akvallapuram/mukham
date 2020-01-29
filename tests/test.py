import unittest
import os
import cv2

class MukhamTest(unittest.TestCase):
    """
        Some simple functionality tests. 
    """

    def test_DimensionError(self):
        """Test if DimensionError is raised"""
        from mukham.mukham.detector import DimensionError, detect_largest_face
        self.assertRaises(DimensionError, detect_largest_face, 'images/kobe_large.jpg')

    def test_detector(self):
        """Test if the detector works."""
        from mukham.mukham.detector import detect_largest_face
        bounding_box = detect_largest_face('images/bryant_daughter.jpg')
        img = cv2.imread('images/bryant_daughter.jpg')
        self.assertTrue(bounding_box[0][0] < img.shape[1])
        self.assertTrue(bounding_box[1][0] < img.shape[1])
        self.assertTrue(bounding_box[0][1] < img.shape[0])
        self.assertTrue(bounding_box[1][1] < img.shape[0])
