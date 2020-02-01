import unittest
import os
import cv2

# path of this file
test_path = os.path.split(os.path.abspath(__file__))[0] + '/'


class MukhamTest(unittest.TestCase):
    """
        Tests for the required functionalities:
        1. Dimension Error: allow only images of max size 1024x1024
        2. Bounding box for the largest face must be returned
        3. save the largest file to an appropriate location. 
    """
    
    def setUp(self):
        """Executed before the any tests start."""
        # remove existing test files
        if os.path.isfile(test_path + 'data/faces/bryant_daughter.jpg'):
            os.remove(test_path + 'data/faces/bryant_daughter.jpg')
    
    def test_DimensionError(self):
        """Test if DimensionError is raised"""
        from mukham.detector import DimensionError, detect_largest_face
        self.assertRaises(DimensionError, detect_largest_face, test_path + 'data/images/kobe_large.jpg')
    
    def test_detector(self):
        """Test if the detector works."""
        from mukham.detector import detect_largest_face
        bounding_box = detect_largest_face(test_path + 'data/images/bryant_daughter.jpg')
        img = cv2.imread(test_path + 'data/images/bryant_daughter.jpg')
        self.assertTrue(bounding_box[0][0] < img.shape[1])
        self.assertTrue(bounding_box[1][0] < img.shape[1])
        self.assertTrue(bounding_box[0][1] < img.shape[0])
        self.assertTrue(bounding_box[1][1] < img.shape[0])
    
    def test_save_file(self):
        """Test if the detector saves the file."""
        from mukham.detector import detect_largest_face
        # detect largest face
        detect_largest_face(
            test_path + 'data/images/bryant_daughter.jpg', 
            out_path=test_path + 'data/faces/bryant_daughter.jpg'
            )
        # test if saved
        self.assertTrue(os.path.isfile(test_path + 'data/faces/bryant_daughter.jpg'))

if __name__ == "__main__":
    unittest.main()