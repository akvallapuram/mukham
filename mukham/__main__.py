import argparse
from detectors import detect_largest_face


"""
    Argument parser for command line interface
"""
my_parser = argparse.ArgumentParser(description="mukham: Crop the largest face from an image.")
my_parser.add_argument('-i', '--input', type=str, required=True, help='path to input image file')
my_parser.add_argument('-o', '--output', type=str, required=True, help='path to save output image file')
my_parser.add_argument(
    '-d', '--detector', type=str, required=False, 
    help='choice of face detection algorithm: [haar, hog, dnn]',
    default='haar'
)

# parse arguments
args =  my_parser.parse_args()

# run
detect_largest_face(args.input, args.output, args.detector)