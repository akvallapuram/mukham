import argparse
import os
from mukham.detector import detect_largest_face


"""
    Argument parser for command line interface
"""
my_parser = argparse.ArgumentParser(description="mukham: Crop the largest face from an image.")
my_parser.add_argument('-i', '--input', type=str, required=True, help='path to input image file')
my_parser.add_argument('-o', '--output', type=str, required=True, help='path to save output image file')
my_parser.add_argument(
    '-c', '--conf', type=float, required=False, 
    help='confidence threshold for face detected',
    default=0.8
    )

# parse arguments
args =  my_parser.parse_args()

# check input file
if not os.path.isfile(args.input):
    raise FileNotFoundError(args.input + ' not found!')

# run
detect_largest_face(args.input, out_path=args.output, min_conf=args.conf)