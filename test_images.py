import os
from traceback import format_exc


if __name__ == "__main__":
    for f in os.listdir('data/images'):
        try:
            detect_largest_face('data/images/' + f, 'data/faces/' + f)
        
        except:
            print(format_exc())