import PIL
from PIL import Image
import numpy as np

def segment_image(image, width):
    '''
    '''
    size_x, size_y, _ = image.shape
    x = 0
    y = 0
    rv = []
    
    while x + width <= size_x:
        
        while y + width <= size_y:
            
            sub_image = image[x:x+width, y:y+width]
            rv.append(sub_image)
            y += width
    
        y = 0
        x += width
    
    return rv

def main():

    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--city', nargs='?', type=str,
                        default='kam', help='City in the data')
    parser.add_argument('--width', nargs='?', type=int,
                        default=256, help='Width of image window size')
    args = parser.parse_args()

    file = 

if __name__ == '__main__':
    main()