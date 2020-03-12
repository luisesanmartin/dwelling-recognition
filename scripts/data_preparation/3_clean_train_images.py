import PIL
from PIL import Image
import numpy as np

def crop_image(image, x, y, width):
	'''
	'''
	return image[x: x + width, y: y+width]

def main():

	parser = argparse.ArgumentParser(description='Parameters')
	parser.add_argument('--city', nargs='?', type=str,
						default='acc', help='City in the data')
    parser.add_argument('--width', nargs='?', type=int,
    					default=256, help='Width of image window size')
    args = parser.parse_args()

    file = 

if __name__ == '__main__':
	main()