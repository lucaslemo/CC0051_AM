import sys
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
import pickle
import os
import matplotlib.pyplot as plt
import time
# import cv2


def load_images():
	dir = os.getcwd()
	path = os.path.join(dir, 'dataset')
	for card in os.listdir(path):
		image_path = os.path.join(path, card)
		image = imread(image_path)
		print(image)
		break


def main():
	load_images()


if __name__ == '__main__':
	main()
