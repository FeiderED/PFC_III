import copy
import logging
import sys

import cv2
import imutils
import numpy as np


class DataAugment(object):

	def __init__(self, config):
		super(DataAugment, self).__init__()

		self.config = config
		logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

	def augment(self, img_data, augment=True):
		"""It does the variations defined in file config."""

		self.__verify_data(img_data)
		img_data_aug = copy.deepcopy(img_data)

		img = cv2.imread(img_data_aug['filepath'])

		if augment and self.config.data_augmentation and np.random.randint(0, 10) <= 5:
			rows, cols = img.shape[:2]
			option = np.random.randint(0, 4)
			if(option == 0): 
				delta = np.random.randint(1, 70)
				contrast = np.random.randint(-delta, delta)
				brightness = np.random.randint(-delta, delta)
				img = np.int16(img)
				img = img * (contrast/127+1) - contrast + brightness
				img = np.clip(img, 0, 255)
				img = np.uint8(img)
			elif(option == 1): 
				correction = 0.5
				img = img / 255.0
				img = cv2.pow(img, correction)
				img = np.uint8(img * 255)
			elif(option == 2): 
				angle = 1 if(np.random.randint(0,2) == 1) else -1
				img = imutils.rotate_bound(img, angle)
			else: 
				if(np.random.randint(0,2) == 1):
					src_points = np.array([
						[0,0],
						[img.shape[1]-1, 0],
						[img.shape[1]-1, img.shape[0]-1]
					]).astype(np.float32)

					dest_src_points = np.array([
						[img.shape[1]*0.04,0],
						[img.shape[1]-1, 0],
						[img.shape[1]*0.96, img.shape[0]-1]
					]).astype(np.float32)
				else:
					src_points = np.array([
						[img.shape[1]-1, 0],
						[0, img.shape[0]-1],
						[img.shape[1]-1, img.shape[0]-1]
					]).astype(np.float32)

					dest_src_points = np.array([
						[img.shape[1]*0.96, 0],
						[img.shape[1]*0.04, img.shape[0]-1],
						[img.shape[1]-1, img.shape[0]-1]
					]).astype(np.float32)

				warp_mat = cv2.getAffineTransform(src_points, dest_src_points)
				img = cv2.warpAffine(img, warp_mat, (img.shape[1], img.shape[0]))

		img_data_aug['width'] = img.shape[1]
		img_data_aug['height'] = img.shape[0]

		return img_data_aug, img

	def __verify_data(self, img_data):
		"""Verify full data in img_data"""

		assert 'filepath' in img_data
		assert 'bboxes' in img_data
		assert 'width' in img_data
		assert 'height' in img_data
