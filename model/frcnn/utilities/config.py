import math


class Config:

	def __init__(self):
		self.verbose = True
		self.data_augmentation = False
		self.num_epochs = 5
		self.epoch_length = 32
		self.learning_rate = 0.00001
		self.use_gpu = False
		self.anchor_box_scales = [128, 256, 512]
		self.anchor_box_ratios = [
			[1, 1],
			[1./math.sqrt(2), 2./math.sqrt(2)],
			[2./math.sqrt(2), 1./math.sqrt(2)]
		]
		self.im_size = 600
		self.img_channel_mean = [103.939, 116.779, 123.68]
		self.img_scaling_factor = 1.0
		self.num_rois = 32
		self.rpn_stride = 16
		self.balanced_classes = False
		self.std_scaling = 4.0
		self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
		self.rpn_min_overlap = 0.3
		self.rpn_max_overlap = 0.7
		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5
		self.config_file_path = "config.pickle"
		self.class_mapping = None
		self.weights_output_path = "model_frcnn.hdf5"
		self.weights_input_path = "model_frcnn.hdf5"
