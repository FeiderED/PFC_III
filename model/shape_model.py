
import json
import os

from report import Report
from train import Trainer


class ShapeModel(object):


	def __init__(self, dataset_path, num_rois=32, weights_input_path="none"):

		super(ShapeModel, self).__init__()

		self.dataset_path = dataset_path
		self.num_rois = num_rois
		self.weights_input_path = weights_input_path

	def __generate_results_path(self, base):

		ans = base + "_results"
		folder = os.listdir(ans)
		num_results = len(folder)

		name = ans + "/" + str(num_results)
		while(os.path.isdir(name)):
			num_results += 1
			name = ans + "/" + str(num_results)

		return name

	def train(
			self,
			data_augmentation,
			num_epochs=5,
			epoch_length=32,
			learning_rate=1e-5,
			num_rois=32,
			use_gpu=False,
	):

		base_path = self.__generate_results_path("training")
		annotate_path = base_path + "/annotate.txt"
		weights_output_path = base_path + "/flowchart_3b_model.hdf5"
		config_output_filename = base_path + "/config.pickle"
		os.mkdir(base_path)
		trainer = Trainer(base_path, use_gpu)
		trainer.recover_data(
			self.dataset_path,
			annotate_path,
			generate_annotate=True
		)
		trainer.configure(
			data_augmentation,
			self.num_rois,
			weights_output_path,
			self.weights_input_path,
			num_epochs=num_epochs,
			epoch_length=epoch_length,
			learning_rate=learning_rate,
		)
		trainer.save_config(config_output_filename)
		trainer.train()

	def generate_classification_report(
		self,
		results_path,
		generate_annotate=False,
		use_gpu=False
		):


		report = Report(
			results_path=results_path,
			dataset_path=self.dataset_path,
			generate_annotate=generate_annotate,
			use_gpu=use_gpu
		)
		report.generate()


def get_options():
	try:
		with open('args.json', 'r') as f:
			options_dict = json.load(f)
	except Exception as e:
		print("Options file (JSON) don't found!")
		exit()

	return options_dict


if __name__ == '__main__':

	options_dict = get_options()
	print(options_dict)

	if(options_dict['rois'] == None):
		options_dict['rois'] = 32
	if(options_dict['input_weight_path'] == None):
		options_dict['input_weight_path'] = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
	if(options_dict['epochs'] == None):
		options_dict['epochs'] = 5
	if(options_dict['learning_rate'] == None):
		options_dict['learning_rate'] = 1e-5

	shape_model = ShapeModel(
		dataset_path=options_dict['dataset'],
		num_rois=options_dict['rois'],
		weights_input_path=options_dict['input_weight_path']

	)
	shape_model.train(
	    data_augmentation=True,
	    num_epochs=options_dict['epochs'],
		learning_rate=options_dict['learning_rate'],
		use_gpu=options_dict['gpu']
	)

