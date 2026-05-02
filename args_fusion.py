class args():

	# training args
	epochs = 50 #"number of training epochs, ct-mri 85,pet-mri 80,spect-mri 50"
	batch_size = 2 #"batch size for training, default is 2"
	dataset_a = "...data/images"
	dataset_b = "...data/images"

	dataset = 'medical'

	HEIGHT = 256
	WIDTH = 256

	save_fusion_model = "models/train/fusionnet/"
	save_loss_dir = '...models/train/loss_fusionnet/'

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"

	lr = 1e-4 #"learning rate, default is 0.001"
	log_interval = 10 #"number of images after which the training loss is logged, default is 10"

	resume_nestfuse = '...models/nestfuse/edcoder.model'

	mode = "MBAfusion"



