import os
import sys
import numpy as np
from keras.callbacks import TensorBoard
from keras.models import load_model, Model
from keras.utils import generic_utils
from denoising_autoencoder_network import dae_model
import math
os.environ['CUDA_VISIBLE_DEVICES']='0'
import data_utils
import batch_utils
import tensorflow as tf


def run_autoencoder( mode, X, model_file=None ):
#	if in_dir is None:
#		in_dir = '../datasets/MTGVS/train'

#	X = data_utils.parse_ims_in_dir( in_dir )
	n_ims = X.shape[0]

	batch_size = min(n_ims,8)
	n_batches_per_epoch = int( math.ceil(n_ims / float(batch_size)))

	if mode=='train':
		model = dae_model( X.shape[1:] )
		model.compile( optimizer='adam', loss='mean_absolute_error' )
	else:
		model = load_model( model_file )
		model = Model(inputs=model.input, outputs=model.get_layer('conv2d_8').output)
		test_encoding = model.predict( X[:batch_size,:,:,:] )
		encoding_length = test_encoding.shape[-1]
		X_encoded = np.zeros( (n_ims,encoding_length))
	model.summary()

	
	batch_gen = batch_utils.gen_batch( X, batch_size, augment = mode=='train', randomize = mode=='train' )  

	start_epoch = 0
	if mode=='train':
		max_n_epochs = 20
	else:
		max_n_epochs = 1

#	n_batches_per_epoch = 1	
	tbw = tf.summary.FileWriter('./logs/')
	for epoch in range(start_epoch, max_n_epochs):
		print('Epoch {} of {}'.format(epoch, max_n_epochs-start_epoch))
		pb = generic_utils.Progbar( n_batches_per_epoch )
		for batch in range( n_batches_per_epoch ):
			X_batch,Y_batch = next(batch_gen)

			if mode=='train':
				loss = model.train_on_batch( [X_batch], [Y_batch] )
				pb.add( 1, values=[('mae',loss)] )
				tbw.add_summary( tf.Summary( value=[tf.Summary.Value(tag='dae_mae', simple_value = loss),]), epoch*n_batches_per_epoch + batch )
			else:
				Y_out = model.predict( [X_batch] )
				batch_start_idx = epoch*n_batches_per_epoch*batch_size + batch*batch_size
				batch_end_idx = min(batch_start_idx + Y_out.shape[0],n_ims)
				#print('Filling results {}:{}'.format(batch_start_idx, batch_end_idx))
				batch_size_to_keep = batch_end_idx - batch_start_idx
				X_encoded[ batch_start_idx:batch_end_idx,:] = np.reshape( Y_out[:batch_size_to_keep,:,:,:], (batch_size_to_keep,encoding_length) )
		if epoch % 5 == 0 and mode=='train':
			print('Saving model')
			model.save( './models/dae_epoch_{}.h5'.format(epoch))

	if mode=='test':
		return X_encoded

def build_database( epoch_num ):
	print('Building database json using epoch {}'.format(epoch_num))			

if __name__ == '__main__':
	X,_ = data_utils.parse_ims_in_dir('../datasets/MTGVS/train')
	run_autoencoder( sys.argv[1], X )
