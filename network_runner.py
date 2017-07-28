import os
import sys
import numpy as np
from keras.callbacks import TensorBoard
from keras.models import load_model, Model
from keras.utils import generic_utils
#from denoising_autoencoder_network import dae_model
import networks
import math
os.environ['CUDA_VISIBLE_DEVICES']='0'
import data_utils
import batch_utils
import tensorflow as tf
import argparse
import re
import cv2


def load_encoder_model(model_file, cutoff_at_encoding_layer = False):
	model = None
	if model_file is not None and os.path.isfile(model_file):
		model = load_model( model_file )
	model.summary()
	if cutoff_at_encoding_layer:
		# if evaluating model on images, extract only the encoding
		for l in model.layers:
			if 'encoding' in l.name or 'dense' in l.name:
				encoding_layer_name = l.name
				break
		model = Model(inputs=model.input, outputs=model.get_layer(encoding_layer_name).output)
	return model
	
def make_batch_generators(mode, X_train = None, X_test = None, batch_size=8, siamese = False ):
	batch_gen_test = None
	if not siamese:
		batch_gen_train = batch_utils.gen_batch( X_train, batch_size, augment=mode=='train', randomize=mode=='train' )  
		if X_test is not None:
			batch_gen_test = batch_utils.gen_batch( X_test, batch_size, augment=mode=='train', randomize=mode=='train' )  
	else:
		batch_gen_train = batch_utils.gen_siamese_batch( X_train, batch_size, augment=mode=='train', write_examples = False, randomize = mode=='train' )
		if X_test is not None:
			batch_gen_test = batch_utils.gen_siamese_batch( X_test, batch_size, augment=mode=='train', write_examples = False, randomize = mode=='train' )
	return batch_gen_train, batch_gen_test 


def run_autoencoder( mode, batch_gen_train=None, batch_gen_test=None, n_ims = 0, batched_input_size=(8,256,256,3), model=None, start_epoch=0, end_epoch=1, siamese=False ):
	batch_size = batched_input_size[0]
	n_batches_per_epoch = int( math.ceil(n_ims / float(batch_size)))
	
	if siamese:
		model_name = model[1].name
		models = model[:]
		model = models[1]
	else:
		model_name = model.name
		models = None
	print('Running model {} in mode {} on {} ims'.format(model_name, mode, n_ims))

	test_every_n_epochs = 5

	if mode == 'predict':	
		# figure out the size of our output
		encoding_length = model.layers[-1].output_shape[-1]
		X_encoded = np.zeros( (n_ims,encoding_length))

	# track training and test loss using tensorboard
	tbw = tf.summary.FileWriter('./logs/' + model_name)

	for epoch in range(start_epoch, end_epoch):
		print('Epoch {} of {}'.format(epoch, end_epoch))
		pb = generic_utils.Progbar( n_batches_per_epoch )

		for batch in range( n_batches_per_epoch ):
			ex_count = epoch*n_batches_per_epoch + batch

			if mode=='train':
				X_batch_train,Y_batch_train = next( batch_gen_train )
				loss = model.train_on_batch( X_batch_train, Y_batch_train )

				# record loss in tensorboard, also display it in the progress bar
				tbw.add_summary( tf.Summary( value=[tf.Summary.Value(tag='{}_train'.format(model_name), simple_value = loss),]), ex_count )
				pb.add( 1, values=[('mae',loss)] )
			else:
				X_batch, Y_batch = next( batch_gen_train )
				Y_out = model.predict( X_batch )
				batch_start_idx = epoch*n_batches_per_epoch*batch_size + batch*batch_size
				batch_end_idx = min(batch_start_idx + Y_out.shape[0],n_ims)
				batch_size_to_keep = batch_end_idx - batch_start_idx
				X_encoded[ batch_start_idx:batch_end_idx,:] = np.reshape( Y_out[:batch_size_to_keep], (batch_size_to_keep,encoding_length) )

		if mode == 'train':	# evaluate on test set
			if epoch > 0 and epoch % 50 == 0:
				if models is not None:
					for m in models:
						m.save('./models/{}_epoch_{}.h5'.format(m.name, epoch))
				else:
					print('Saving model')
					model.save( './models/{}_epoch_{}.h5'.format(model_name,epoch))
			if epoch > 0 and epoch % test_every_n_epochs == 0:
				X_batch_test, Y_batch_test = next( batch_gen_test )
				loss = model.evaluate( X_batch_test, Y_batch_test )

				train_out = model.predict( X_batch_train )
				test_out = model.predict( X_batch_test ) 

				tbw.add_summary( tf.Summary( value=[tf.Summary.Value(tag='{}_test'.format(model_name), simple_value = loss),]), ex_count )

				if not siamese:
					out_im 	= batch_utils.compile_ae_results( np.concatenate( [X_batch_train, X_batch_test] ), np.concatenate( [train_out, test_out] ), np.concatenate( [Y_batch_train, Y_batch_test] )) 
				else:
					out_im 	= batch_utils.compile_siamese_results( np.concatenate( [X_batch_train, X_batch_test] ), np.concatenate( [train_out, test_out] ), np.concatenate( [Y_batch_train, Y_batch_test] )) 
				cv2.imwrite('{}_output_epoch_{}.jpg'.format(model_name,epoch), out_im)
	if mode=='predict':
		return X_encoded

def train( model_file ):
	X_train,_ = data_utils.parse_ims_in_dir('../datasets/MTGVS/train')
	X_test,_ = data_utils.parse_ims_in_dir('../datasets/MTGVS/test')
	n_ims = X_train.shape[0]
	batch_size = min(X_train.shape[0],8)

	max_n_epochs = 300 + 1

	if model_file is not None and os.path.isfile(model_file):
		model = load_encoder_model(model_file)
		start_epoch = int(re.search( '(?<=epoch_)[0-9]*', os.path.basename(model_file)).group(0)) + 1
	else:
		model = networks.make_model( model_file )
		start_epoch = 0
	batch_gen_train, batch_gen_test = make_batch_generators( 'train', X_train, X_test, batch_size, siamese=('siamese' in model_file) )
	run_autoencoder( 'train', batch_gen_train, batch_gen_test, n_ims, (batch_size,)+X_train.shape[1:], model, start_epoch, max_n_epochs, siamese='siamese' in model_file)


def predict( model_file, X ):
	n_ims = X.shape[0]
	batch_size = min(X.shape[0],8)
	model = load_encoder_model(model_file, cutoff_at_encoding_layer = True)
	start_epoch = 0
	end_epoch = 1

	batch_gen_train, batch_gen_test = make_batch_generators( 'predict', X, None, batch_size, siamese=('siamese' in model_file) )
	return run_autoencoder( 'predict', batch_gen_train, None, n_ims, (batch_size,)+X.shape[1:], model, start_epoch, end_epoch, siamese='siamese' in model_file )
		

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('--mode', nargs='?', type=str, default='train' )
	ap.add_argument('--model_file', nargs='?', type=str, default=None )
	ap.add_argument('--model_name', nargs='?', type=str, help='Name of model architecture to train')
	args = ap.parse_args()
	print(args)

	if args.mode=='train':
		train( args.model_file )
	else:
		print('Unable to directly run a mode other than train')
