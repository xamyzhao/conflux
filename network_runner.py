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
			if 'encoding' in l.name:
				encoding_layer_name = l.name
				break
		model = Model(inputs=model.input, outputs=model.get_layer(encoding_layer_name).output)
	return model

	
def make_batch_generators(mode, X_train = None, X_test = None, batch_size=8, siamese = False ):
	batch_gen_test = None

	if not siamese:
		batch_gen_train = batch_utils.gen_batch( X_train, batch_size, augment=mode=='train', randomize=mode=='train' )  
	
		if X_test is not None:
			batch_gen_test = batch_utils.gen_batch( X_test, batch_size, augment=mode=='train', randomize = False )  
	
	else:
		batch_gen_train = batch_utils.gen_siamese_batch( X_train, batch_size, augment=mode=='train', write_examples = False, randomize = mode=='train' )
		if X_test is not None:
			batch_gen_test = batch_utils.gen_siamese_batch( X_test, batch_size, augment=mode=='train', write_examples = False, randomize = False )
	return batch_gen_train, batch_gen_test 


def run_network( 	mode, 
									batch_gen_train=None, batch_gen_val=None, 
									n_ims_train = 0, n_ims_test=0, batched_input_size=(8,256,256,3), 
									model=None, start_epoch=0, end_epoch=1, 
									siamese=False,
									test_every_n_batches = 500,
									save_every_n_epochs = 1 ):

	batch_size = batched_input_size[0]
	nb_per_epoch_train = int( math.ceil( n_ims_train / float(batch_size)))
	nb_per_epoch_test = int( math.ceil( n_ims_test / float(batch_size)))
	
	if siamese:
		model_name = model[1].name
		models = model[:]
		model = models[1]
	else:
		model_name = model.name
		models = None
	print('Running model {} in mode {} on {} training ims, {} validation ims'.format(model_name, mode, n_ims_train, n_ims_test))

	test_every_n_batches = 200
	save_every_n_epochs = 1

	if mode == 'predict':	
		# figure out the size of our output
		encoding_length = model.layers[-1].output_shape[-1]
		X_encoded = np.zeros( (n_ims_train,encoding_length))

	# track training and test loss using tensorboard
	tbw = tf.summary.FileWriter('./logs/' + model_name)

	for epoch in range(start_epoch, end_epoch):
		print('Epoch {} of {}'.format(epoch, end_epoch))
		pb = generic_utils.Progbar( nb_per_epoch_train )

		for bi in range( nb_per_epoch_train ):
			total_batch_count = epoch*nb_per_epoch_train + bi
			if mode=='train':
				X_batch_train,Y_batch_train = next( batch_gen_train )
				loss = model.train_on_batch( X_batch_train, Y_batch_train )

				# record loss in tensorboard, also display it in the progress bar
				tbw.add_summary( tf.Summary( value=[tf.Summary.Value(tag='{}_train'.format(model_name), simple_value = loss),]), total_batch_count )
				pb.add( 1, values=[('train loss',loss)] )
				print(total_batch_count)

				if epoch > 0 and total_batch_count % test_every_n_batches == 0 or bi == nb_per_epoch_train - 1:
					print('Running on validation set...')
					val_loss = 0
					for tbi in range( nb_per_epoch_test ):
						X_batch_val, Y_batch_val = next( batch_gen_val )
						loss = model.evaluate( X_batch_val, Y_batch_val )
						val_loss += loss

					tbw.add_summary( tf.Summary( value=[tf.Summary.Value(tag='{}_validation'.format(model_name), simple_value = float(val_loss)/nb_per_epoch_test ),]), total_batch_count )

					# save output images only for the last test batch
					train_out = model.predict( X_batch_train )
					val_out = model.predict( X_batch_val ) 

					if not siamese:
						out_im 	= batch_utils.compile_ae_results( np.concatenate( [X_batch_train, X_batch_val] ), 
																									np.concatenate( [train_out, val_out] ), 
																									np.concatenate( [Y_batch_train, Y_batch_val] )) 
					else:
						out_im 	= batch_utils.compile_siamese_results( 	np.concatenate( [X_batch_train, X_batch_val] ), 
																											np.concatenate( [train_out, val_out] ), 
																											np.concatenate( [Y_batch_train, Y_batch_val] )) 
					cv2.imwrite('{}_output_epoch_{}.jpg'.format(model_name,epoch), out_im)

			else:
				X_batch, Y_batch = next( batch_gen_train )
				Y_out = model.predict( X_batch )

#				batch_start_idx = epoch*nb_per_epoch_train*batch_size + batch*batch_size
				batch_start_idx = total_batch_count*batch_size
				batch_end_idx = min(batch_start_idx + Y_out.shape[0],n_ims_train)
				batch_size_to_keep = batch_end_idx - batch_start_idx

				X_encoded[ batch_start_idx:batch_end_idx,:] = np.reshape( Y_out[:batch_size_to_keep], (batch_size_to_keep,encoding_length) )

		if mode == 'train' and epoch > 0 and epoch % save_every_n_epochs == 0:
			print('Saving model')
			if models is not None:
				for m in models:
					m.save('./models/{}_epoch_{}.h5'.format(m.name, epoch))
			else:
				print('Saving model')
				model.save( './models/{}_epoch_{}.h5'.format(model_name,epoch))

	if mode=='predict':
		return X_encoded


def train( model_file, X_train, X_val, max_n_epochs = 300 ):
	n_ims = X_train.shape[0]
	batch_size = min(X_train.shape[0],8)

	if os.path.isfile(model_file): # continue training
		model = load_encoder_model(model_file)
		start_epoch = int(re.search( '(?<=epoch_)[0-9]*', os.path.basename(model_file)).group(0)) + 1
	else:	# start a new model
		model = networks.make_model( model_file )
		start_epoch = 0

	batch_gen_train, batch_gen_val = make_batch_generators( 'train', X_train, X_val, batch_size, siamese=('siamese' in model_file) )

	run_network( 'train', batch_gen_train, batch_gen_val, X_train.shape[0], X_val.shape[0], (batch_size,)+X_train.shape[1:], model, start_epoch, max_n_epochs, siamese='siamese' in model_file)


def predict( model_file, X ):
	n_ims = X.shape[0]
	batch_size = min(X.shape[0],8)
	model = load_encoder_model(model_file, cutoff_at_encoding_layer = True)
	start_epoch = 0
	end_epoch = 1

	batch_gen_train,_  = make_batch_generators( 'predict', X,None, batch_size, siamese=('siamese' in model_file) )

	return run_network( 'predict', batch_gen_train, None, n_ims, 0, (batch_size,)+X.shape[1:], model, start_epoch, end_epoch, siamese='siamese' in model_file )
		

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('--mode', nargs='?', type=str, default='train' )
	ap.add_argument('--model_file', nargs='?', type=str, default=None )
	ap.add_argument('--load_fewer', action='store_true', help='Load some of the training set' )
	ap.add_argument('--max_n_epochs', type=int, default=200 )
#	ap.add_argument('--model_name', nargs='?', type=str, help='Name of model architecture to train')
	args = ap.parse_args()
	print(args)

	dataset_root = '../datasets/MTGVS/'
	if args.mode=='train':
		if args.load_fewer:
			X_train, X_val, _ = data_utils.load_train_validation_test_sets( dataset_root + 'train', dataset_root + 'test', load_n = 100 )
		else:
			X_train, X_val, _ = data_utils.load_train_validation_test_sets( dataset_root + 'train', dataset_root + 'test' )
		print('Training model {} on {} training, {} validation examples'.format(args.model_file, X_train.shape[0], X_val.shape[0] ))
		train( args.model_file, X_train, X_val, max_n_epochs = args.max_n_epochs )
	else:
		print('Unable to directly run a mode other than train')
