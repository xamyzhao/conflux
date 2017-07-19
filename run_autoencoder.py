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
import argparse
import re
import cv2
def train_on_batch( model, batch_gen, logger, ex_count ):
	X_batch,Y_batch = next(batch_gen)
	loss = model.train_on_batch( [X_batch], [Y_batch] )
	logger.add_summary( tf.Summary( value=[tf.Summary.Value(tag='dae_mae', simple_value = loss),]), ex_count )
	return loss

def predict_on_batch( model, batch_gen ):
	X_batch,Y_batch = next(batch_gen)
	Y_predict = model.predict( [X_batch] ) 
	return Y_predict

def eval_on_batch( model, batch_gen, logger, ex_count ):
	X_batch,Y_batch = next(batch_gen)
	loss = model.evaluate( [X_batch], [Y_batch] )
	logger.add_summary( tf.Summary( value=[tf.Summary.Value(tag='dae_mae_test', simple_value = loss),]), ex_count )
	return loss


def run_autoencoder( mode, X_train=None, X_test=None, model_file=None ):
	print(mode)
	n_ims = X_train.shape[0]
	batch_size = min(n_ims,8)
	n_batches_per_epoch = int( math.ceil(n_ims / float(batch_size)))

	test_every_n_epochs = 5
	
	if mode=='train':
		batch_gen_train = batch_utils.gen_batch( X_train, batch_size, augment=True, randomize=True )  
		batch_gen_test = batch_utils.gen_batch( X_test, batch_size, augment=True, randomize=True )  
	elif mode=='predict':
		batch_gen = batch_utils.gen_batch( X_train, batch_size, augment = mode=='train', randomize = mode=='train' )  

	# load the model if it exists
	if model_file is not None and os.path.isfile(model_file):
		model = load_model( model_file )

		if mode=='train':
			start_epoch = int(re.search( '(?<=epoch_)[0-9]*', os.path.basename(model_file)).group(0)) + 1
		else:
			start_epoch = 0
	else:
		model = dae_model( X_train.shape[1:] )
		model.compile( optimizer='adam', lr=2e-4, loss='mean_absolute_error' )
		start_epoch = 0
	
	if mode=='predict':
		# if evaluating model on images, extract only the encoding
		model = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_8').output)

		# figure out the size of our output
		first_im_encoding = model.predict( X_train[:batch_size,:,:,:] )
		encoding_length = first_im_encoding.shape[-1]
		X_encoded = np.zeros( (n_ims,encoding_length))

	model.summary()

	if mode=='train':
		max_n_epochs = 200
	else:
		max_n_epochs = start_epoch + 1
	
	# track training and test loss using tensorboard
	tbw = tf.summary.FileWriter('./logs/')

	for epoch in range(start_epoch, max_n_epochs):
		print('Epoch {} of {}'.format(epoch, max_n_epochs))
		pb = generic_utils.Progbar( n_batches_per_epoch )
		for batch in range( n_batches_per_epoch ):
			ex_count = epoch*n_batches_per_epoch + batch  
			if mode=='train':
				X_batch_train,Y_batch_train = next( batch_gen_train )
				loss = model.train_on_batch( [X_batch_train], [Y_batch_train] )
				tbw.add_summary( tf.Summary( value=[tf.Summary.Value(tag='dae_mae', simple_value = loss),]), ex_count )

#				loss = train_on_batch( model, batch_gen_train, tbw, epoch*n_batches_per_epoch + batch )
				pb.add( 1, values=[('mae',loss)] )
			else:
				Y_out = predict_on_batch( model, batch_gen )
				print(Y_out.shape)
				batch_start_idx = epoch*n_batches_per_epoch*batch_size + batch*batch_size
				batch_end_idx = min(batch_start_idx + Y_out.shape[0],n_ims)
				batch_size_to_keep = batch_end_idx - batch_start_idx
				print(batch_size_to_keep)
				X_encoded[ batch_start_idx:batch_end_idx,:] = np.reshape( Y_out[:batch_size_to_keep,:,:,:], (batch_size_to_keep,encoding_length) )

		if mode == 'train':
			if epoch > 0 and (epoch+1) % 10 == 0:
				print('Saving model')
				model.save( './models/dae_epoch_{}.h5'.format(epoch))
			if epoch > 0 and (epoch+1) % test_every_n_epochs == 0:
				X_batch_test, Y_batch_test = next( batch_gen_test )
#				loss = eval_on_batch( model, batch_gen_test, tbw, epoch*n_batches_per_epoch + batch  )
				loss = model.evaluate( [X_batch_test], [Y_batch_test] )
				tbw.add_summary( tf.Summary( value=[tf.Summary.Value(tag='dae_mae_test', simple_value = loss),]), ex_count )
				h = X_batch_train.shape[1]
				w = X_batch_train.shape[2]
				out_im = np.zeros( (batch_size*h, 4*w, 3), dtype=np.float32)
			
				train_out = model.predict( [X_batch_train] )
				test_out = model.predict( [X_batch_test] ) 
				for i in range(batch_size):
					out_im[ i*h:(i+1)*h, :w, :] = X_batch_train[i,:,:,:]
					out_im[ i*h:(i+1)*h, w:2*w, :] = train_out[i,:,:,:]
					out_im[ i*h:(i+1)*h, 2*w:3*w, :] = X_batch_test[i,:,:,:]
					out_im[ i*h:(i+1)*h, 3*w:4*w, :] = test_out[i,:,:,:]
				cv2.imwrite('dae_output_epoch_{}.jpg'.format(epoch), np.multiply(out_im,255))
	if mode=='predict':
		print(np.max(X_encoded))
		return X_encoded

def build_database( epoch_num ):
	print('Building database json using epoch {}'.format(epoch_num))			

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('--mode', nargs='?', type=str, default='train' )
	ap.add_argument('--model_file', nargs='?', type=str, default=None )
	args = ap.parse_args()
	print(args)
	X_train,_ = data_utils.parse_ims_in_dir('../datasets/MTGVS/train')
	X_test,_ = data_utils.parse_ims_in_dir('../datasets/MTGVS/test')
	run_autoencoder( args.mode, X_train, X_test, args.model_file )
