import numpy as np
import sys

sys.path.append('../cnn_utils')
from augment_utils import augSaturation,augBlur,augNoise,augScale,augRotate,randScale
import cv2

def aug_im( Y ):
	scale = randScale(0.8,1.2)
	Y,_ = augRotate(Y, None, 10, border_color=(255,255,255))
	Y,_ = augScale(Y, None, scale, border_color=(255,255,255) )
	Y = augSaturation( Y,0.1 )
	Y = augNoise(Y,0.02)
		
	Y = augBlur(Y)
	return Y

def gen_batch( X, batch_size, augment=False, write_examples = False, randomize=False ):
	np.random.seed(17)
	idx = -1

	while True:
		X_out_all = np.zeros( (batch_size,) + X.shape[1:] )
		Y_all = np.zeros( (batch_size,) + X.shape[1:] )

			
		for i in range(batch_size):
			if randomize:
				idx = int(np.floor(np.random.rand(1) * X.shape[0]))
			else:
				idx += 1
				if idx >= X.shape[0]:
					print('Resetting generator index to 0')
					idx = 0
					np.random.seed(17) # make sure all the same augmentations happen				
	
			X_out = X[idx,:,:,:]
			Y = X_out.copy().astype(np.float32)

			if augment:
				X_out = aug_im( X_out )
		
			X_out_all[i,:,:,:] = X_out.copy()
			Y_all[i,:,:,:] = Y.copy()

		if write_examples:
			cv2.imwrite( './examples_train/train_ex_{}.jpg'.format(idx), compile_ae_results(X_out_all,None,Y_all)) 
		
		yield X_out_all, Y_all


def gen_siamese_batch( X, batch_size, augment=True, write_examples = False, randomize = False ):
	batch_count = 0
	batch_gen = gen_batch(X, batch_size, augment=augment, write_examples=False, randomize = randomize)

	while True:
		X_A, X_B = next( batch_gen )
		y = np.ones( (batch_size,1), dtype=np.float32 )
		split_idx = int(batch_size/2)
		# split batch so that half of it is unmatched
		X_A[split_idx:] = X_A[-1:split_idx-1:-1]
		y[split_idx:] = np.zeros( (batch_size-split_idx,1), dtype=np.float32 )

		if write_examples and batch_count % 50 == 0:
			out_im = compile_siamese_results( [X_A, X_B], None, y )
			cv2.imwrite( './examples_train/siamese_ex_{}.jpg'.format(batch_count), out_im )

		batch_count += 1

		yield [X_A, X_B], y

def compile_ae_results( X, X_predict, Y):
	if X_predict is not None:
		return np.multiply(np.concatenate([X,X_predict,Y],axis=1),255.0)
	else:
		return np.multiply(np.concatenate([X,Y],axis=1),255.0)

def compile_siamese_results( X, X_predict, Y ):
	X_A = X[0]
	X_B = X[1]

	batch_size = X_A.shape[0]
	h = X_A.shape[1]
	w = X_A.shape[2]

	out_im = np.zeros( (batch_size*h, w*2, 3), dtype=np.float32)

	for i in range(batch_size):
		out_im[ i*h:(i+1)*h, :w, :] = np.multiply(X_A[i],255)
		out_im[ i*h:(i+1)*h, w:2*w, :] = np.multiply(X_B[i],255)

		if Y[i]<1:
			label_color = (0,0,255)
			label_text = 'False'
		else:
			label_color = (0,255,0)
			label_text = 'True'
		if X_predict is not None:
			predict_text = X_predict[i]
		else:
			predict_text = ''

		cv2.putText( out_im , '{}, {}'.format(label_text, predict_text), (10,i*h+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, label_color, 1 )

	return out_im

