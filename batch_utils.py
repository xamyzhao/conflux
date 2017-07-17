import numpy as np
import sys

sys.path.append('../cnn_utils')
from augment_utils import augSaturation,augBlur,augNoise,augScale,augRotate,randScale
import cv2

def gen_batch( X, batch_size, augment=False, write_examples = False, randomize=False ):
	np.random.seed(17)
	print('seeding')
	idx = 0
	while True:
		X_out_all = np.zeros( (batch_size,) + X.shape[1:] )
		Y_all = np.zeros( (batch_size,) + X.shape[1:] )

		for i in range(batch_size):
			print(idx)
			if randomize:
				idx = int(np.random.rand(1) * X.shape[0])
			
			X_out = X[idx,:,:,:]
			Y = X_out.copy().astype(np.float32)

			if augment:
				scale = randScale(0.8,1.2)
				Y,_ = augRotate(Y, None, 15, border_color=(255,255,255))
				Y,_ = augScale(Y, None, scale, border_color=(255,255,255) )
				Y = augSaturation( Y,0.1 )
				Y = augNoise(Y,0.02)
		
				Y = augBlur(Y)
	#			print('after blur, {} {}'.format(np.max(Y),np.min(Y)))
	#			print('after noise, {} {}'.format(np.max(Y),np.min(Y)))
			if not randomize:
				idx += 1
				if idx >= X.shape[0]:
					idx = 0
			X_out_all[i,:,:,:] = X_out
			Y_all[i,:,:,:] = Y
			

		if write_examples:
			cv2.imwrite( './examples_train/train_ex_{}.jpg'.format(idx), np.multiply(np.concatenate([X_out,Y],axis=1),255.0)) 
		yield X_out_all, Y_all

