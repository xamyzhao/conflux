import os
import sys
import cv2
import numpy as np

def parse_ims_in_dir( dataset_dir, randomize = False, load_n = None ):
	im_files =  [ os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

	if load_n is not None:
		im_files = im_files[:load_n]

	X = read_im_files( im_files )
	return X, im_files

def read_im_files( im_files ):
	print('Reading {} files'.format(len(im_files)))
	im = cv2.imread( im_files[0] )
	X = np.ndarray( (len(im_files),) + im.shape, dtype=np.float32 )

	im_count = 0
	for f in im_files:
		X[im_count,:,:,:] = np.multiply( cv2.imread( im_files[im_count] ),1/255.0)
		im_count += 1
	return X

def load_train_validation_test_sets( train_dir, test_dir, val_dir = None, random_seed=17, percent_val=0.1, load_n = None ):
	X_train,_   = parse_ims_in_dir( train_dir, load_n=load_n )

	if test_dir is not None:
		X_test,_    = parse_ims_in_dir( test_dir, load_n=load_n )
	else:
		X_test = None
	
	# if specified, load validation set from dir
	if val_dir is not None:
		X_val,_   = parse_ims_in_dir( val_dir )
	else: # otherwise, split training set into training and validation sets
		np.random.seed( random_seed )
		n_train = X_train.shape[0]
		train_idxs = np.random.permutation( n_train ).tolist()
		
#		X_train = X_train[train_idxs]
		n_val = int(percent_val * n_train)
		X_val = X_train[ train_idxs[:n_val] ]
		X_train = X_train[ train_idxs[n_val:]]
	return X_train, X_val, X_test
