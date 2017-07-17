import os
import sys
import cv2
import numpy as np

def parse_ims_in_dir( dataset_dir, randomize = False ):
	im_files =  [ os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

	im = cv2.imread( im_files[0] )
	X = np.ndarray( (len(im_files),) + im.shape, dtype=np.float32 )

	im_count = 0
	for f in im_files:
		X[im_count,:,:,:] = np.multiply( cv2.imread( im_files[im_count] ),1/255.0)
		im_count += 1
	return X, im_files
		
