import os
import sys
import json
import autoencoder_runner
#import data_utils
import numpy as np 
import re
import data_utils
import cv2

def build_database( model_file, dataset_root ):
	model_name = os.path.splitext( os.path.basename( model_file ) )[0]
	database_name = 'encodings_{}.json'.format( model_name )
	dataset_dirs = [ os.path.join(dataset_root,d) for d in os.listdir(dataset_root) if os.path.isdir( os.path.join(dataset_root, d) )]

	X = None
	im_names = []
	for d in dataset_dirs:
		X_curr,im_files = data_utils.parse_ims_in_dir( d )
		print('Collecting {} images from {} for database'.format(len(im_files),d))
		im_names += [os.path.basename(f) for f in im_files]

		if X is None:
			X = X_curr
		else:
			X = np.append(X, X_curr, axis=0)

	encodings = autoencoder_runner.predict( model_file, X ) 
	im_encodings = dict()
	im_count = 0
	for im_name in im_names:
		im_num = os.path.splitext(im_name)[0]
		im_encodings[im_num] = encodings[im_count,:].tolist()
		im_count += 1

	with open(database_name,'w') as f:
		f.write( json.dumps(im_encodings))

if __name__ == '__main__':
	build_database( sys.argv[1], '../datasets/MTGVS' )
	
