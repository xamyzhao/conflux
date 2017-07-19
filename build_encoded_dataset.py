import os
import sys
import json
from run_autoencoder import run_autoencoder
#import data_utils
import numpy as np 
import re
import data_utils
import cv2
def build_database( epoch_num, dataset_dir ):
	model_file = './models/dae_epoch_{}.h5'.format(epoch_num)

#	im_names = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg') ]

	X,im_files = data_utils.parse_ims_in_dir( dataset_dir )

	im_names = [os.path.basename(f) for f in im_files]

	encodings = run_autoencoder( 'predict', X, None, model_file ) 
	print(encodings.shape)
	print(np.max(encodings))
	im_encodings = dict()
	im_count = 0
	for im_name in im_names:
#		print(im_name)
#		if im_name=='386603.jpg':
#			print( encodings[im_count,:].tolist())
#			cv2.imwrite('testim.jpg', np.multiply(X[im_count,:,:,:],255))
		im_num = re.search('[0-9]*(?=.jpg)',im_name).group(0)
#		im_encodings[im_num] = np.round(encodings[im_count,:],20).tolist()
		im_encodings[im_num] = encodings[im_count,:].tolist()
		im_count += 1

#	values = [{'multiverse_id':k, 'encoding':v} for k,v, in im_encodings.items()]
	with open('encodings.json','w') as f:
		f.write( json.dumps(im_encodings))

if __name__ == '__main__':
	build_database( sys.argv[1], '../datasets/MTGVS/train' )
	
