import os
import sys
import numpy as np
from run_autoencoder import run_autoencoder
import json
import re
import data_utils
from sklearn.neighbors import NearestNeighbors
import cv2

sys.path.append('../cnn_utils')
from augment_utils import augSaturation,augBlur,augNoise,augScale,augRotate,randScale


def show_nn( im_file, model_iters ):

	with open('encodings.json','r') as ef:
		encodings = json.load(ef)

	n_encodings = len(encodings)
	encoding_len = len(encodings.values()[0])
	print('Found {} encodings of len {}'.format(n_encodings,encoding_len))
	encodings_mat = np.ndarray( (n_encodings, encoding_len), dtype=np.float32)

	im_count = 0
	encoding_im_names = [None]*n_encodings
	for k,v in encodings.items():
		encoding_im_names[im_count] = k
		encodings_mat[im_count,:] = np.asarray(v)
		im_count += 1

	_,encoding_im_files = data_utils.parse_ims_in_dir( '../datasets/MTGVS/train' )

	in_dir = os.path.dirname( im_file) 
	test_im = cv2.imread(im_file)
	h = test_im.shape[0]
	w = test_im.shape[1]

	X,im_files = data_utils.parse_ims_in_dir( in_dir )
	idx = im_files.index(im_file)
	X = X[idx,:,:,:]

	print(X.shape)
	scale = randScale(0.9,1.1)
	print(scale)
	X,_ = augRotate(X, None, 15, border_color=(255,255,255))
	X,_ = augScale(X, None, scale, border_color=(255,255,255) )
	X = augSaturation( X,0.1 )
	X = augNoise(X,0.01)
	X = augBlur(X)
	X = np.reshape(X,(1,256,256,3))
#	X = np.reshape(np.multiply(X.astype(np.float32),1/255.0), (1,256,256,3))
	test_encoding = run_autoencoder('predict', X, None,  model_file='./models/dae_epoch_{}.h5'.format(model_iters))
	#print(test_encoding)

	im_num = re.search( '[0-9]*(?=.jpg)', os.path.basename(im_file) ).group(0)
	print(im_num)
	lookedup_encoding = np.reshape(np.asarray(encodings[im_num]),(1,1024))
	print(np.linalg.norm( lookedup_encoding - test_encoding))

	nbrs = NearestNeighbors( n_neighbors=5 ).fit(encodings_mat)
	print(nbrs)
	dists, idxs = nbrs.kneighbors( test_encoding )
	print(dists)
	out_im = np.zeros( (h*5, w*2, 3), dtype=np.float32 )

	out_im[:h, :w, :] = np.multiply(np.reshape(X,(256,256,3)),255)
	im_count = 0
	for ni in idxs[0]:
		print(ni)
		neighbor_im = cv2.imread( os.path.join('../datasets/MTGVS/train', encoding_im_names[ni] + '.jpg'))

		out_im[im_count*h:(im_count+1)*h, w:2*w, :] = neighbor_im
		im_count += 1

	cv2.imwrite('nn.jpg', out_im)		
#	neighbor_im_files = im_files[idxs[0]]
#	print(neighbor_im_files)

if __name__ == '__main__':
	show_nn( sys.argv[1], sys.argv[2] )
