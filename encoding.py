import json
import os
import sys
import numpy as np
import run_autoencoder
import json
import re
import data_utils
import encoding
from sklearn.neighbors import NearestNeighbors
import cv2

sys.path.append('../cnn_utils')
from augment_utils import augSaturation,augBlur,augNoise,augScale,augRotate,randScale

def load_encoding_database( encoding_file ):
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
	return encodings_mat, encoding_im_names


def find_nn( im_file, model, encodings_mat, encoding_im_names,  augment=True ):

	_,encoding_im_files = data_utils.parse_ims_in_dir( '../datasets/MTGVS/train' )

	print(im_file)
	in_dir = os.path.dirname( im_file) 
	test_im = cv2.imread(im_file)
	print(test_im.shape)
	h = test_im.shape[0]
	w = test_im.shape[1]

	X,im_files = data_utils.parse_ims_in_dir( in_dir )
	idx = im_files.index(im_file)
	X = X[idx,:,:,:]

	if augment:
		scale = randScale(0.9,1.1)
		X,_ = augRotate(X, None, 15, border_color=(255,255,255))
		X,_ = augScale(X, None, scale, border_color=(255,255,255) )
		X = augSaturation( X,0.1 )
		X = augNoise(X,0.01)
		X = augBlur(X)
	X = np.reshape(X,(1,256,256,3))
	# run the model
	test_encoding = run_autoencoder.run_autoencoder('predict', X, None, model)

	im_num = re.search( '[0-9]*(?=.jpg)', os.path.basename(im_file) ).group(0)
	if im_num in encoding_im_names:
		idx = encoding_im_names.index(im_num)
		lookedup_encoding = np.reshape(np.asarray(encodings_mat[idx,:]),(1,1024))
		print('Distance from database encoding: {}'.format(np.linalg.norm( lookedup_encoding - test_encoding)))

	nn = 5
	nbrs = NearestNeighbors( n_neighbors=nn ).fit(encodings_mat)
	dists, idxs = nbrs.kneighbors( test_encoding )
	out_im = np.zeros( (h*nn, w*2, 3), dtype=np.float32 )

	out_im[:h, :w, :] = np.multiply(np.reshape(X,(256,256,3)),255)
	im_count = 0
	for ni in idxs[0]:
		print(encoding_im_names[ni])
		neighbor_im = cv2.imread( os.path.join('../datasets/MTGVS/train', encoding_im_names[ni] + '.jpg'))
		labeled_im = neighbor_im.copy()
		cv2.putText( labeled_im, '{0:.4f}'.format(dists[0,im_count]), (5,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,255,0), 1)
#		print(neighbor_im.shape)
#		print(labeled_im.shape)
#		cv2.imwrite('./results_nn/test.jpg', np.concatenate( [neighbor_im,labeled_im], axis=0) )
		
		out_im[im_count*h:(im_count+1)*h, w:2*w, :] = labeled_im		
		im_count += 1

	cv2.imwrite('./results_nn/{}_nn.jpg'.format(im_num), out_im)		

	nn_names = [encoding_im_names[i] for i in idxs[0].tolist()]

	return dists, nn_names
if __name__ == '__main__':
	encodings_mat, encoding_im_names =  encoding.load_encoding_database('encodings.py')
	model_file='./models/dae_epoch_{}.h5'.format(sys.argv[2])
	model = run_autoencoder.load_dae_model(model_file,'predict')

	find_nn( sys.argv[1], model, encodings_mat, encoding_im_names )
