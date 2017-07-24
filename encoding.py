import json
import os
import sys
import numpy as np
import autoencoder_runner
import json
import re
import data_utils
import encoding
from sklearn.neighbors import NearestNeighbors
import cv2

sys.path.append('../cnn_utils')
from augment_utils import augSaturation,augBlur,augNoise,augScale,augRotate,randScale

img_ext = '.jpg'


def load_encoding_database( encoding_file ):
	with open( encoding_file,'r') as ef:
		 encodings = json.load(ef)

	n_encodings = len(encodings)
	encoding_len = len(encodings.values()[0])
	print('Found {} encodings of len {}'.format(n_encodings,encoding_len))
	encodings_mat = np.ndarray( (n_encodings, encoding_len), dtype=np.float32)

	im_count = 0
	encoding_im_names = [None]*n_encodings
	for k,v in encodings.items():
		encoding_im_names[im_count] = k.encode('ascii','ignore') # unicode to regular str
		encodings_mat[im_count,:] = np.asarray(v)
	 	im_count += 1
	return encodings_mat, encoding_im_names


def find_nns_for_ims( query_im_files, model_file, augment = False, save_results=True ):
	model_name = os.path.splitext(os.path.basename( model_file ))[0]
	encodings_db, encodings_db_im_names = load_encoding_database( 'encodings_{}.json'.format(model_name) )

	n_queries = len(query_im_files )
	dataset_root = '../datasets/MTGVS'
	dataset_dirs = [ os.path.join(dataset_root, d) for d in os.listdir( dataset_root ) if os.path.isdir(os.path.join(dataset_root,d)) ]
	database_im_files = [ os.path.join(d, f) for d in dataset_dirs for f in os.listdir(d) if os.path.isfile( os.path.join(d,f) ) and f.endswith(img_ext) ]
	database_im_names = [ os.path.splitext( os.path.basename(f))[0] for f in database_im_files ]
	
	encoding_orderings = [ database_im_names.index(n) for n in encodings_db_im_names ]
	edb_im_files_ordered = [ database_im_files[i] for i in encoding_orderings ]
#	print(encodings_db_im_names[:10])
#	print(edb_im_files_ordered[:10])

	encoding_len = encodings_db.shape[1]

	X = data_utils.read_im_files( query_im_files )
	if save_results:
		h = X.shape[1]
		w = X.shape[2]		
		results_dir ='./results_nn_{}'.format(model_name) 
		if not os.path.isdir( results_dir ):
			os.mkdir( results_dir )

	if augment:
		for idx in range(X.shape[0]):
			scale = randScale(0.9,1.1)
			X[idx],_ = augRotate(X[idx], None, 15, border_color=(255,255,255))
			X[idx],_ = augScale(X[idx], None, scale, border_color=(255,255,255) )
			X[idx] = augSaturation( X[idx],0.1 )
			X[idx] = augNoise(X[idx],0.01)
			X[idx] = augBlur(X[idx])
	
	encodings_query = autoencoder_runner.predict( model_file, X )
	print(encodings_query.shape)

	match_idxs = [None] * n_queries
	dists = [None] * n_queries
	nn_names = [None] * n_queries

	nn = 5
	out_im = np.zeros( (h*nn, w*2, 3), dtype=np.float32 )

	for idx in range(encodings_query.shape[0]):
		curr_match_idxs, curr_dists = find_nns( encodings_query[idx], encodings_db, nn=nn )
		curr_nn_names = [ os.path.basename( edb_im_files_ordered[i] ) for i in curr_match_idxs[0].tolist()]
		dists[idx] = curr_dists
		nn_names[idx] = curr_nn_names
		
		if save_results:
			out_im[:h, :w, :] = np.multiply(X[idx],255)#np.multiply(np.reshape(qu,(256,256,3)),255)
			im_count = 0
			
			for ni in curr_match_idxs[0]:
				neighbor_im = cv2.imread( edb_im_files_ordered[ni] )
				labeled_im = neighbor_im.copy()
				dist_str = '{0:.4f}'.format(curr_dists[0,im_count])
				cv2.putText( labeled_im, dist_str, (5,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,255,0), 1)
				out_im[im_count*h:(im_count+1)*h, w:2*w, :] = labeled_im		
				im_count += 1

			cv2.imwrite('{}/{}_nn.jpg'.format( results_dir, os.path.splitext(os.path.basename( query_im_files[idx] ))[0]), out_im)		

	return dists, nn_names


def find_nns( encoding_query, encodings_db, nn=5 ):
	nbrs = NearestNeighbors( n_neighbors=nn ).fit(encodings_db)
	dists, idxs = nbrs.kneighbors( np.reshape( encoding_query, (1,-1)) )

	return idxs, dists

if __name__ == '__main__':
	encodings_mat, encoding_im_names =  encoding.load_encoding_database('encodings.py')
	model_file='./models/dae_epoch_{}.h5'.format(sys.argv[2])
	model = run_autoencoder.load_dae_model(model_file,'predict')

	find_nn( sys.argv[1], model, encodings_mat, encoding_im_names )
	
