import os
import sys

import numpy as np
import re
import encoding

dataset_root = '../datasets/MTGVS'

def eval_accuracy( model_file ):
	for d in os.listdir(dataset_root):
		dataset_dir = os.path.join( dataset_root, d)
		if not os.path.isdir(dataset_dir):
			continue
		
		im_files = [ os.path.join(dataset_dir,f) for f in os.listdir(dataset_dir) if f.endswith('.jpg') ]
		n_ims = len(im_files )

		n_eval = min(100, n_ims)
		n_correct = 0


		im_files_to_eval = [ im_files[i] for i in np.linspace(0, n_ims-1, n_eval, dtype=int) ]
		dists, match_names = encoding.find_nns_for_ims( im_files_to_eval, model_file, augment=True )

		for idx in range( n_eval ):
			if os.path.basename( im_files_to_eval[idx] ) in match_names[idx]:
				n_correct += 1

		#encoding_mat, encoding_im_names = encoding.load_encoding_database('encodings.py')
		#model_file = './models/dae_epoch_{}.h5'.format(199)
		#model = run_autoencoder.load_dae_model(model_file,'predict')

		#for i in np.linspace(0,n_ims-1, n_eval,dtype=int):
	#		im_num = re.search('[0-9]*(?=.jpg)', os.path.basename(ims_train[i])).group(0) 
	#		dists, matches = nn_by_encoding.find_nn( ims_train[i], model, encoding_mat, encoding_im_names )
	#		if im_num in matches:
	#			n_correct += 1

		print('% correct: {}'.format( float(n_correct)/n_eval))


if __name__ == '__main__':
	eval_accuracy( sys.argv[1] )				
