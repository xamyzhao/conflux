import os
import sys

import nn_by_encoding
import numpy as np
import re
import encoding
import run_autoencoder

dataset_dir = '../datasets/MTGVS/train'
ims_train = [ os.path.join(dataset_dir,f) for f in os.listdir(dataset_dir) if f.endswith('.jpg') ]
n_train = len(ims_train)

n_eval = 50
n_correct = 0

encoding_mat, encoding_im_names = encoding.load_encoding_database('encodings.py')
model_file = './models/dae_epoch_{}.h5'.format(199)
model = run_autoencoder.load_dae_model(model_file,'predict')

for i in np.linspace(0,n_train-1, n_eval,dtype=int):
	im_num = re.search('[0-9]*(?=.jpg)', os.path.basename(ims_train[i])).group(0) 
	dists, matches = nn_by_encoding.find_nn( ims_train[i], model, encoding_mat, encoding_im_names )
	if im_num in matches:
		n_correct += 1

print('% correct: {}'.format( float(n_correct)/n_eval))
	
