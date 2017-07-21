import json
import numpy as np

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
