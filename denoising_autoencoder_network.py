from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
import math
import numpy as np

def dae_model( img_shape ):
	
	n_convs = int(math.log( img_shape[0], 2))

	#conv_channels = [ 2**i for i in range(n_convs) ]
	conv_channels = [8*2**i for i in range(n_convs) ]
	print(conv_channels)
	x0 = Input( img_shape )
	x = x0
	for i in range(n_convs):
		x = Conv2D( conv_channels[i], (3,3),  activation='relu', padding='same' )(x)
		x = MaxPooling2D( (2,2), strides=(2,2) )(x)
		print(x.shape)
	print('Starting deconv')
	for i in range(n_convs-1):
		x = UpSampling2D( size=(2,2) )(x)
 		x = Conv2D( conv_channels[-i-2], (3,3), activation='relu', padding='same' )(x)
		print(x.shape)	

	x = UpSampling2D( size=(2,2) )(x)
	x_out = Conv2D( 3, (3,3),  activation='relu', padding='same' )(x)

	model = Model( input = [x0], output=[x_out] )
	return model


if __name__ == '__main__':
	model = denoising_autoencoder_model( (256,256,3) )
	model.summary()

