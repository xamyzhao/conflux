from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
import math
import numpy as np

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
import math
import numpy as np


def dae_model( img_shape ):
	
	n_convs = int(math.log( img_shape[0], 2))

	#conv_channels = [ 2**i for i in range(n_convs) ]
	conv_channels = [8*2**i for i in range(n_convs) ]
	print(conv_channels)
	x0 = Input( img_shape, name='dae_input' )
	x = x0
	kernel_size=4
	for i in range(n_convs-1):
		x = Conv2D( conv_channels[i], kernel_size=kernel_size, strides=(2,2),  activation='relu', padding='same', name='dae_conv2D_{}'.format(i) )(x)

	x = Conv2D( conv_channels[-1], kernel_size=kernel_size, strides=(2,2),  activation='relu', padding='same', name='encoding' )(x)

	for i in range(n_convs-1):
		x = UpSampling2D( size=(2,2) )(x)
 		x = Conv2D( conv_channels[-i-2], kernel_size=kernel_size, activation='relu', padding='same', name='dae_deconv2D_{}'.format(i) )(x)

	x = UpSampling2D( size=(2,2) )(x)
	x_out = Conv2D( 3, kernel_size=kernel_size,  activation='relu', padding='same', name='dae_deconv2D_last' )(x)

	model = Model( input = [x0], output=[x_out], name='dae' )
	return model


if __name__ == '__main__':
	model = denoising_autoencoder_model( (256,256,3) )
	model.summary()

def dae_stackedconv_model( img_shape ):
	kernel_size=3

	max_n_chans = 512	
	first_conv_chans = 64

	n_convs = int(math.log( img_shape[0], 2)/2)
	
	conv_channels = [first_conv_chans*2**i for i in range(n_convs) ]
	x0 = Input( img_shape, name='dae_input' )
	x = x0

	for i in range(n_convs):
		x = Conv2D( conv_channels[i], kernel_size=kernel_size, strides=(2,2),  activation='relu', padding='same', name='dae_conv2D_{}'.format(i*2+1) )(x)
		x = Conv2D( conv_channels[i], kernel_size=kernel_size, strides=(2,2),  activation='relu', padding='same', name='dae_conv2D_{}'.format(i*2+2) )(x)

	x = Flatten()(x)
	x = Dense(512, name='encoding')(x)
	x = Reshape( (1,1,-1) )(x)

	x = UpSampling2D( size=(2,2) )(x)
	x = Conv2D( conv_channels[-1], kernel_size=kernel_size, activation='relu', padding='same', name='dae_deconv2D_{}'.format(0) )(x)
	
	for i in range(0,n_convs-1):
		x = UpSampling2D( size=(2,2) )(x)
 		x = Conv2D( conv_channels[-i-2], kernel_size=kernel_size, activation='relu', padding='same', name='dae_deconv2D_{}'.format(i*2+1) )(x)
		x = UpSampling2D( size=(2,2) )(x)
 		x = Conv2D( conv_channels[-i-2], kernel_size=kernel_size, activation='relu', padding='same', name='dae_deconv2D_{}'.format(i*2+2) )(x)

	x = UpSampling2D( size=(2,2) )(x)
	x_out = Conv2D( 3, kernel_size=kernel_size,  activation='relu', padding='same', name='dae_deconv2D_last' )(x)

	model = Model( input = [x0], output=[x_out] )
	return model


def make_model( model_name ):
	img_shape = (256,256,3)
	if model_name == 'dae':
		model = dae_model( img_shape )
	elif model_name == 'dae_stackedconv':
		model = dae_stackedconv_model( img_shape )
	model.summary()
	return model

if __name__ == '__main__':
#	model = dae_model( (256,256,3) )
#	model.summary()
	if len(sys.argv) > 1:
		maek_model( sys.argv[1] )
	else:
		make_model('dae')
			

