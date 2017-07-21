from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
import math
import numpy as np

def dae_model( img_shape ):
	kernel_size=3

	max_n_chans = 512	
	first_conv_chans = 32

	n_convs = int(math.log( max_n_chans/first_conv_chans, 2)) + 1
	conv_channels = [first_conv_chans*2**i for i in range(n_convs) ]
	
	x0 = Input( img_shape, name='dae_input' )
	x = x0

	n_downsampling_convs = int( math.log( img_shape[0] / 2**n_convs, 2))
	print(n_downsampling_convs)
	# apply strided convs down to desired number of channels first
	for i in range(n_convs):
		x = Conv2D( conv_channels[i], kernel_size=kernel_size, strides=(2,2),  activation='relu', padding='same', name='dae_conv2D_{}'.format(i) )(x)

	for i in range(n_downsampling_convs-1):
		x = Conv2D( conv_channels[-1], kernel_size=kernel_size, strides=(2,2), activation='relu', padding='same', name= 'dae_conv2D_{}'.format( n_convs + i) )(x)
#	x = Conv2D( 1, kernel_size=kernel_size, activation='relu', padding='same', name='dae_conv2D_last')(x)
#	x = Flatten()(x)		
#	x = Dense(activation='relu')(x)
#	x = Reshape
	x = Conv2D( conv_channels[-1], kernel_size=kernel_size, strides=(2,2), activation='relu', padding='same', name= 'dae_conv2D_encoding')(x)
	

	for i in range(n_convs-1):
		x = UpSampling2D( size=(2,2) )(x)
 		x = Conv2D( conv_channels[-i-2], kernel_size=kernel_size, activation='relu', padding='same', name='dae_deconv2D_{}'.format(i) )(x)
		print(x.shape)	

	x = UpSampling2D( size=(2,2) )(x)
	x_out = Conv2D( 3, kernel_size=kernel_size,  activation='relu', padding='same', name='dae_deconv2D_last' )(x)

	model = Model( input = [x0], output=[x_out] )
	return model


if __name__ == '__main__':
	model = dae_model( (256,256,3) )
	model.summary()

