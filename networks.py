from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import math
import numpy as np


def dae_model( img_shape ):
	n_convs = int(math.log( img_shape[0], 2))
	conv_channels = [8*2**i for i in range(n_convs) ]
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

	model = Model( input = [x0], output=[x_out], name='dae_stackedconv' )
	return model

def siamese_tower( img_shape, name_prefix ):
	ks = 3
	x0 = Input( img_shape, name='{}_input'.format(name_prefix) )
	
	n_channels = [ 64, 64, 128, 128, 256, 256, 512, 512 ]
	x = x0
	for i in range(len(n_channels)):
		x = Conv2D( n_channels[i], kernel_size=ks, strides=(2,2), activation='relu', padding='same', name='{}_conv2D_{}'.format( name_prefix,i ))(x)

	x = Flatten()(x)
	y = Dense( 512 )(x)
#	y = Conv2D( n_channels[-1], kernel_size=ks, strides=(2,2), activation='relu', padding='same', name='{}_conv2D_encoding'.format( name_prefix ))(x)

	model = Model( inputs = x0, outputs = y, name= name_prefix )
	return model

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def siamese_model( img_shape, tower_model ):
	input_A = Input( img_shape )
	input_B = Input( img_shape )

#	tower_A = siamese_tower( img_shape, 'tower_A')
#	tower_B = siamese_tower( img_shape, 'tower_B')
	tower_model.summary()
	x_A = tower_model( input_A )
	x_B = tower_model( input_B )

	distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([x_A, x_B])

	model = Model([input_A, input_B], distance, name='siamese')
	return model


def make_model( model_name ):
	img_shape = (256,256,3)
	if model_name == 'dae':
		model = dae_model( img_shape )
		model.compile( optimizer='adam', lr=2e-4, loss='mean_absolute_error' )
	elif model_name == 'dae_stackedconv':
		model = dae_stackedconv_model( img_shape )
		model.compile( optimizer='adam', lr=2e-4, loss='mean_absolute_error' )
	elif 'siamese' in model_name:
		model_tower = siamese_tower( (256,256,3), 'tower' )
		model_tower.compile( optimizer='adam', lr=2e-4, loss='mean_absolute_error' )

		model = siamese_model( (256,256,3), model_tower )
		model.compile( optimizer='adam', lr=2e-4, loss=contrastive_loss )

		models = [model_tower, model]
	model.summary()

	if 'siamese' in model_name:
		return models
	else:
		return model

if __name__ == '__main__':
	if len(sys.argv) > 1:
		make_model( sys.argv[1] )
	else:
		make_model('dae')
			

