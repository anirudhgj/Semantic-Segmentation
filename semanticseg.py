import LoadBatches
from keras import backend as K
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16
import os
import glob
import cv2
import numpy as np
import random
import itertools
from keras.utils import plot_model
	
save_weights_path="weights/ex1"
epoch_number=5
test_images="data/dataset1/images_prepped_test/"
output_path="data/predictions/"
input_height=224
input_width=224
model_name = "vgg_segnet"
n_classes = 10
VGG_Weights_path = "data/vgg16_weights_th_dim_ordering_th_kernels.h5"
train_images_path = "data/dataset1/images_prepped_train/"
train_segs_path = "data/dataset1/annotations_prepped_train/"
train_batch_size = 2
validate = 0
epochs = 5
optimizer_name = "adadelta"
load_weights=""

#==============================================================================
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.allocator_type ='BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
#==============================================================================


if validate:
    val_images_path = "data/dataset1/images_prepped_test/"
    val_segs_path = "data/dataset1/annotations_prepped_test/"
    val_batch_size = 2
    


#==============================================================================
# def getImageArr( path , width , height):
#     img = cv2.imread(path, 1)
#     img = cv2.resize(img, ( width , height ))
#     img = img.astype(np.float32)
#     img[:,:,0] -= 103.939
#     img[:,:,1] -= 116.779
#     img[:,:,2] -= 123.68
#     return img
# 
# def getSegmentationArr( path , nClasses ,  width , height  ):
#     seg_labels = np.zeros((  height , width  , nClasses ))
#     img = cv2.imread(path, 1)
#     img = cv2.resize(img, ( width , height ))
#     img = img[:, : , 0]
#     for c in range(nClasses):
#         seg_labels[: , : , c ] = (img == c ).astype(int)   
#     seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
#     return seg_labels
# 
# 
# 
# def imageSegmentationGenerator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width   ):
#     print (output_height)
#     images = glob.glob( images_path + "*.png"  ) 
#     images.sort()
#     segmentations  = glob.glob( segs_path + "*.png"  )
#     segmentations.sort()
# 
#     zipped = itertools.cycle( zip(images,segmentations) )
#     while 1:
#         X = []
#         Y = []
#         for _ in range( batch_size) :
#             im , seg = next(zipped)
#             X.append( getImageArr(im , input_width , input_height ))  
#             Y.append( getSegmentationArr( seg , n_classes , output_width , output_height ))
# 
#==============================================================================

def VGGSegnet(n_classes):
    model = VGG16()
    i = model.get_layer("input_1").output
    o = model.get_layer("block4_pool").output 
    o = ( ZeroPadding2D( (1,1) , data_format='channels_last' ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D( (2,2), data_format='channels_last'))(o)
    o = ( ZeroPadding2D( (1,1), data_format='channels_last'))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_last' ) )(o)
    o = ( ZeroPadding2D((1,1) , data_format='channels_last' ))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_last' ))(o)
    o = ( BatchNormalization())(o)

    o = ( UpSampling2D((2,2)  , data_format='channels_last' ))(o)
    o = ( ZeroPadding2D((1,1)  , data_format='channels_last' ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_last' ))(o)
    o = ( BatchNormalization())(o)


    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_last',name='last_layer' )( o )
    o = ( BatchNormalization())(o)
    o_shape = Model(i, o ).output_shape
    outputHeight = o_shape[-3]
    outputWidth = o_shape[-2]
    x=(outputHeight*outputWidth)
    o = (Reshape((n_classes,x)))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( i, o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    model.summary()

    return model
      

m = VGGSegnet( n_classes   )
plot_model( m , to_file='C:/Users/Admin/Desktop/semantic seg/image-segmentation-keras-master/model.png',show_shapes=True, show_layer_names=True)
m.compile(loss='categorical_crossentropy',optimizer= optimizer_name,metrics=['accuracy'])

if len( load_weights ) > 0:
    m.load_weights(load_weights)


print ("Model output shape",m.output_shape )

output_height = m.outputHeight
output_width = m.outputWidth

G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


if validate:
    G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
    for ep in range( epochs ):
        m.fit_generator( G , 512  , epochs=1 )
        m.save_weights( save_weights_path + "." + str( ep ) )
        m.save( save_weights_path + ".model." + str( ep ) )
else:
    for ep in range( epochs ):
        m.fit_generator( G , 512  , validation_data=G2 , validation_steps=200 ,  epochs=1 )
        m.save_weights( save_weights_path + "." + str( ep )  )
        m.save( save_weights_path + ".model." + str( ep ) )



