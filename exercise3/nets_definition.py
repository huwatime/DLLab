from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

import tensorflow.contrib as tc

from layers_slim import *



def FCN_Seg(self, is_training=True):

    #Set training hyper-parameters
    self.is_training = is_training
    self.normalizer = tc.layers.batch_norm
    self.bn_params = {'is_training': self.is_training}


    print("input", self.tgt_image)

    with tf.variable_scope('First_conv'):
        # conv1: blurred input (HxWx32)
        conv1 = tc.layers.conv2d(self.tgt_image, 32, 3, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

        print("Conv1 shape")
        print(conv1.get_shape())

    # x1: blurred conv1 (HxWx16)
    x = inverted_bottleneck(conv1, 1, 16, 0,self.normalizer, self.bn_params, 1)
    #print("Conv 1")
    #print(x.get_shape())

    #180x180x24
    # x2: downsampled x1 (H/2xW/2x24)
    x = inverted_bottleneck(x, 6, 24, 1,self.normalizer, self.bn_params, 2)
    # x3: x2 + blurred x2 (H/2xW/2x24)
    x = inverted_bottleneck(x, 6, 24, 0,self.normalizer, self.bn_params, 3)

    print("Block One dim ")
    print(x)

    DB2_skip_connection = x

    #90x90x32
    # x4: downsampled x3 (H/4xW/4x32)
    x = inverted_bottleneck(x, 6, 32, 1,self.normalizer, self.bn_params, 4)
    # x5: x4 + blurred x4 (H/4xW/4x32)
    x = inverted_bottleneck(x, 6, 32, 0,self.normalizer, self.bn_params, 5)

    print("Block Two dim ")
    print(x)

    DB3_skip_connection = x

    #45x45x96
    x = inverted_bottleneck(x, 6, 64, 1,self.normalizer, self.bn_params, 6)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 7)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 8)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 9)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 10)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 11)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 12)

    print("Block Three dim ")
    print(x)

    DB4_skip_connection = x

    #23x23x160
    x = inverted_bottleneck(x, 6, 160, 1,self.normalizer, self.bn_params, 13)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 14)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 15)

    print("Block Four dim ")
    print(x)

    #23x23x320
    x = inverted_bottleneck(x, 6, 320, 0,self.normalizer, self.bn_params, 16)

    print("Block Four dim ")
    print(x)


    # Configuration 1 - single upsampling layer
    if self.configuration == 1:

        #input is features named 'x'

        # (1.1) - incorporate a upsample function which takes the features of x
        # and produces 120 output feature maps, which are 16x bigger in resolution than
        # x. Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up5

        current_up5 = TransitionUp_elu(x, 120, 16, 'conf1_up5')
        current_up5 = crop(current_up5, self.tgt_image)
        current_up5 = tf.layers.dropout(current_up5)

        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    # Configuration 2 - single upsampling layer plus skip connection
    if self.configuration == 2:

        #input is features named 'x'

        # (2.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps
        current_up2 = TransitionUp_elu(x, 160, 2, 'conf2_up2')
        current_up2 = crop(current_up2, DB4_skip_connection)
        concat2 = Concat_layers(current_up2, DB4_skip_connection)
        conv2 = Convolution(concat2, 256, 3, 'conf2_conv2')

        # (2.2) - incorporate a upsample function which takes the features from (2.1)
        # and produces 120 output feature maps, which are 8x bigger in resolution than
        # (2.1). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up3
        current_up3 = TransitionUp_elu(conv2, 120, 8, 'conf2_up3')
        current_up3 = crop(current_up3, self.tgt_image)
        current_up3 = tf.layers.dropout(current_up3)

        End_maps_decoder1 = slim.conv2d(current_up3, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    # Configuration 3 - Two upsampling layer plus skip connection
    if self.configuration == 3:

        #input is features named 'x'

        # (3.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps
        current_up2 = TransitionUp_elu(x, 160, 2, 'conf3_up2')
        current_up2 = crop(current_up2, DB4_skip_connection)
        concat2 = Concat_layers(current_up2, DB4_skip_connection)
        conv2 = Convolution(concat2, 256, 3, 'conf3_conv2')

        # (3.2) - Repeat (3.1) now producing 160 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.
        current_up3 = TransitionUp_elu(conv2, 128, 2, 'conf3_up3')
        current_up3 = crop(current_up3, DB3_skip_connection)
        concat3 = Concat_layers(current_up3, DB3_skip_connection)
        conv3 = Convolution(concat3, 160, 3, 'conf3_conv3')

        # (3.3) - incorporate a upsample function which takes the features from (3.2)
        # and produces 120 output feature maps which are 4x bigger in resolution than
        # (3.2). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4
        current_up4 = TransitionUp_elu(conv3, 120, 4, 'conf3_up4')
        current_up4 = crop(current_up4, self.tgt_image)
        current_up4 = tf.layers.dropout(current_up4)

        End_maps_decoder1 = slim.conv2d(current_up4, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    #Full configuration
    if self.configuration == 4:

        ######################################################################################
        ######################################### DECODER Full #############################################



        # (4.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps
        current_up2 = TransitionUp_elu(x, 160, 2, 'conf4_up2')
        current_up2 = crop(current_up2, DB4_skip_connection)
        concat2 = Concat_layers(current_up2, DB4_skip_connection)
        conv2 = Convolution(concat2, 256, 3, 'conf4_conv2')

        # (4.2) - Repeat (4.1) now producing 160 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.
        current_up3 = TransitionUp_elu(conv2, 128, 2, 'conf4_up3')
        current_up3 = crop(current_up3, DB3_skip_connection)
        concat3 = Concat_layers(current_up3, DB3_skip_connection)
        conv3 = Convolution(concat3, 160, 3, 'conf4_conv3')

        # (4.3) - Repeat (4.2) now producing 96 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB2_skip_connection) through concatenation.
        current_up4 = TransitionUp_elu(conv3, 72, 2, 'conf4_up4')
        current_up4 = crop(current_up4, DB2_skip_connection)
        concat4 = Concat_layers(current_up4, DB2_skip_connection)
        conv4 = Convolution(concat4, 96, 3, 'conf4_conv4')

        # (4.4) - incorporate a upsample function which takes the features from (4.3)
        # and produce 120 output feature maps which are 2x bigger in resolution than
        # (4.3). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4
        current_up5 = TransitionUp_elu(conv4, 120, 2, 'conf4_up5')
        current_up5 = crop(current_up5, self.tgt_image)
        current_up5 = tf.layers.dropout(current_up5)

        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    return Reshaped_map

