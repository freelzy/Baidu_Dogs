#coding:utf-8
import gc
from keras.applications import *
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.optimizers import *
from keras.preprocessing.image import *
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import *
from keras.models import Model
from keras import optimizers
import math
from keras.callbacks import *
import matplotlib.pyplot as plt
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
import tensorflow as tf
import datetime

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(Concatenate(axis=0)(outputs))
        
        return Model(model.inputs, merged)

#VGG，ResNet图片预处理
def preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

image_size=(299,299)#跑ResNet等时修改为(224,224)
ft_epoch=12 #微调迭代的epoch数

batch_size=32
np.random.seed(1000)
train_dir = ""
val_dir = ""
top_weights_file='bottleneck_fc_model.h5'#已经训练好的顶层模型权重
ft_weights_file='FineTurning/'#微调后，整个网络的权重文件保存目录


for freeze_layer in [133，165，197，229，249]:
    input_tensor = Input(shape=(image_size[0],image_size[1],3))
    x = input_tensor
    x = Xception(weights='imagenet', include_top=False, input_tensor=x)#换网络时要修改的
    x = GlobalAveragePooling2D()(x.output)

    top_model_input=Input((int(x.shape[-1]),))
    x2=top_model_input
    x2=Dropout(0.5)(x2)
    x2=Dense(256, activation='relu')(x2)
    x2=Dropout(0.5)(x2)
    prediction=Dense(100, activation='softmax')(x2)
    top_model=Model(top_model_input,prediction)

    top_model.load_weights(top_weights_file)

    output_tensor = top_model(x)

    # this is the model we will train
    model = Model(inputs=input_tensor, outputs=output_tensor)
    print(len(model.layers))

    for layer in model.layers[:freeze_layer]:
       layer.trainable = False
    for layer in model.layers[freeze_layer:]:
       layer.trainable = True


    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
    model_parallel = make_parallel(model, 2)
    model_parallel.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    train_gen = ImageDataGenerator(
        preprocessing_function=xception.preprocess_input
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=xception.preprocess_input
    )

    classes=list(range(100))
    for i,c in zip(range(100),classes):
        classes[i]=str(c)

    train_generator = train_gen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        # shuffle=False,
        classes=classes,
    )
    val_generator = val_gen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        # shuffle=False,
        classes=classes
    )

    model_parallel.fit_generator(
    #model.fit_generator(
            train_generator,
            steps_per_epoch= math.ceil(len(train_generator.filenames)/batch_size),
            epochs=ft_epoch,
            validation_data=val_generator,
            validation_steps=math.ceil(len(val_generator.filenames)/batch_size)
    )

    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    model.save_weights(ft_weights_file+'/%s.h5' % (now+'_'+str(freeze_layer)))
    gc.collect()


