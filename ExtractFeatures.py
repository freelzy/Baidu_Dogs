#Baseline参考博客 https://ypw.io/dogs-vs-cats-2/#more
from keras.applications import *
from keras.preprocessing.image import *
import h5py
import math
import gc
from keras.models import *
from keras.layers import *
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.callbacks import *
from nets.DenseNet.densenet121 import DenseNet121
from nets.DenseNet.densenet161 import DenseNet161
from nets.DenseNet.densenet169 import DenseNet169
from nets.resnet152 import ResNet152
from nets.resnet101 import ResNet101
from nets import inception_resnet_v2
from nets.inception_resnet_v2 import InceptionResNetV2
from nets.InceptionV4.inception_v4 import InceptionV4
import warnings
warnings.filterwarnings('ignore')
import winsound

Freq = 2500
Dur = 1200


#VGG，ResNet图片预处理
def preprocess_input(x):
    x[:, :, 0] = (x[:, :, 0] - 124) * 0.0167
    x[:, :, 1] = (x[:, :, 1] - 117) * 0.0167
    x[:, :, 2] = (x[:, :, 2] - 104) * 0.0167
    return x
def r_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x
def d_preprocess_input(x):
    x = x[:, :, ::-1]
    x[:, :, 0] = (x[:, :, 0] - 103.94) * 0.017
    x[:, :, 1] = (x[:, :, 1] - 116.78) * 0.017
    x[:, :, 2] = (x[:, :, 2] - 123.68) * 0.017
    return x
# def addup(x):
#     return 0.5*(x[0]+x[1])

h5path="D:\\PyCharm\\PyProjects\\NN\\model-final\\"
densenet_weights_root='nets\\DenseNet\\imagenet_models\\'

def write_gap(MODEL, image_size, lambda_func=None):
    batch_size = 32
    print(MODEL.__name__)
    if MODEL.__name__ in ['ResNet50','InceptionV3','Xception','ResNet152','ResNet101','InceptionResNetV2']:
        batch_size = 20
        input_tensor = Input((image_size[0], image_size[1], 3))
        x = input_tensor
        base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
        out=GlobalAveragePooling2D()(base_model.output)
        print(out.shape)
        model = Model(input_tensor, out)
    elif MODEL.__name__=='DenseNet121':
        densenet_weights_path = densenet_weights_root +'densenet121_weights_tf.h5'
        model = MODEL(reduction=0.5, weights_path=densenet_weights_path)
    elif MODEL.__name__=='DenseNet161':
        densenet_weights_path = densenet_weights_root +'densenet161_weights_tf.h5'
        model = MODEL(reduction=0.5, weights_path=densenet_weights_path)
    elif MODEL.__name__=='DenseNet169':
        densenet_weights_path = densenet_weights_root + 'densenet169_weights_tf.h5'
        model = MODEL(reduction=0.5, weights_path=densenet_weights_path)
    elif MODEL.__name__=='InceptionV4':
        base_model = MODEL(weights='imagenet', include_top=False)
        out = GlobalAveragePooling2D()(base_model.output)
        model = Model(base_model.input, out)

    gen = ImageDataGenerator(
        preprocessing_function=lambda_func
    )
    classes = list(range(97))
    for i, c in zip(range(97), classes):
        classes[i] = str(c)
    train_generator = gen.flow_from_directory(
                    "bddogtrain\\train",
                    image_size,
                    shuffle=False,
                    batch_size=batch_size,
                    classes=classes,
                    )
    test_generator = gen.flow_from_directory(
                    "bddogtrain\\test",
                    image_size,
                    shuffle=False,
                    batch_size=batch_size,
                    class_mode=None
    )
    y_train=np.array(train_generator.classes)
    # y_t=pd.Series(train_generator.filenames).str.split('\\').apply(lambda x:x[0]).astype(int).values
    # print((y_train-y_t).sum())
    train = model.predict_generator(train_generator, math.ceil(len(train_generator.filenames) / batch_size),verbose=1)
    with h5py.File(h5path+"gap_%s.h5" % MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("label", data=y_train)
    test = model.predict_generator(test_generator, math.ceil(len(test_generator.filenames)/batch_size),verbose=1)
    with h5py.File(h5path+"gap_%s_test.h5" % MODEL.__name__) as h:
        h.create_dataset("test", data=test)

# Xception 和 Inception V3 都需要将数据限定在 (-1, 1) 的范围内
write_gap(ResNet50, (224, 224),r_preprocess_input)
write_gap(ResNet101, (224, 224), r_preprocess_input)
write_gap(ResNet152, (224, 224), r_preprocess_input)
write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap(InceptionV4, (299, 299), inception_v3.preprocess_input)
write_gap(Xception, (299, 299), xception.preprocess_input)
write_gap(InceptionResNetV2, (299, 299), inception_resnet_v2.preprocess_input)
write_gap(DenseNet121, (224, 224), d_preprocess_input)
write_gap(DenseNet161, (224, 224), d_preprocess_input)
write_gap(DenseNet169, (224, 224), d_preprocess_input)

winsound.Beep(Freq,Dur)
gc.collect()

