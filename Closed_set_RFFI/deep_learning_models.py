from keras.layers import Input, Lambda, ReLU, Add
from keras.models import Model
from keras import backend as K
from keras.layers import (Dense, Conv2D, Flatten, BatchNormalization, AveragePooling2D)

import numpy as np

# In[]
'''Residual block'''
def resblock(x, kernelsize, filters, first_layer=False):
    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)

        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        fx = BatchNormalization()(fx)

        x = Conv2D(filters, 1, padding='same')(x)

        out = Add()([x, fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)

        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        fx = BatchNormalization()(fx)
        # 
        out = Add()([x, fx])
        out = ReLU()(out)

    return out


def classification_net(datashape, num_classes):
    datashape = datashape

    inputs = Input(shape=(np.append(datashape[1:-1], 1)))

    x = Conv2D(32, 7, strides=2, activation='relu', padding='same')(inputs)

    x = resblock(x, 3, 32)
    x = resblock(x, 3, 32)

    x = resblock(x, 3, 64, first_layer=True)
    x = resblock(x, 3, 64)

    x = AveragePooling2D(pool_size=2)(x)

    x = Flatten()(x)

    x = Dense(512)(x)

    x = Lambda(lambda x: K.l2_normalize(x, axis=1), name='feature_layer')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
