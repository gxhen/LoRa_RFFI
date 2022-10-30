import numpy as np


from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, ReLU, Add, Dense, Conv2D, Flatten, AveragePooling2D


# In[]
def resblock(x, kernelsize, filters, first_layer = False):

    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        
        x = Conv2D(filters, 1, padding='same')(x)
        
        out = Add()([x,fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same')(x)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same')(fx)
        
        
        out = Add()([x,fx])
        out = ReLU()(out)

    return out 

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)           

    
class TripletNet():
    def __init__(self):
        pass
        
    def create_triplet_net(self, embedding_net, alpha):
        
#        embedding_net = encoder()
        self.alpha = alpha
        
        input_1 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_2 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_3 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N = embedding_net(input_3)
   
        loss = Lambda(self.triplet_loss)([A, P, N]) 
        model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
        return model
      
    def triplet_loss(self,x):
    # Triplet Loss function.
        anchor,positive,negative = x
#        K.l2_normalize
    # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor-negative),axis=1)

        basic_loss = pos_dist-neg_dist + self.alpha
        loss = K.maximum(basic_loss,0.0)
        return loss   
    
    def feature_extractor(self, datashape):
            
        self.datashape = datashape
        
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)
        
        x = resblock(x, 3, 32)
        x = resblock(x, 3, 32)

        x = resblock(x, 3, 64, first_layer = True)
        x = resblock(x, 3, 64)

        x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)
    
        x = Dense(512)(x)
  
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model             

    
    def get_triplet(self):
        """Choose a triplet (anchor, positive, negative) of images
        such that anchor and positive have the same label and
        anchor and negative have different labels."""
        
        
        n = a = self.dev_range[np.random.randint(len(self.dev_range))]
        
        while n == a:
            # keep searching randomly!
            n = self.dev_range[np.random.randint(len(self.dev_range))]
        a, p = self.call_sample(a), self.call_sample(a)
        n = self.call_sample(n)
        
        return a, p, n

          
    def call_sample(self,label_name):
        """Choose an image from our training or test data with the
        given label."""
        num_sample = len(self.label)
        idx = np.random.randint(num_sample)
        while self.label[idx] != label_name:
            # keep searching randomly!
            idx = np.random.randint(num_sample) 
        return self.data[idx]


    def create_generator(self, batchsize, dev_range, data, label):
        """Generate a triplets generator for training."""
        self.data = data
        self.label = label
        self.dev_range = dev_range
        
        while True:
            list_a = []
            list_p = []
            list_n = []

            for i in range(batchsize):
                a, p, n = self.get_triplet()
                list_a.append(a)
                list_p.append(p)
                list_n.append(n)
            
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')
            
           # a "dummy" label which will come in to our identity loss
           # function below as y_true. We'll ignore it.
            label = np.ones(batchsize)
            yield [A, P, N], label  

