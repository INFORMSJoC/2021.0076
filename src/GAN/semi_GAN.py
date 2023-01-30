import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense,Conv2D,Conv2DTranspose,Input,Reshape,Activation,Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.layers import BatchNormalization,Dropout,Flatten
from keras.models import Sequential,Model
import keras.backend as K
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

from keras import backend as K


z_dim = 40

num_classes = 2
iterations = 3000
batch_size = 5
sample_interval = 1000

class Dataset:
  def __init__(self, X, y, unlabelRate):
    
    sss = StratifiedShuffleSplit(n_splits=1,test_size=unlabelRate, random_state=1)              
    for train_index, test_index in sss.split(X, y):
        X_labeled = X[train_index]
        y_labeled = y[train_index]
        
        X_unlabel = X[test_index]
        y_unlabel = y[test_index]

    self.num_classes = num_classes
    self.datashape = X[0].shape
   
    self.X_labeled = X_labeled
    self.y_labeled = y_labeled
    self.X_unlabel = X_unlabel
    self.y_unlabel = y_unlabel
    
#    self.x_test = X_te
#    self.y_test = y_te
      
    def preprocess_labels(y):
      return y.reshape((-1,1))

    self.y_labeled = preprocess_labels(self.y_labeled)
    self.y_unlabel = preprocess_labels(self.y_unlabel)

  def batch_labeled(self, batch_size):
      
      ix  = np.random.randint(0,len(self.X_labeled),batch_size)  #每次从有标签样本里面抽取batch_size个样本
      X_batch = self.X_labeled[ix]
      y_batch = self.y_labeled[ix]
      
      return X_batch, y_batch

  def batch_unlabeled(self, batch_size):
      
      idx = np.random.randint(0, len(self.X_unlabel), batch_size)
      X_batch = self.X_unlabel[idx]
      return X_batch

def build_generator(z_dim, img_shape):
    

    model = Sequential()
    model.add(Dense(32,  input_dim=z_dim))
    model.add(Dense(32,  activation='relu'))
    model.add(Dense(32,  activation='sigmoid'))
  
    model.add(Dense(np.prod(img_shape), activation='tanh'))
  
    noise = Input(shape=(z_dim,))
    img = model(noise)
  
    return Model(noise, img)


def build_discriminator(img_shape):
    
    model = Sequential()
    model.add(Dense(20, input_shape=img_shape, init='uniform', activation='sigmoid'))
#        model.add(Dense(40,input_shape=self.img_shape))
    model.add(Dense(20,  activation='sigmoid'))
    model.add(Dense(20,  activation='sigmoid'))
    model.add(Dense(num_classes))

    img = Input(shape=img_shape)
    validity = model(img)    
    return Model(img, validity)
    


def build_discriminator_supervised(discriminator):
  model = Sequential()
  model.add(discriminator)
  model.add(Activation('softmax'))
  return model

def build_discriminator_unsupervised(discriminator):
  model = Sequential()
  model.add(discriminator)
  def custom_activation(x):
        
    prediction = 1.0 - (1.0 /
                           (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
    return prediction
  model.add(Lambda(custom_activation))
  
  return model


def build_gan(generator,discriminator):
    
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model




def train(dataset, iterations, batch_size, sample_interval):
    
    discriminator = build_discriminator(dataset.datashape)
    discriminator_supervised = build_discriminator_supervised(discriminator)
    discriminator_supervised.compile(optimizer= Adam(learning_rate=0.001),loss="categorical_crossentropy",metrics=['accuracy'])
    discriminator_unsupervised = build_discriminator_unsupervised(discriminator)
    discriminator_unsupervised.compile(optimizer = Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
#    
    generator = build_generator(z_dim, dataset.datashape)
    discriminator_unsupervised.trainable = False
    gan = build_gan(generator,discriminator_unsupervised)
    gan.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    
        
    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))
    

    for iteration in range(iterations):
#        print('iteration',iteration)
        imgs,labels = dataset.batch_labeled(batch_size)
    
        #print(labels)
        labels = to_categorical(labels,num_classes=num_classes)

        unlabeled_imgs = dataset.batch_unlabeled(batch_size)
    
        z = np.random.normal(0,1,(batch_size,z_dim))
        
        fake_imgs = generator.predict(z)
        #print(fake_imgs.shape)
    
        d_supervised_loss,accuracy = discriminator_supervised.train_on_batch(imgs,labels)
        d_unsupervised_loss_real = discriminator_unsupervised.train_on_batch(unlabeled_imgs,real)
        d_unsupervised_loss_fake = discriminator_unsupervised.train_on_batch(fake_imgs,fake)
        d_unsupervised_loss = 0.5*np.add(d_unsupervised_loss_real,d_unsupervised_loss_fake)
    
        z = np.random.normal(0,1,(batch_size,z_dim))
        fake_imgs = generator.predict(z)
        generator_loss = gan.train_on_batch(z,real)

    
        if(iteration+1) % sample_interval ==0:
    
    #      val_loss = discriminator_supervised.evaluate(x=x_test,y=y_test,verbose=0)
    
          print("Iteration No.:",iteration+1,end="\n")
#          print('Generator Loss:',generator_loss,end="\n")
#          print('Discriminator Unsuperived Loss:',d_unsupervised_loss,end="\n")
#          print('Accuracy Supervised:', accuracy)
      
    return discriminator_supervised

def creat_data(X, y, unlabelRate):
    
    dataset = Dataset(X, y, unlabelRate)
    
    X_unlabel = dataset.X_unlabel
    y_unlabel = dataset.y_unlabel
    X_labeled = dataset.X_labeled
    y_labeled = dataset.y_labeled
    
    D = train(dataset, iterations,batch_size, sample_interval)
    
    score = D.predict(X_unlabel)
    index  = np.max(score,1)>0.97
    print(score)
    print(sum(index))
    X_add = X_unlabel[index]
    y_add = D.predict_classes(X_unlabel)[index]
    print(y_add.shape)
    y_add = y_add.reshape((-1,1))

#    y_unlabel_pred = D.predict_classes(X_unlabel)
#    y_unlabel_pred = y_unlabel_pred.reshape((-1,1))
       
    X_total = np.vstack((X_labeled, X_add))
    y_total = np.vstack((y_labeled, y_add))
    y_total = y_total.reshape(1,-1)[0]
    
    K.clear_session()
    
    return X_total, y_total

