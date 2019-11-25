import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Dropout,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from layers.graph import GraphLayer,GraphConv


class KerasModel(object):
    def __init__(self):#master,fc,shallow,tanh,unified
        pass
    def createMaster(self,drug_dim,cellline_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(celline_input)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)

        x_drug = Conv2D(filters=50, kernel_size=(1,200),strides=(1, 3), activation = 'tanh',padding='valid')(drug_input)
        x_drug = MaxPooling2D(pool_size=(1,5))(x_drug)
        x_drug = Conv2D(filters=30, kernel_size=(1,50),strides=(1, 5), activation = 'relu',padding='valid')(x_drug)
        x_drug = MaxPooling2D(pool_size=(1,10))(x_drug)
        x_drug = Flatten()(x_drug)
        x_drug = Dense(100,activation = 'relu')(x_drug)
        x_drug = Dropout(0.1)(x_drug)

        x = Concatenate()([x_cellline,x_drug])
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input],outputs=output)  
        return model
    def createFC(self,drug_dim,cellline_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(celline_input)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)

        x_drug = Conv2D(filters=50, kernel_size=(1,200),strides=(1, 3), activation = 'tanh',padding='valid')(drug_input)
        x_drug = MaxPooling2D(pool_size=(1,5))(x_drug)
        x_drug = Conv2D(filters=30, kernel_size=(1,50),strides=(1, 5), activation = 'relu',padding='valid')(x_drug)
        x_drug = MaxPooling2D(pool_size=(1,10))(x_drug)
        x_drug = Flatten()(x_drug)
        x_drug = Dense(100,activation = 'relu')(x_drug)
        x_drug = Dropout(0.1)(x_drug)

        x = Concatenate()([x_cellline,x_drug])
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(40,activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(10,activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input],outputs=output)  
        return model
    def createShallow(self,drug_dim,cellline_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        x_cellline = Conv2D(filters=64, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(celline_input)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)

        x_drug = Conv2D(filters=64, kernel_size=(1,200),strides=(1, 3), activation = 'tanh',padding='valid')(drug_input)
        x_drug = MaxPooling2D(pool_size=(1,5))(x_drug)
        x_drug = Flatten()(x_drug)
        x_drug = Dense(100,activation = 'relu')(x_drug)
        x_drug = Dropout(0.1)(x_drug)

        x = Concatenate()([x_cellline,x_drug])
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(40)(x)
        x = Dropout(0.2)(x)
        x = Dense(10,activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input],outputs=output)  
        return model
    def createTanh(self,drug_dim,cellline_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        x_cellline = BatchNormalization()(celline_input)
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)
        
        x_drug = BatchNormalization()(drug_input)
        x_drug = Conv2D(filters=50, kernel_size=(1,200),strides=(1, 3), activation = 'tanh',padding='valid')(x_drug)
        x_drug = MaxPooling2D(pool_size=(1,5))(x_drug)
        x_drug = Conv2D(filters=30, kernel_size=(1,50),strides=(1, 5), activation = 'relu',padding='valid')(x_drug)
        x_drug = MaxPooling2D(pool_size=(1,10))(x_drug)
        x_drug = Flatten()(x_drug)
        x_drug = Dense(100,activation = 'relu')(x_drug)
        x_drug = Dropout(0.1)(x_drug)

        x = Concatenate()([x_cellline,x_drug])
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(40,activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(10,activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input],outputs=output)  
        return model
    def createUnified(self,drug_dim,cellline_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        x_cellline = Concatenate(axis=-2)([drug_input,celline_input])
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)
        x_cellline = Dense(300,activation = 'tanh')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)
        x_cellline = Lambda(lambda x: K.expand_dims(x,axis=-1))(x_cellline)
        x_cellline = Lambda(lambda x: K.expand_dims(x,axis=1))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,2))(x_cellline)
        x_cellline = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,3))(x_cellline)
        x_cellline = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,3))(x_cellline)

        x = Dropout(0.1)(x_cellline)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input],outputs=output)  
        return model

class KerasModelIntegrated(object):#genomic mutation + gene expression
    def __init__(self,):
        pass
    def createMaster(self,drug_dim,cellline_dim,gexpr_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(celline_input)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)

        x_drug = Conv2D(filters=50, kernel_size=(1,200),strides=(1, 3), activation = 'tanh',padding='valid')(drug_input)
        x_drug = MaxPooling2D(pool_size=(1,5))(x_drug)
        x_drug = Conv2D(filters=30, kernel_size=(1,50),strides=(1, 5), activation = 'relu',padding='valid')(x_drug)
        x_drug = MaxPooling2D(pool_size=(1,10))(x_drug)
        x_drug = Flatten()(x_drug)
        x_drug = Dense(100,activation = 'relu')(x_drug)
        x_drug = Dropout(0.1)(x_drug)
        
        x_gexpr = Dense(100,activation = 'relu')(gexpr_input)
        x_gexpr = Dropout(0.1)(x_gexpr)
        
        x = Concatenate()([x_cellline,x_drug,x_gexpr])
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input,gexpr_input],outputs=output)  
        return model
    
    def createShallow(self,drug_dim,cellline_dim,gexpr_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        x_cellline = Conv2D(filters=64, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(celline_input)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)

        x_drug = Conv2D(filters=64, kernel_size=(1,200),strides=(1, 3), activation = 'tanh',padding='valid')(drug_input)
        x_drug = MaxPooling2D(pool_size=(1,5))(x_drug)
        x_drug = Flatten()(x_drug)
        x_drug = Dense(100,activation = 'relu')(x_drug)
        x_drug = Dropout(0.1)(x_drug)
        #gene expression feature
        x_gexpr = Dense(512,activation = 'relu')(gexpr_input)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(256,activation = 'relu')(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100,activation = 'relu')(x_gexpr)

        x = Concatenate()([x_cellline,x_drug,x_gexpr])
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(40)(x)
        x = Dropout(0.2)(x)
        x = Dense(10,activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input,gexpr_input],outputs=output)  
        return model
    
    def createFC(self,drug_dim,cellline_dim,gexpr_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(celline_input)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)

        x_drug = Conv2D(filters=50, kernel_size=(1,200),strides=(1, 3), activation = 'tanh',padding='valid')(drug_input)
        x_drug = MaxPooling2D(pool_size=(1,5))(x_drug)
        x_drug = Conv2D(filters=30, kernel_size=(1,50),strides=(1, 5), activation = 'relu',padding='valid')(x_drug)
        x_drug = MaxPooling2D(pool_size=(1,10))(x_drug)
        x_drug = Flatten()(x_drug)
        x_drug = Dense(100,activation = 'relu')(x_drug)
        x_drug = Dropout(0.1)(x_drug)

        #gene expression feature
        x_gexpr = Dense(512, activation = 'relu')(gexpr_input)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(256,activation = 'relu')(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100, activation = 'relu')(x_gexpr)
        
        x = Concatenate()([x_cellline,x_drug,x_gexpr])
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(40,activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(10,activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input,gexpr_input],outputs=output)  
        return model

    def createTanh(self,drug_dim,cellline_dim,gexpr_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        x_cellline = BatchNormalization()(celline_input)
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)
        
        x_drug = BatchNormalization()(drug_input)
        x_drug = Conv2D(filters=50, kernel_size=(1,200),strides=(1, 3), activation = 'tanh',padding='valid')(x_drug)
        x_drug = MaxPooling2D(pool_size=(1,5))(x_drug)
        x_drug = Conv2D(filters=30, kernel_size=(1,50),strides=(1, 5), activation = 'relu',padding='valid')(x_drug)
        x_drug = MaxPooling2D(pool_size=(1,10))(x_drug)
        x_drug = Flatten()(x_drug)
        x_drug = Dense(100,activation = 'relu')(x_drug)
        x_drug = Dropout(0.1)(x_drug)
        
        #gene expression feature
        x_gexpr = Dense(512,activation = 'relu')(gexpr_input)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(256,activation = 'relu')(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100,activation = 'relu')(x_gexpr)

        x = Concatenate()([x_cellline,x_drug,x_gexpr])
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dense(40,activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(10,activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input,gexpr_input],outputs=output)  
        return model
    
    def createUnified(self,drug_dim,cellline_dim,gexpr_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        
        #gene expression feature
        x_gexpr = Dense(512,activation = 'relu')(gexpr_input)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(256,activation = 'relu')(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100,activation = 'relu')(x_gexpr)
        
        x_cellline = Concatenate(axis=-2)([drug_input,celline_input])
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)
        x_cellline = Dense(300,activation = 'tanh')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)
        x_cellline = Concatenate()([x_cellline,x_gexpr])
        x_cellline = Lambda(lambda x: K.expand_dims(x,axis=-1))(x_cellline)
        x_cellline = Lambda(lambda x: K.expand_dims(x,axis=1))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,2))(x_cellline)
        x_cellline = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,3))(x_cellline)
        x_cellline = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,3))(x_cellline)

        x = Dropout(0.1)(x_cellline)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input,gexpr_input],outputs=output)  
        return model
        
    
class KerasMultiSourceModel(object):
    def __init__(self,use_gexpr=True,use_cn=True,use_methylation=True):#
        self.use_gexpr = use_gexpr
        self.use_cn = use_cn
        self.use_methylation = use_methylation
    def createMaster(self,drug_dim,cellline_dim,gexpr_dim,cn_dim,methy_dim):
        drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        cn_input = Input(shape=(cn_dim,),name='cn_feat_input')
        methy_input = Input(shape=(methy_dim,),name='methy_feat_input')
        #genomic mutation feature 
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(celline_input)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)
        #drug feature
        x_drug = Conv2D(filters=50, kernel_size=(1,200),strides=(1, 3), activation = 'tanh',padding='valid')(drug_input)
        x_drug = MaxPooling2D(pool_size=(1,5))(x_drug)
        x_drug = Conv2D(filters=30, kernel_size=(1,50),strides=(1, 5), activation = 'relu',padding='valid')(x_drug)
        x_drug = MaxPooling2D(pool_size=(1,10))(x_drug)
        x_drug = Flatten()(x_drug)
        x_drug = Dense(100,activation = 'relu')(x_drug)
        x_drug = Dropout(0.1)(x_drug)
        #gexp feature
        x_gexpr = Dense(256)(gexpr_input)
        x_gexpr = Activation('tanh')(x_gexpr)
        x_gexpr = BatchNormalization()(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100,activation='relu')(x_gexpr)
        #cn feature
        x_cn = Dense(256)(cn_input)
        x_cn = Activation('tanh')(x_cn)
        x_cn = BatchNormalization()(x_cn)
        x_cn = Dropout(0.1)(x_cn)
        x_cn = Dense(100,activation='relu')(x_cn)  
        #methylation feature
        x_methy = Dense(256)(methy_input)
        x_methy = Activation('tanh')(x_methy)
        x_methy = BatchNormalization()(x_methy)
        x_methy = Dropout(0.1)(x_methy)
        x_methy = Dense(100,activation='relu')(x_methy)
        x = Concatenate()([x_cellline,x_drug,x_gexpr,x_cn,x_methy])#100+100+100+100+100
        #combine
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        output = Dense(1,name='output')(x)
        model  = Model(inputs=[drug_input,celline_input,gexpr_input,cn_input,methy_input],outputs=output)  
        return model        
        
class KerasMultiSourceGCNModel(object):
    def __init__(self,use_mut,use_gexp,use_cn,use_methy,regr=True):#
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_cn = use_cn
        self.use_methy = use_methy
        self.regr = regr
    def createMaster(self,drug_dim,cellline_dim,gexpr_dim,cn_dim,methy_dim,units_list,use_relu=True,use_bn=True,use_GMP=True):
        #drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        drug_feat_input = Input(shape=(None,drug_dim),name='drug_feat_input')#drug_dim=75
        drug_adj_input = Input(shape=(None,None),name='drug_adj_input')
        
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        cn_input = Input(shape=(cn_dim,),name='cn_feat_input')
        methy_input = Input(shape=(methy_dim,),name='methy_feat_input')
        #drug feature with GCN
        GCN_layer = GraphConv(units=units_list[0],step_num=1)([drug_feat_input,drug_adj_input])
        if use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        
        for i in range(len(units_list)-1):
            GCN_layer = GraphConv(units=units_list[i+1],step_num=1)([GCN_layer,drug_adj_input])
            if use_relu:
                GCN_layer = Activation('relu')(GCN_layer)
            else:
                GCN_layer = Activation('tanh')(GCN_layer)
            if use_bn:
                GCN_layer = BatchNormalization()(GCN_layer)
            GCN_layer = Dropout(0.1)(GCN_layer)
        
        GCN_layer = GraphConv(units=100,step_num=1)([GCN_layer,drug_adj_input])
        if use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        #global pooling
        if use_GMP:
            x_drug = GlobalMaxPooling1D()(GCN_layer)
        else:
            x_drug = GlobalAveragePooling1D()(GCN_layer)

        #genomic mutation feature 
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(celline_input)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)
        #gexp feature
        x_gexpr = Dense(256)(gexpr_input)
        x_gexpr = Activation('tanh')(x_gexpr)
        x_gexpr = BatchNormalization()(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100,activation='relu')(x_gexpr)
        #cn feature
        x_cn = Dense(256)(cn_input)
        x_cn = Activation('tanh')(x_cn)
        x_cn = BatchNormalization()(x_cn)
        x_cn = Dropout(0.1)(x_cn)
        x_cn = Dense(100,activation='relu')(x_cn)  
        #methylation feature
        x_methy = Dense(256)(methy_input)
        x_methy = Activation('tanh')(x_methy)
        x_methy = BatchNormalization()(x_methy)
        x_methy = Dropout(0.1)(x_methy)
        x_methy = Dense(100,activation='relu')(x_methy)
#         x = x_drug
#         if self.use_mut:
#             x = Concatenate()([x,x_cellline])
#         if self.use_gexp:
#             x = Concatenate()([x,x_gexpr])
#         if self.use_cn:
#             x = Concatenate()([x,x_cn])
#         if self.use_methy:
#             x = Concatenate()([x,x_methy])
        x = Concatenate()([x_cellline,x_drug,x_gexpr,x_cn,x_methy])#100+100+100+100+100
        #combine
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        model  = Model(inputs=[drug_feat_input,drug_adj_input,celline_input,gexpr_input,cn_input,methy_input],outputs=output)  
        return model    
    def createNoAdj(self,drug_dim,cellline_dim,gexpr_dim,cn_dim,methy_dim,units_list,use_relu=True,use_bn=True,use_GMP=True):
        #drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        drug_feat_input = Input(shape=(None,drug_dim),name='drug_feat_input')#drug_dim=75
        drug_adj_input = Input(shape=(None,None),name='drug_adj_input')
        
        celline_input = Input(shape=(1,cellline_dim,1),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        cn_input = Input(shape=(cn_dim,),name='cn_feat_input')
        methy_input = Input(shape=(methy_dim,),name='methy_feat_input')
        
        #drug feature with Dense
        fc_layer = Dense(units_list[0])(drug_feat_input)
        
        if use_relu:
            fc_layer = Activation('relu')(fc_layer)
        else:
            fc_layer = Activation('tanh')(fc_layer)
            
        for i in range(len(units_list)-1):
            fc_layer = Dense(units_list[i+1])(fc_layer)
            if use_relu:
                fc_layer = Activation('relu')(fc_layer)
            else:
                fc_layer = Activation('tanh')(fc_layer)
            
        fc_layer = Dense(100)(fc_layer)
        if use_relu:
            fc_layer = Activation('relu')(fc_layer)
        else:
            fc_layer = Activation('tanh')(fc_layer)
            
        #global pooling
        if use_GMP:
            x_drug = GlobalMaxPooling1D()(fc_layer)
        else:
            x_drug = GlobalAveragePooling1D()(fc_layer)

        #genomic mutation feature 
        x_cellline = Conv2D(filters=50, kernel_size=(1,700),strides=(1, 5), activation = 'tanh',padding='valid')(celline_input)
        x_cellline = MaxPooling2D(pool_size=(1,5))(x_cellline)
        x_cellline = Conv2D(filters=30, kernel_size=(1,5),strides=(1, 2), activation = 'relu',padding='valid')(x_cellline)
        x_cellline = MaxPooling2D(pool_size=(1,10))(x_cellline)
        x_cellline = Flatten()(x_cellline)
        x_cellline = Dense(100,activation = 'relu')(x_cellline)
        x_cellline = Dropout(0.1)(x_cellline)
        #gexp feature
        x_gexpr = Dense(256)(gexpr_input)
        x_gexpr = Activation('tanh')(x_gexpr)
        x_gexpr = BatchNormalization()(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100,activation='relu')(x_gexpr)
        #cn feature
        x_cn = Dense(256)(cn_input)
        x_cn = Activation('tanh')(x_cn)
        x_cn = BatchNormalization()(x_cn)
        x_cn = Dropout(0.1)(x_cn)
        x_cn = Dense(100,activation='relu')(x_cn)  
        #methylation feature
        x_methy = Dense(256)(methy_input)
        x_methy = Activation('tanh')(x_methy)
        x_methy = BatchNormalization()(x_methy)
        x_methy = Dropout(0.1)(x_methy)
        x_methy = Dense(100,activation='relu')(x_methy)
        x = Concatenate()([x_cellline,x_drug,x_gexpr,x_cn,x_methy])#100+100+100+100+100
        #combine
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        model  = Model(inputs=[drug_feat_input,drug_adj_input,celline_input,gexpr_input,cn_input,methy_input],outputs=output)  
        return model  
    
class KerasMultiSourceDualGCNModel(object):
    def __init__(self,use_gexpr=True,use_cn=True,use_methylation=True,regr=True):#
        self.use_gexpr = use_gexpr
        self.use_cn = use_cn
        self.use_methylation = use_methylation
        self.regr = regr
    def createMaster(self,drug_dim,cell_line_dim,units_list,use_relu=True,use_bn=True,use_GMP=True):
        #drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        drug_feat_input = Input(shape=(None,drug_dim),name='drug_feat_input')#batch*100*75
        drug_adj_input = Input(shape=(None,None),name='drug_adj_input')#batch*100*100
        
        cell_line_feat_input = Input(shape=(None,cell_line_dim),name='cell_line_feat_input')#batch*577*4
        cell_line_adj_input = Input(shape=(None,None),name='cell_line_adj_input')#batch*577*577
        
        #drug feature with GCN
        GCN_layer = GraphConv(units=units_list[0],step_num=1)([drug_feat_input,drug_adj_input])
        #GCN_layer = GraphConv(units=64,step_num=1)([drug_feat_input,drug_adj_input])
        if use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        
        for i in range(len(units_list)-1):
        #for i in range(3):
            GCN_layer = GraphConv(units=units_list[i+1],step_num=1)([GCN_layer,drug_adj_input])
            #GCN_layer = GraphConv(units=64,step_num=1)([GCN_layer,drug_adj_input])
            if use_relu:
                GCN_layer = Activation('relu')(GCN_layer)
            else:
                GCN_layer = Activation('tanh')(GCN_layer)
            if use_bn:
                GCN_layer = BatchNormalization()(GCN_layer)
            GCN_layer = Dropout(0.1)(GCN_layer)
        
        GCN_layer = GraphConv(units=250,step_num=1)([GCN_layer,drug_adj_input])
        if use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        #global pooling
        if use_GMP:
            x_drug = GlobalMaxPooling1D()(GCN_layer)
        else:
            x_drug = GlobalAveragePooling1D()(GCN_layer)

        #cell line feature
#         cell_line_GCN = GraphConv(units=512,step_num=1)([cell_line_feat_input,cell_line_adj_input])
#         cell_line_GCN = Activation('relu')(cell_line_GCN)
#         cell_line_GCN = BatchNormalization()(cell_line_GCN)
#         cell_line_GCN = Dropout(0.1)(cell_line_GCN)
#         for i in range(1):
#             cell_line_GCN = GraphConv(units=512,step_num=1)([cell_line_GCN,cell_line_adj_input])
#             cell_line_GCN = Activation('relu')(cell_line_GCN)
#             cell_line_GCN = BatchNormalization()(cell_line_GCN)
#             cell_line_GCN = Dropout(0.1)(cell_line_GCN)
#         cell_line_GCN = GraphConv(units=250,step_num=1)([cell_line_GCN,cell_line_adj_input])
#         cell_line_GCN = Activation('relu')(cell_line_GCN)
#         cell_line_GCN = BatchNormalization()(cell_line_GCN)
#         cell_line_GCN = Dropout(0.1)(cell_line_GCN)
        cell_line_GCN = Dense(250,activation = 'relu')(cell_line_feat_input)
        cell_line_GCN = BatchNormalization()(cell_line_GCN)
        x_cell_line = GlobalAveragePooling1D()(cell_line_GCN)

        x = Concatenate()([x_cell_line,x_drug])#250+250
        #combine
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        #x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        #x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        #x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        model  = Model(inputs=[drug_feat_input,drug_adj_input,cell_line_feat_input,cell_line_adj_input],outputs=output)  
        return model
    def validMaster(self,drug_dim,cell_line_dim,nb_genes,units_list,use_relu=True,use_bn=True,use_GMP=True):
        drug_feat_input = Input(shape=(None,drug_dim),name='drug_feat_input')#batch*100*75
        drug_adj_input = Input(shape=(None,None),name='drug_adj_input')#batch*100*100        
        cell_line_feat_input = Input(shape=(nb_genes,cell_line_dim),name='cell_line_feat_input')#batch*nb_gene*nb_feat
        #drug feature with GCN
        GCN_layer = GraphConv(units=units_list[0],step_num=1)([drug_feat_input,drug_adj_input])
        #GCN_layer = GraphConv(units=64,step_num=1)([drug_feat_input,drug_adj_input])
        if use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        
        for i in range(len(units_list)-1):
        #for i in range(3):
            GCN_layer = GraphConv(units=units_list[i+1],step_num=1)([GCN_layer,drug_adj_input])
            #GCN_layer = GraphConv(units=64,step_num=1)([GCN_layer,drug_adj_input])
            if use_relu:
                GCN_layer = Activation('relu')(GCN_layer)
            else:
                GCN_layer = Activation('tanh')(GCN_layer)
            if use_bn:
                GCN_layer = BatchNormalization()(GCN_layer)
            GCN_layer = Dropout(0.1)(GCN_layer)
        
        GCN_layer = GraphConv(units=250,step_num=1)([GCN_layer,drug_adj_input])
        if use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        #global pooling
        if use_GMP:
            x_drug = GlobalMaxPooling1D()(GCN_layer)
        else:
            x_drug = GlobalAveragePooling1D()(GCN_layer)

        #cell line feature
        x_cell_line = Lambda(lambda x: K.permute_dimensions(x,pattern=(0, 2, 1)))(cell_line_feat_input)
        x_cell_line = Lambda(lambda x: K.expand_dims(x,axis=-1))(x_cell_line)
        x_cell_line = Conv2D(filters=256, kernel_size=(cell_line_dim,50),strides=(1, 1), activation = 'relu',padding='valid')(x_cell_line)
        x_cell_line = BatchNormalization()(x_cell_line)
        x_cell_line = MaxPooling2D(pool_size=(1,2))(x_cell_line)
        x_cell_line = Dropout(0.2)(x_cell_line)
        x_cell_line = Conv2D(filters=256, kernel_size=(1,20),strides=(1, 1), activation = 'relu',padding='valid')(x_cell_line)
        x_cell_line = BatchNormalization()(x_cell_line)
        x_cell_line = MaxPooling2D(pool_size=(1,2))(x_cell_line)
        x_cell_line = Dropout(0.2)(x_cell_line)
        x_cell_line = Lambda(lambda x: K.squeeze(x,axis=1))(x_cell_line)
        x_cell_line = GlobalAveragePooling1D()(x_cell_line)
        #x_cell_line = Flatten()(x_cell_line)
        
        x = Concatenate()([x_cell_line,x_drug])#250+250
        #combine
        x = Dense(300,activation = 'tanh')(x)
        x = Dropout(0.1)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=1))(x)
        x = Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
        #x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1,2))(x)
        x = Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        #x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
        #x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        model  = Model(inputs=[drug_feat_input,drug_adj_input,cell_line_feat_input],outputs=output)  
        return model
        

class KerastCNNSModel(object):
    def __init__(self,use_mut,use_gexp,use_cn,use_methy,regr=True):#
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_cn = use_cn
        self.use_methy = use_methy
        self.regr = regr
    def createMaster(self,drug_shape,cellline_dim,gexpr_dim,cn_dim,methy_dim,use_relu=True,use_bn=True,use_GMP=True):
        #drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        #drug_shape.shape[-2] = 28,drug_shape.shape[1] = 272
        drug_input = Input(shape=(272,34),name='drug_feat_input')
        
        celline_input = Input(shape=(cellline_dim,1),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        cn_input = Input(shape=(cn_dim,),name='cn_feat_input')
        methy_input = Input(shape=(methy_dim,),name='methy_feat_input')
        #drug feature with GCN
        x_drug = Conv1D(filters=40,kernel_size=7,strides=1,activation = 'relu',padding='same')(drug_input)
        x_drug = MaxPooling1D(pool_size=3,strides=3,padding='valid')(x_drug)
        x_drug = Conv1D(filters=80,kernel_size=7,strides=1,activation = 'relu',padding='same')(x_drug)
        x_drug = MaxPooling1D(pool_size=3,strides=3,padding='valid')(x_drug)
        x_drug = Conv1D(filters=60,kernel_size=7,strides=1,activation = 'relu',padding='same')(x_drug)
        x_drug = MaxPooling1D(pool_size=3,strides=3,padding='valid')(x_drug)
        x_drug = Flatten()(x_drug)
        #genomic mutation feature 
        x_cellline = Conv1D(filters=40,kernel_size=7,strides=1,activation = 'relu',padding='same')(celline_input)
        x_cellline = MaxPooling1D(pool_size=3,strides=3,padding='valid')(x_cellline)
        x_cellline = Conv1D(filters=80,kernel_size=7,strides=1,activation = 'relu',padding='same')(x_cellline)
        x_cellline = MaxPooling1D(pool_size=3,strides=3,padding='valid')(x_cellline)
        x_cellline = Conv1D(filters=60,kernel_size=7,strides=1,activation = 'relu',padding='same')(x_cellline)
        x_cellline = MaxPooling1D(pool_size=3,strides=3,padding='valid')(x_cellline)
        x_cellline = Flatten()(x_cellline)
        #gexp feature
        x_gexpr = Dense(256)(gexpr_input)
        x_gexpr = Activation('tanh')(x_gexpr)
        x_gexpr = BatchNormalization()(x_gexpr)
        x_gexpr = Dropout(0.1)(x_gexpr)
        x_gexpr = Dense(100,activation='relu')(x_gexpr)
        #cn feature
        x_cn = Dense(256)(cn_input)
        x_cn = Activation('tanh')(x_cn)
        x_cn = BatchNormalization()(x_cn)
        x_cn = Dropout(0.1)(x_cn)
        x_cn = Dense(100,activation='relu')(x_cn)  
        #methylation feature
        x_methy = Dense(256)(methy_input)
        x_methy = Activation('tanh')(x_methy)
        x_methy = BatchNormalization()(x_methy)
        x_methy = Dropout(0.1)(x_methy)
        x_methy = Dense(100,activation='relu')(x_methy)
#         x = x_drug
#         if self.use_mut:
#             x = Concatenate()([x,x_cellline])
#         if self.use_gexp:
#             x = Concatenate()([x,x_gexpr])
#         if self.use_cn:
#             x = Concatenate()([x,x_cn])
#         if self.use_methy:
#             x = Concatenate()([x,x_methy])
        x = Concatenate()([x_drug,x_cellline])
        #combine
        x = Dense(1024,activation = 'relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024,activation = 'relu')(x)
        x = Dropout(0.5)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        model  = Model(inputs=[drug_input,celline_input,gexpr_input,cn_input,methy_input],outputs=output)  
        return model    

class KerasMOLI(object):
    def __init__(self,use_mut,use_gexp,use_cn,regr=True):#
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_cn = use_cn
        self.regr = regr
    def createMaster(self,cellline_dim,gexpr_dim,cn_dim,use_relu=True,use_bn=True,use_GMP=True):
        #drug_input = Input(shape=(1,drug_dim,1),name='drug_feat_input')
        #drug_shape.shape[-2] = 28,drug_shape.shape[1] = 272
        
        celline_input = Input(shape=(cellline_dim,),name='cellline_feat_input')
        gexpr_input = Input(shape=(gexpr_dim,),name='gexpr_feat_input')
        cn_input = Input(shape=(cn_dim,),name='cn_feat_input')

        #genomic mutation feature 
        x_cellline = Dense(32)(celline_input)
        x_cellline = Activation('relu')(x_cellline)
        x_cellline = Dropout(0.5)(x_cellline)
        #gexp feature
        x_gexpr = Dense(32)(gexpr_input)
        x_gexpr = Activation('relu')(x_gexpr)
        x_gexpr = Dropout(0.5)(x_gexpr)
        #cn feature
        x_cn = Dense(32)(cn_input)
        x_cn = Activation('relu')(x_cn)
        x_cn = Dropout(0.5)(x_cn)

        x = Concatenate()([x_cellline,x_gexpr,x_cn])
        #combine
        x = Dropout(0.5)(x)
        if self.regr:
            output = Dense(1,name='output')(x)
        else:
            output = Dense(1,activation = 'sigmoid',name='output')(x)
        model  = Model(inputs=[celline_input,gexpr_input,cn_input],outputs=output)  
        return model  
        
        
        
       
    