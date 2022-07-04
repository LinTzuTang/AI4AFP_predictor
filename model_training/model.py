import os
import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout,Add
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input,Flatten,Masking,BatchNormalization
from tensorflow.keras.layers import LSTM,Conv1D,MaxPool1D
from tensorflow.keras import Model
from tensorflow.keras import optimizers

def train_doc2vec_model(train_data, train_label, validation_data, validation_label, model_name, path = None):
    path = path or os.getcwd()
    if not os.path.isdir(path):
        os.mkdir(path)
    input_ = Input(shape=(50,1))
    cnn = Conv1D(64 ,8,activation = 'relu', padding="same")(input_)
    norm = BatchNormalization()(cnn)
    pool = MaxPool1D(5)
    d = Dropout(0.5)(norm)
    f = Flatten()(d)
    result = Dense(1, activation = "sigmoid")(f)
    model = Model(inputs=input_,outputs=result)

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    best_weights_filepath = os.path.join(path,'%s_best_weights.h5'%model_name)
    saveBestModel = tensorflow.keras.callbacks.ModelCheckpoint(best_weights_filepath, 
                                                        monitor='val_loss', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        mode='auto')
    CSVLogger = tensorflow.keras.callbacks.CSVLogger(os.path.join(path, "%s_csvLogger.csv"%model_name) ,separator=',', append=False)
    e_s = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=150,
                                          verbose=0, mode='min')

    history = model.fit(train_data,train_label, validation_data=(validation_data, validation_label),shuffle=True,
                    epochs=2000, batch_size=256,callbacks=[saveBestModel,CSVLogger,e_s])
    
    history.model.save(os.path.join(path,'%s_final_weights.h5'%model_name))
    
    return history

def train_pc6_model(train_data, train_label, validation_data, validation_label, model_name, path = None):
    path = path or os.getcwd()
    if not os.path.isdir(path):
        os.mkdir(path)
    input_ = Input(shape=(50,6))
    cnn = Conv1D(64 ,30,activation = 'relu', padding="same")(input_)
    norm = BatchNormalization()(cnn)
    pool = MaxPool1D(5)
    d = Dropout(0.5)(norm)
    f = Flatten()(d)
    result = Dense(1, activation = "sigmoid")(f)
    model = Model(inputs=input_,outputs=result)

    model.compile(optimizer=optimizers.Adam(learning_rate=2*1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    best_weights_filepath = os.path.join(path,'%s_best_weights.h5'%model_name)
    saveBestModel = tensorflow.keras.callbacks.ModelCheckpoint(best_weights_filepath, 
                                                        monitor='val_loss', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        mode='auto')
    CSVLogger = tensorflow.keras.callbacks.CSVLogger(os.path.join(path, "%s_csvLogger.csv"%model_name) ,separator=',', append=False)
    e_s = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=150,
                                          verbose=0, mode='min')

    history = model.fit(train_data,train_label, validation_data=(validation_data, validation_label),shuffle=True,
                    epochs=2000, batch_size=256,callbacks=[saveBestModel,CSVLogger,e_s])
    
    history.model.save(os.path.join(path,'%s_final_weights.h5'%model_name))
    
    return history