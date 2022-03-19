# import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns
import pickle
import pandas as pd
import re
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense,Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import numpy as np
import string
from string import digits
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from Data import Dataset,Dataloder



"""########################################------MODEL------########################################
"""

########################################------Encoder model------########################################
class Encoder(tf.keras.Model):


    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):
        super().__init__()

        self.inp_vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.input_length = input_length
        #Initialize Embedding layer

    def build(self,input_shape):
        self.embedding = Embedding(input_dim=self.inp_vocab_size, output_dim=self.embedding_size,
                                  input_length=self.input_length,trainable=True,name="encoder_embed")
        #Intialize Encoder LSTM layer
        self.bilstm = tf.keras.layers.Bidirectional(LSTM(units = self.lstm_size,return_sequences=True,return_state=True),merge_mode='sum')

    def call(self,input_sequence,initial_state):
        '''
          Input:Input_sequence[batch_size,input_length]
                Initial_state 4x[batch_size,encoder_units]
          
          Output: lstm_enc_output [batch_size,input_length,encoder_units]
                  forward_h/c & backward_h/c [batch_size,encoder_units]
        '''
        # print("initial_state",len(initial_state))
        input_embd = self.embedding(input_sequence)
        lstm_enc_output, forward_h, forward_c, backward_h, backward_c = self.bilstm(input_embd,initial_state)
        return lstm_enc_output, forward_h, forward_c, backward_h, backward_c
        # return lstm_enc_output, forward_h, forward_c

    
    def initialize_states(self,batch_size):
      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
      '''
      self.lstm_state_h = tf.random.uniform(shape=[batch_size,self.lstm_size],dtype=tf.float32)
      self.lstm_state_c = tf.random.uniform(shape=[batch_size,self.lstm_size],dtype=tf.float32)
      return self.lstm_state_h,self.lstm_state_c

    def initialize_states_bidirectional(self,batch_size):
      states = [tf.zeros((batch_size, self.lstm_size)) for i in range(4)]
      return states



########################################------Attention model------########################################
class Attention(tf.keras.layers.Layer):
    def __init__(self,scoring_function, att_units):
        super().__init__()
        self.att_units = att_units
        self.scoring_function = scoring_function
        # self.batch_size = batch_size
        # Please go through the reference notebook and research paper to complete the scoring functions

        if self.scoring_function=='dot':
            pass
        
        elif scoring_function == 'general':
            self.dense = Dense(self.att_units)
        
        elif scoring_function == 'concat':
            self.dense = tf.keras.layers.Dense(att_units, activation='tanh')
            self.dense1 = tf.keras.layers.Dense(1)
  
  
    def call(self,decoder_hidden_state,encoder_output):


    
        if self.scoring_function == 'dot':
            decoder_hidden_state = tf.expand_dims(decoder_hidden_state,axis=2)
            similarity = tf.matmul(encoder_output,decoder_hidden_state)
            weights    = tf.nn.softmax(similarity,axis=1)
            context_vector = tf.matmul(weights,encoder_output,transpose_a=True)
            context_vector = tf.squeeze(context_vector, axis=1)
            return context_vector,weights

        elif self.scoring_function == 'general':
            decoder_hidden_state=tf.expand_dims(decoder_hidden_state, 1)
            score = tf.matmul(decoder_hidden_state, self.dense(
                    encoder_output), transpose_b=True)
            attention_weights = tf.keras.activations.softmax(score, axis=-1) 
            context_vector = tf.matmul(attention_weights, encoder_output)
            context_vector=tf.reduce_sum(context_vector, axis=1)
            attention_weights=tf.reduce_sum(attention_weights, axis=1)
            attention_weights=tf.expand_dims(attention_weights, 2)
            return context_vector,attention_weights

        elif self.scoring_function == 'concat':
            decoder_hidden_state=tf.expand_dims(decoder_hidden_state, 1)
            decoder_hidden_state = tf.tile(
                        decoder_hidden_state, [1,30, 1])
            score = self.dense1(
                        self.dense(tf.concat((decoder_hidden_state, encoder_output), axis=-1)))
            score = tf.transpose(score, [0, 2, 1])
            attention_weights = tf.keras.activations.softmax(score, axis=-1) 
            context_vector = tf.matmul(attention_weights, encoder_output)
            context_vector=tf.reduce_sum(context_vector, axis=1)
            attention_weights=tf.reduce_sum(attention_weights, axis=1)
            attention_weights=tf.expand_dims(attention_weights, 2)
            
            return context_vector,attention_weights
    

########################################------OneStepDecoder model------########################################
class OneStepDecoder(tf.keras.Model):
    def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):

      # Initialize decoder embedding layer, LSTM and any other objects needed
      super().__init__()
      self.tar_vocab_size = tar_vocab_size
      self.embedding_dim = embedding_dim
      self.input_length = input_length
      self.dec_units = dec_units
      self.score_fun = score_fun
      self.att_units = att_units

    def build(self,input_shape):
      self.attention = Attention('concat', self.att_units)
      self.embedding = Embedding(input_dim=self.tar_vocab_size,output_dim=self.embedding_dim,
                                 input_length=self.input_length,mask_zero=True,trainable=True,name="Decoder_Embed")
      self.bilstm = tf.keras.layers.Bidirectional(LSTM(units = self.dec_units,return_sequences=True,return_state=True),merge_mode='sum')
      self.dense = Dense(self.tar_vocab_size)
      


    def call(self,input_to_decoder, encoder_output, f_state_h,f_state_c,b_state_h,b_state_c):
        dec_embd = self.embedding(input_to_decoder)
        context_vectors,attention_weights = self.attention(f_state_h,encoder_output)
        context_vectors_ = tf.expand_dims(context_vectors,axis=1)
        concat_vector = tf.concat([dec_embd,context_vectors_],axis=2)
        states = [f_state_h,f_state_c,b_state_h,b_state_c]
        decoder_outputs,dec_f_state_h,dec_f_state_c,dec_b_state_h,dec_b_state_c = self.bilstm(concat_vector,states)
        decoder_outputs = tf.squeeze(decoder_outputs,axis=1)
        dense_output = self.dense(decoder_outputs)
        
        return dense_output,dec_f_state_h,dec_f_state_c,attention_weights,context_vectors
    
    
########################################------Decoder model------########################################
class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      #Intialize necessary variables and create an object from the class onestepdecoder
      super().__init__()
      self.out_vocab_size = out_vocab_size
      self.embedding_dim = embedding_dim
      self.input_length = input_length
      self.dec_units = dec_units
      self.score_fun = score_fun
      self.att_units = att_units

    def build(self,input_shape):
      self.onestep_decoder = OneStepDecoder(self.out_vocab_size, self.embedding_dim, self.input_length, self.dec_units ,self.score_fun ,
                                            self.att_units)

    def call(self, input_to_decoder,encoder_output,f_decoder_hidden_state,f_decoder_cell_state,b_decoder_hidden_state,b_decoder_cell_state ):

      all_outputs = tf.TensorArray(tf.float32, size=self.input_length,name="output_array")
      
      for timestep in range(self.input_length):
        output,state_h,state_c,attention_weights,context_vector = self.onestep_decoder(input_to_decoder[:,timestep:timestep+1],encoder_output,
                                                                                       f_decoder_hidden_state,f_decoder_cell_state,b_decoder_hidden_state,b_decoder_cell_state)
        all_outputs = all_outputs.write(timestep,output)

      all_outputs = tf.transpose(all_outputs.stack(),[1,0,2])
      
      return all_outputs
  
########################################------encoder_decoder model------########################################
class encoder_decoder(tf.keras.Model):
    def __init__(self,out_vocab_size,inp_vocab_size,embedding_dim,embedding_size,in_input_length,tar_input_length,dec_units,lstm_size,att_units,batch_size):
        super().__init__()
        #Intialize objects from encoder decoder
        self.out_vocab_size = out_vocab_size
        self.inp_vocab_size = inp_vocab_size

        self.embedding_dim_target = embedding_dim
        self.embedding_dim_input = embedding_size
        self.in_input_length = in_input_length
        self.tar_input_length = tar_input_length

        self.dec_lstm_size = dec_units 
        self.enc_lstm_size = lstm_size

        self.att_units = att_units
        self.batch_size = batch_size

    def build(self,input_shape):
        self.encoder = Encoder(self.inp_vocab_size,self.embedding_dim_input,self.enc_lstm_size,self.in_input_length)
        self.decoder = Decoder(self.out_vocab_size,self.embedding_dim_target, self.tar_input_length, self.dec_lstm_size ,'general' ,self.att_units)
    
    def call(self,data):
        input_sequence, target_sequence = data[0],data[1]
        # print(input_sequence.shape)
        encoder_initial_state = self.encoder.initialize_states_bidirectional(self.batch_size)
        # print(len(encoder_initial_state))
        encoder_output,f_encoder_state_h,f_encoder_state_c,b_encoder_state_h,b_encoder_state_c = self.encoder(input_sequence,encoder_initial_state)
        decoder_output = self.decoder(target_sequence,encoder_output,f_encoder_state_h,f_encoder_state_c,b_encoder_state_h,b_encoder_state_c)
        return decoder_output


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def accuracy(real,pred):
    pred_val = K.cast(K.argmax(pred,axis=-1),dtype='float32')
    real_val = K.cast(K.equal(real,pred_val),dtype='float32')

    mask = K.cast(K.greater(real,0),dtype='float32')
    n_correct = K.sum(mask*real_val)
    n_total = K.sum(mask)

    return n_correct/n_total

def load_weights():
    """======================================================LOADING======================================================"""
    # Dataset
    with open('dataset/30_length/train.pickle', 'rb') as handle:
        train = pickle.load(handle)

    with open('dataset/30_length/validation.pickle', 'rb') as handle:
        validation = pickle.load(handle)

    # Tokenizer
    with open('tokenizer/30_tokenizer_eng.pickle', 'rb') as handle:
        tokenizer_eng = pickle.load(handle)

    with open('tokenizer/30_tokenizer_ass.pickle', 'rb') as handle:
        tokenizer_ass = pickle.load(handle)

    # Vocab Size
    vocab_size_ass = len(tokenizer_ass.word_index.keys())
    vocab_size_eng = len(tokenizer_eng.word_index.keys())
    
    return train,validation,tokenizer_eng,tokenizer_ass,vocab_size_ass,vocab_size_eng

def main():
    train,validation,tokenizer_eng,tokenizer_ass,vocab_size_ass,vocab_size_eng = load_weights()
    in_input_length = 30
    tar_input_length = 30
    inp_vocab_size = vocab_size_ass
    out_vocab_size = vocab_size_eng

    dec_units = 128
    lstm_size = 128
    att_units = 256
    batch_size = 32
    embedding_dim = 300
    embedding_size = 300

    train_dataset = Dataset(train, tokenizer_ass, tokenizer_eng, in_input_length)
    test_dataset  = Dataset(validation, tokenizer_ass, tokenizer_eng, in_input_length)

    train_dataloader = Dataloder(train_dataset, batch_size)
    test_dataloader = Dataloder(test_dataset, batch_size)


    print(train_dataloader[0][0][0].shape, train_dataloader[0][0][1].shape, train_dataloader[0][1].shape)
    
    model = encoder_decoder(out_vocab_size,inp_vocab_size,embedding_dim,embedding_size,in_input_length,tar_input_length,dec_units,lstm_size,att_units,batch_size)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss=loss_function,metrics=[accuracy])
    
    # train_steps=train.shape[0]//32
    # valid_steps=validation.shape[0]//32
    model.fit(train_dataloader, steps_per_epoch=10, epochs=1,verbose=1, validation_data=train_dataloader, validation_steps=1)
    
    model.load_weights('models/bi_directional_concat_256_batch_160_epoch_30_length_ass_eng_nmt_weights.h5')
    model.fit(train_dataloader, steps_per_epoch=10, epochs=1,verbose=1, validation_data=train_dataloader, validation_steps=1)
    model.summary()
    
    return model,tokenizer_eng,tokenizer_ass,in_input_length
# if __name__=="__main__":
#     main()