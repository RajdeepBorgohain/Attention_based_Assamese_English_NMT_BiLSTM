from tensorflow import argmax,expand_dims,convert_to_tensor
from tensorflow.keras.preprocessing.sequence import pad_sequences
# BRUTE FORCE
def infer(model,sentence,tokenizer_ass,tokenizer_eng,in_input_length):
  encoder_seq = tokenizer_ass.texts_to_sequences([sentence]) # need to pass list of values
  encoder_seq = pad_sequences(encoder_seq, maxlen=in_input_length, dtype='int32', padding='post')
  encoder_seq = convert_to_tensor(encoder_seq)
  initial_state = model.layers[0].initialize_states_bidirectional(batch_size=1)
  encoder_outputs, f_encoder_hidden, f_encoder_cell,b_encoder_hidden, b_encoder_cell = model.layers[0](encoder_seq,initial_state)
  dec_input = expand_dims([tokenizer_eng.word_index['<start>']],0)

  result = ''
  for t in range(30):
    Output, dec_h,dec_c,attention_w,context_vec = model.layers[1].onestep_decoder(dec_input,encoder_outputs,f_encoder_hidden, f_encoder_cell,b_encoder_hidden, b_encoder_cell)
    # result_beam_list = beam_search(Output,k=1)
    # result_beam = result_beam_list[0][0]
    # attention_weights = tf.reshape(attention_w,(-1,))
    predict_id = argmax(Output[0]).numpy()
    result += tokenizer_eng.index_word[predict_id]+' '
    if tokenizer_eng.index_word[predict_id] == '<end>':
      break
    dec_input = expand_dims([predict_id],0)

  
  print(result)
  return result