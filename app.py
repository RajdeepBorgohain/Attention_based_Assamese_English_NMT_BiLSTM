import streamlit as st
import pandas as pd
import datetime
import numpy as np
import datetime
import model
import inference


# Global params
if 'model' not in st.session_state:
    loaded_model,tokenizer_eng,tokenizer_ass,in_input_length = model.main()
    st.session_state['model'] = loaded_model
    st.session_state['tokenizer_eng'] = tokenizer_eng
    st.session_state['tokenizer_ass'] = tokenizer_ass
    st.session_state['in_input_length'] = in_input_length
    
# st.write(st.session_state)

# def model_loading():
#     return model.main()

def show_information():
    # Show Information about the selected Stock
    st.header('Now translate everything into English!')
    # st.caption("Analyzing data from 2015 to 2021")
    # st.text("1) There is a 60% chance of gap up opening in any random trade in Reliance 😮 ")
    # st.text("2) 1% of the gap up is more than Rs:15.00 i.e more quantity == more profit😇")
    # st.text("3) Median, Q3 or 75th percentile have increased from 2015(1.8) to 2021(11.55)💰")

def select_text():
    # Select the Suggested Assamese Text
    option = st.selectbox(
     'Select these suggested Assamese Sentences',
     ('সমগ্ৰ দেশজুৰি ব্যাপক চৰ্চা হৈছিল উক্ত ঘটনাৰ ',
      'দৃষ্টান্ত ব্যৱহাৰ কৰাৰ সম্পৰ্কে আমি যীচুৰ পৰা কি শিকিব পাৰোঁ ',
      'তেওঁ যি ইচ্ছা তাকে কৰিব নোৱাৰে '))

    st.write('You have selected suggested text')
    
    title = st.text_input('Assamese Text Input', option)
    # st.write('Your Assamese Text', title)
    
    return title
    # return selected_date

# @st.cache
# def prepare_data_for_selected_date():
#     df = pd.read_csv("dataset/reliance_30min.csv")
#     df = helper.format_date(df)
#     df = helper.replace_vol(df)
#     df = helper.feature_main(df)
#     df.to_csv('dataset/processed_reliance30m.csv')
    
#     return df

# @st.cache
# def show_result(sentence):
#   pass   


# def show_prediction_result(prepared_data):
#     model = all_model.load_model()
#     result = all_model.prediction(model,prepared_data)
    
#     return result
    



def main():
    st.title('📚Assamese to English Translator🤓')
    show_information()
    text = select_text()
    if st.button('Translate'):
        result = inference.infer(st.session_state['model'],text,st.session_state['tokenizer_ass'],
                                 st.session_state['tokenizer_eng'],st.session_state['in_input_length'])
        st.caption('Your Assamese translated text')
        st.text(result[:-6])
      
if __name__ == "__main__":
    main()