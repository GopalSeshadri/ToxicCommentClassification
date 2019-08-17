import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
import dash
import dash_core_components as core
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import sys
from preprocess import Preprocess
import pickle
from keras.models import model_from_json
from keras import backend as K

app = dash.Dash()

toxic_label_dict = {'label' : ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], 'p' : [0, 0, 0, 0, 0, 0]}
toxic_label_df =  pd.DataFrame(toxic_label_dict)

app.layout = html.Div([
    html.Div([
        html.H1('Toxic Sentence Classification', style = {'fontFamily' : 'Helvetica',
                                                        'textAlign' : 'center',
                                                        'width' : '100%'})
    ], style = {'display' : 'flex'}),

    html.Div([
        html.Div([
            html.H3('Enter Text',
                style = {'font' : {'size' : 16},
                                'display' : 'inline-block',
                                'float' : 'left'}),

            core.Textarea(id = 'text-area',
                placeholder = 'Enter something here ...',
                style = {'width': '600px',
                        'height' : '200px',
                        'display' : 'inline-block',
                        'float' : 'left'}
            )
        ], style = {'width' : '100%'}),

        html.Div([
            html.Button(id = 'submit-button', children = 'Submit',
                        style = {'width' : '80px',
                                'height' : '30px'})
        ], style = {'width' : '100%',
                    'paddingLeft' : '50px'})

    ], style = {'width' : '60%',
                'height' : '80%',
                'display' : 'flex',
                'float' : 'left',
                'paddingLeft' : '50px'}),

    html.Div([
        dash_table.DataTable(
            id = 'table',
            columns = [{'name' : i, 'id' : i} for i in toxic_label_df.columns],
            data = toxic_label_df.to_dict('records'),
            hidden_columns = ['p'],
            style_data_conditional=[{
                'if': {'column_id': 'p',
                'filter_query': '{p} eq 1'},
                'backgroundColor': '#3D9970',
                'color': 'white'
            }])
    ], style = {'width' : '25%',
                'display' : 'inline-block',
                'float' : 'left',
                'paddingLeft' : '50px'})
], style = {'fontFamily' : 'Helvetica',
            'width' : '100%',
            'height' : '100%'})

@app.callback([Output(component_id = 'table', component_property = 'data'),
            Output(component_id = 'table', component_property = 'style_data_conditional')],
            [Input(component_id = 'submit-button', component_property = 'n_clicks')],
            [State(component_id = 'text-area', component_property = 'value')])
def affect_table(n_clicks, text):
    '''
    This function is a callback that takes in an number of clicks and text area as input and
    returns dictionary for data attribute of dash table

    Parameters:
    n_clicks (int) : Number of Clicks
    text (str) : The text entered in the text box

    Returns:
    output_dict (dict) : A dictionary for data attribute
    '''
    if text is None:
        text = 'Enter something here ...'
        
    MAX_SEQ_LENGTH = 200
    with open('Models/tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)

    text = Preprocess.preprocessText(text)
    text = [list(np.array([each for each in text.split(' ')]).ravel())]
    text = tokenizer.texts_to_sequences(text)
    padded_text = Preprocess.padSequences(text, MAX_SEQ_LENGTH)

    json_file = open('Models/multicnn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Models/multicnn_model.h5")
    prediction = loaded_model.predict(padded_text)
    prediction = [1 if p >= 0.5 else 0 for p in prediction[0]]

    output_dict = {'label' : ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], 'p' : prediction}
    output_df =  pd.DataFrame(output_dict)

    K.clear_session()

    style_data_conditional = [{
        'if': {'column_id': 'label',
        'filter_query': '{p} eq 1'},
        'backgroundColor': '#3D9970',
        'color': 'white'
    },
    {
        "if": {'column_id': 'label',
        'filter_query': '{p} eq 0'},
        'backgroundColor': '#ffffff',
        'color': 'black'
    }]

    return output_df.to_dict('records'), style_data_conditional



if __name__ == '__main__':
    app.run_server()
