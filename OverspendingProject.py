import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import json
import datetime
import pandas as pd
import plotly
import io
import os
import copy


try:
    import cPickle as pickle
except ImportError:
    import pickle

def load_pickle_files():
    ## TESTING WITH RANDOM FOREST  
    ngram_size = 1

    list_congressionaldistrict = None
    list_contractingofficeid = None
    list_pl_perf_district = None
    list_productorservicecode = None
    list_psc_cat = None
    list_reasonformodification = None
    list_vendor_cd = None
    classifier = None
    

def classify(contract):
    global classifier
    classifier = None
    #review_counts = loaded_vectorizer.fit_transform([contract]).toarray()
    predictions = classifier.predict(review_counts)
    return predictions
  
def create_service_request(contract):
    return contract

## ------------------ Feature Engineering & Data Cleaning --------------------------------------------------
def calculate_days(signeddate, calcDate):
    import datetime
    from datetime import date
    signeddate = datetime.datetime.strptime(signeddate, '%m/%d/%Y').date()
    calcDate = datetime.datetime.strptime(calcDate, '%m/%d/%Y').date()
    estimated = ((calcDate - signeddate))
    return estimated.days

def convert_categorical(cat_col, cat_list):
    if (cat_col != ''):
        if (cat_col in cat_list):
            result = cat_list.index(cat_col)
        else:
            result = '-1'
    else:
        result = '-1'
    return result

def split_convert_categorical(cat_col, cat_list):
    if (cat_col != ''):
        cat_col = cat_col.split(':')[0]
        if (cat_col in cat_list):
            result = cat_list.index(cat_col)
        else:
            result = '-1'
    else:
        result = '-1'
    return result

def convert_numerical(num_conv):
    import numpy as np
    if (num_conv != np.nan):
        result = int(num_conv)
    else:
        result = 0
    return result

def convertOneRowOfData(row):
    import datetime
    
    row[0] = split_convert_categorical(row[0], list_reasonformodification)
   
    row[13] = calculate_days(row[1], row[2])
    row[14] = calculate_days(row[1], row[3])
    
    row[4] = split_convert_categorical(row[4], list_contractingofficeid)
    row[5] = split_convert_categorical(row[5], list_productorservicecode)
    row[6] = convert_categorical(row[6], list_pl_perf_district)
    row[7] = convert_categorical(row[7], list_psc_cat)
    row[8] =  convert_numerical(row[8]) 
    row[9] = convert_numerical(row[9]) 
    row[10] = convert_categorical(row[10],list_congressionaldistrict)
    row[11] =  convert_categorical(row[11], list_vendor_cd)
    row[12] = convert_numerical(row[12])
    
    return(row)

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

## ------------------ Load the Model and Lists  --------------------------------------------------

# Load the Model
from sklearn.externals import joblib

clf = joblib.load('overSpendingClassifier.pkl')

list_contractingofficeid = joblib.load('list_contractingofficeid.pkl')
list_pl_perf_district = joblib.load('list_pl_perf_district.pkl')
list_productorservicecode = joblib.load('list_productorservicecode.pkl')
list_psc_cat = joblib.load('list_psc_cat.pkl')
list_reasonformodification = joblib.load('list_reasonformodification.pkl')
list_congressionaldistrict = joblib.load('list_congressionaldistrict.pkl')
list_vendor_cd = joblib.load('list_vendor_cd.pkl')


# ------------------ Read in the CSV file --------------------------------------------------

filename = 'Oct01_2016_to_Sep30_2017'
spend_df = pd.read_csv('./Oct01_2016_to_Sep30_2017.csv', low_memory=False)
spend_df = spend_df.sample(300)
spend_df.rename(columns={'contractingofficeagencyid': 'Agency Id', 'vendorname': 'Vendor' ,  'reasonnotcompeted':'Reason No Comp',
                         'signeddate':'SignedDate', 'descriptionofcontractrequirement':'Description','baseandalloptionsvalue':'Amount',
                         'congressionaldistrict':'District','unique_transaction_id':'TransId','original_estimated_days':'EstimatedDuration',
                         'currentcompletiondate':'CompletionDate'}, inplace=True)

# ----------------------  DATA CLEANING --------------------------------------

spend_df['current_estimated_days'] = '1/1/1900'
spend_df['EstimatedDuration'] = '1/1/1900'

spend_df.loc[spend_df.CompletionDate.isnull(), 'CompletionDate'] = '1/1/1900'
spend_df.loc[spend_df.ultimatecompletiondate.isnull(), 'ultimatecompletiondate'] = '1/1/1900'
spend_df.loc[spend_df.SignedDate.isnull(), 'SignedDate'] = '1/1/1900'

#categoricalColumns = ['reasonformodification', 'contractingofficeid','productorservicecode',
 #                     'placeofperformancecongressionaldistrict','psc_cat', 'congressionaldistrict', 'vendor_cd']

categoricalColumns = ['reasonformodification', 'contractingofficeid','productorservicecode',
                      'placeofperformancecongressionaldistrict','psc_cat', 'District', 'vendor_cd']

spend_df[categoricalColumns] = spend_df[categoricalColumns].fillna('NONE')

numericalColumns = ['annualrevenue','numberofemployees','numberofoffersreceived']

for col in numericalColumns:
   spend_df[col] = spend_df[col].apply(lambda x: int(x) if x == x else 0)

#selected_cols = ['reasonformodification','signeddate', 'currentcompletiondate','ultimatecompletiondate','contractingofficeid',
 #               'productorservicecode','placeofperformancecongressionaldistrict','psc_cat','annualrevenue',
  #              'numberofoffersreceived','congressionaldistrict','vendor_cd','numberofemployees','current_estimated_days',
   #             'original_estimated_days','baseandalloptionsvalue']

selected_cols = ['reasonformodification','SignedDate', 'CompletionDate','ultimatecompletiondate','contractingofficeid',
                'productorservicecode','placeofperformancecongressionaldistrict','psc_cat','annualrevenue',
                'numberofoffersreceived','District','vendor_cd','numberofemployees','current_estimated_days',
                'EstimatedDuration','Amount']

spend_df[selected_cols] = spend_df[selected_cols].apply(convertOneRowOfData, axis=1) 
   

# ---------------------- DEFINE X AND Y AND PREDICT --------------------------------------

#   Create the prediction Column : 1 if it is an Overspending observation ( baseandalloptionsvalue < 0 )

spend_df['Prediction'] = spend_df['Amount'].map(lambda x: 1 if x < 0 else 0)

feature_cols = ['reasonformodification','current_estimated_days','EstimatedDuration','contractingofficeid',
                'productorservicecode','placeofperformancecongressionaldistrict','psc_cat','annualrevenue',
                'numberofoffersreceived','District','vendor_cd','numberofemployees']

#feature_cols = ['reasonformodification','current_estimated_days','Estimated Duration','Agency Id',
##                'productorservicecode','placeofperformancecongressionaldistrict','psc_cat','annualrevenue',
 #               'numberofoffersreceived','District','vendor_cd','numberofemployees']

X = spend_df[feature_cols]

y = spend_df.Prediction

show_cols = ['Prediction','Agency Id','Vendor','Reason No Comp','SignedDate','Description',
             'Amount','District','TransId','EstimatedDuration']

#show_cols = ['Prediction','contractingofficeagencyid','vendorname','reasonnotcompeted','signeddate','descriptionofcontractrequirement',
 #            'baseandalloptionsvalue','congressionaldistrict','unique_transaction_id','currentcompletiondate','original_estimated_days']


# -------------- CREATE A NEW DATAFRAME THAT ONLY CONTAINS  --------------------------------------------  
text_options = []

spend_over = spend_df

# ------------------------- PREDICT  --------------------------------------------

y_pred_class = clf.predict(X)

overSpend_count = spend_over[spend_over.Prediction == 1].count()[0]
noOverSpend_count = spend_over[spend_over.Prediction == 0].count()[0]

print (overSpend_count, noOverSpend_count)
    
##--------------------END OF DATAFRAME CREATION-----------------------------------------------

layout = dict(
    autosize=True,
    height=500,
    font=dict(color='#CCCCCC'),
    titlefont=dict(color='#CCCCCC', size='14'),
    margin=dict(l=35, r=35, b=35, t=45),
    hovermode="closest",
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    legend=dict(font=dict(size=10), orientation='h'),
    title='Satellite Overview'
)


## ------------------ START MAKE PIE FIGURE --------------------------------------------------
def make_pie_figure():
    layout_pie = copy.deepcopy(layout)

    global overSpend_count, noOverSpend_count
    
    data = [
        dict(
            type='pie',
            labels=['Over Spending', 'Normal'],
            values=[overSpend_count, noOverSpend_count],
            name='Ratings Breakdown',
            text=['Over Spending', 'Regular'],
            hoverinfo="text+value+percent",
            textinfo="label+percent+name",
            hole=0.5,
            marker=dict(colors=['#fac1b7', '#a9bb95'])
        ),
    ]
    layout_pie['title'] = ''
    layout_pie['font'] = dict(color='#000000')
    layout_pie['legend'] = dict(
        font=dict(color='#000000', size='10'),
        orientation='h',
        bgcolor='rgba(0, 0, 0, 0)'
    )

    figure = dict(data=data, layout=layout_pie)
    return figure

## ------------------ END MAKE PIE FIGURE --------------------------------------------------

app = dash.Dash()
    
app.layout = html.Div(
    [
        html.Div(
        [
                html.H2('Texas Government Budgeting Office | Over Spending Analysis', className='eight columns'),
               html.Div(
                  [
                       html.Img(
                        src='https://content.efilecabinet.com/wp-content/uploads/america-paperless.jpg',
                        style={'height': '300','width': '400','float': 'right','position': 'relative'},
                        ),
                html.Div(
                [
                    html.P('Number of Contracts: ' + str(spend_over.shape[0]),id='contracts_total',className='two columns'),
                    html.P('',id='total_overSpending',className='one column',style={'text-align': 'center'}),
                    html.P('',id='total_noOverSpending',className='one column',style={'text-align': 'right'}),
                    html.Div(
                      [
                           html.P('Date: ' + str(datetime.date.today()),id='today_date',style={'text-align': 'right'}),
                      ],
                      className='four columns',   
                      ),
                ],
                ),
                html.Div(
                [
                    html.P('File Name:'),
                    dcc.Textarea(placeholder='Oct01_2016_to_Sep30_2017',value='', style={'width': '25%'}),
                    html.Button(id='submit_button', n_clicks=0, children='Predict'),
                    html.Div(id='output_prediction'),
                 ],
                 className='eight columns',
                 style={'margin-top': '10'}
                 ),

                ],
                ),
             ],
         ),

         html.Div(
         [
            html.Div(
            [
           
             ],
             className='eight columns',
             style={'margin-top': '10'}
             ),
         ],
         className='row'
         ),
    
         html.Div(
         [
            html.Div(
            [
              dt.DataTable(rows=spend_df[show_cols].to_dict('records')),      
             ],
             className='eight columns',
             ),
             html.Div(
                  [
                      dcc.Graph(id='pie_graph', figure=make_pie_figure())
                  ],
                  className='four columns',
              )
         ],
         className='row'
         ),
])


# ----------------------------------------------------------

@app.callback(
    Output('output_prediction', 'children'),
    [Input('submit_button', 'n_clicks')]#,
  #  [State('contract_dropdown', 'value')]
)

def update_output_div(n_clicks, contract):
    message = ''
    if (n_clicks):
        prediction = classify(contract)
    return message

app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})

if __name__ == '__main__':
    app.run_server(debug=True)
