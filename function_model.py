import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
class models():
    @staticmethod
    def split_data(df,end_train,feature):

        train  = df.loc [:end_train,] 
        test   = df.iloc[len(train):, ]
        X_train = train[feature]
        y_train = train["Preal"]
        X_test = test[feature]
        y_test = test["Preal"]

        return X_train,y_train,X_test,y_test

        
    def mean_squared_errors(dfpred):
        error =   np.sqrt(mean_squared_error(y_true=dfpred['Preal'],y_pred=dfpred['pred']))
        st.write("rmse",error)

    def mean_absolute_percentage_error(y_true, y_pred): 
    
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mapes =np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        st.write("mape",mapes)


    def df_prediction(df, prediction):
        import pandas as pd
        prediction = pd.DataFrame(prediction)
        prediction['Preal'] = df.loc[prediction.index, 'Preal']
        prediction['Prev'] = df.loc[prediction.index, 'Prev']
        return prediction


    def prediction_ts(df,pred):
    
        
        fig, ax = plt.subplots(figsize=(12, 3.5))
        pred['Preal'].plot(ax=ax, linewidth=2, label='real')
        pred['pred'].plot(linewidth=2, label='prediction', ax=ax)
        pred['Prev'].plot(linewidth=2, label='old prediction', ax=ax)
        ax.set_title('Prediction vs real demand')
        ax.legend();


    def pred_per_month(df,predictions,date_ym):
        
        fig, ax = plt.subplots(figsize=(12, 3.5))
        predictions['Preal'][date_ym].plot(ax=ax, linewidth=2, label='real')
        predictions['pred'][date_ym].plot(linewidth=2, label='prediction', ax=ax)
        predictions['Prev'][date_ym].plot(linewidth=2, label='old prediction', ax=ax)
        ax.set_title('Prediction vs real demand')
        ax.legend();
 
    
    def pred_per_hour(df,predictions,date_ymd):
        
        predictions['Preal'] = df.loc[predictions.index, 'Preal']
        fig, ax = plt.subplots(figsize=(12, 3.5))
        predictions['Preal'][date_ymd].plot(ax=ax, linewidth=2, label='real')
        predictions['pred'][date_ymd].plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title('Prediction vs real demand')
        ax.legend();

    def predict_feedback(df, forecaster, initial_train_size):


        train, test = df[:initial_train_size], df[initial_train_size:]
        forecaster.fit(train) 
        predictions = pd.DataFrame(columns=['pred'])
        
    
        for i in range(len(test)):
            
        
            next_date = test.index[i]
            next_pred = forecaster.predict(test.iloc[i])
            
        
            predictions.loc[next_date] = {'pred': next_pred}
            
            # Réajuster le modèle après chaque 48 prédictions
            if (i+1) % 48 == 0:
                train = pd.concat([train, predictions.iloc[-48:]])
                forecaster.fit(train)
            
        
        return pd.concat([train, predictions])



## 
    def graph_compare_all(predcp):
        import plotly.express as px
        import plotly.graph_objects as go
        predcp = predcp.reset_index()
       
                
        fig = px.line(
            data_frame = predcp,
            x      = 'index',
            y      = 'Preal',
            width  = 500,
            height = 500
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1day", step="day", stepmode="todate"),
                    dict(count=1, label="1m", step="month", stepmode="todate"),
                    dict(count=3, label="3m", step="month", stepmode="todate"),
                    dict(count=6, label="6m", step="month", stepmode="todate"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                

                    dict(step="all")
                ])
            )
        )


        fig.add_trace(
            go.Scatter(
                x=predcp['index'], 
                y=predcp['pred'], 
                mode='lines',
                name='pred',
                marker=dict(color='red')
            )
        )

    
        fig.add_trace(
            go.Scatter(
                x=predcp['index'], 
                y=predcp['Prev'], 
                mode='lines',
                name='Prev',
                marker=dict(color='green')
            )
        )

      
        fig.update_layout(
           
            xaxis_title='Date',
            yaxis_title='MGW',
           
            width=500,
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )


        st.plotly_chart(fig)

    def graph_compare(predcp):
        import plotly.express as px
        import plotly.graph_objects as go
        predcp = predcp.reset_index()
                
        fig = px.line(
            data_frame = predcp,
            x      = 'index',
            y      = 'Preal',
            title  = 'MGW',
            width  = 1200,
            height = 700
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1day", step="day", stepmode="todate"),
                    dict(count=1, label="1m", step="month", stepmode="todate"),
                    dict(count=3, label="3m", step="month", stepmode="todate"),
                    dict(count=6, label="6m", step="month", stepmode="todate"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                

                    dict(step="all")
                ])
            )
        )


        fig.add_trace(
            go.Scatter(
                x=predcp['index'], 
                y=predcp['pred'], 
                mode='lines',
                name='pred',
                marker=dict(color='red')
            )
        )


        fig.update_layout(
            title='MGW prevu VS realise',
            xaxis_title='Date',
            yaxis_title='MGW',
            height=700,
            width=1200,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )


        fig.show()

    def compare_pred(df):
        df_cp = df.copy()
        df_cp['diff_pred'] = abs(df_cp['pred'] - df_cp['Preal'])
        df_cp['diff_prev'] = abs(df_cp['Prev'] - df_cp['Preal'])

        
        count_pred_closer = sum(df_cp['diff_pred'] < df_cp['diff_prev'])
        count_prev_closer = sum(df_cp['diff_prev'] < df_cp['diff_pred'])


        st.write(" 'pred'  plus proche:", count_pred_closer)
        st.write("'prev' plus proche:", count_prev_closer)
        
        
    def generer_pred_testFinal_ann(data,tab1,tab2,date_end_train):
    
        date_end_train = pd.Timestamp(date_end_train)
        date_end_train  += pd.Timedelta(hours=1)
        heure_depart = date_end_train.strftime('%Y-%m-%d %H:%M:%S')

        heure_depart = pd.Timestamp(heure_depart)
        tab1 = tab1.flatten()
        
        index_dates = pd.date_range(start=heure_depart, periods=len(tab1), freq='H')

        df = pd.DataFrame({'pred': tab1, 'Preal': tab2}, index=index_dates)
        df['Prev'] =  data.loc[df.index, 'Prev']
  
        return(df)


    def lag_importance(df ,seuil):
        
        import numpy as np
        from statsmodels.graphics.tsaplots import acf
        
        acf_val = acf(df['Preal'])
        ind_lag = np.where(acf_val > seuil)[0]
        
        return(ind_lag)

    
    
    

                