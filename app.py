import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from function_viz import Viz as viz
import seaborn as sns
from function_viz import Viz as viz
from function_preprocess import Preprocess as pr
from function_model import models as md

import os
import streamlit as st
import streamlit.components.v1 as components
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import acf
from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd 
import numpy as np

df = pd.read_csv("prog_conso_2010_2020.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d %H:%M')
df = df.set_index('Date')
df = df.asfreq('H')
df = df.sort_index()

pred_ann1 = pd.read_csv("dt_ann1_step1.csv")
pred_simul_ann1 = pd.read_csv("pred_simul_ann1.csv")

pred_lgbm =  pd.read_csv("dt_pred_lgbm1.csv")
pred_cat = pd.read_csv("dt_pred_catboost.csv")
pred_xgb =  pd.read_csv("dt_pred_xgb1.csv")
# decomposition

dd= df.resample('m').sum()
result = seasonal_decompose(dd.loc[:"2018-01-01 00:00:00,'Preal])
### nbre de lags importants
def lag_importance(df ,seuil):
    
    import numpy as np
   
    acf_val = acf(df['Preal'])

    ind_lag = np.where(acf_val > seuil)[0]
    return(ind_lag)


def main():
    menu = ["analyse","sarimax","xgboost","catboost","ann","lstm"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "analyse" :
        col1,col2 = st.columns([3,3])
        col3,col4 = st.columns([3,3])
        col5,col6 = st.columns([3,2])
        
        with col1:
            st.info("decomposition du dataset")
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
            result.observed.plot(ax=ax1)
            ax1.set_ylabel('Observé')
            result.trend.plot(ax=ax2)
            ax2.set_ylabel('Tendance')
            result.seasonal.plot(ax=ax3)
            ax3.set_ylabel('Saisonnalité')
            result.resid.plot(ax=ax4)
            ax4.set_ylabel('Résiduel')
            plt.tight_layout()
            st.pyplot(fig) 
            
            result.plot()    
        with col2:
            st.info("autocorrelation acf")
            st.pyplot(plot_acf(df.Preal))
            
        with col3:
            st.info("autocorrelation pacf")
            st.pyplot(plot_pacf(df.Preal))
        with col4:
            st.info("distribution par semaine")
            fig, ax = plt.subplots(figsize=(10, 7))
            dfcc = df.copy()
            dfcc['day_of_week'] = df.index.day_of_week + 1
            dfcc.boxplot(column='Preal', by='day_of_week', ax=ax)
            dfcc.groupby('day_of_week')['Preal'].median().plot(style='o-', linewidth=0.8, ax=ax)
            ax.set_ylabel('Preal')
            st.pyplot(fig) 
           
        with col5: 
            st.info("distribution par heure")     
            fig, ax = plt.subplots(figsize=(10, 10))
            
            df.boxplot(column='Preal', by='hour', ax=ax)
            df.groupby('hour')['Preal'].median().plot(style='o-', linewidth=0.8, ax=ax)
            ax.set_ylabel('Preal')
            st.pyplot(fig)
            
        with col6: 
            seuil = st.slider('', 0.0, 1.0, 0.8, step=0.05)
            lags = lag_importance(df,seuil)
            st.write("Lags importants ", lags)
            
       

       
    elif choice == "sarimax" :
        st.subheader("sarimax")
        col1,col2 = st.columns([5,1])
        col3,col4 = st.columns([5,1])
        
        with col1:
            st.info("prediction avec le test_set")
                    
        with col2:
            st.info("erreur")
            
        with col3:
            st.info("prediction 1ere sem 2021")
            
        with col4:
            st.info("erreur 2")
        
    elif choice == "xgboost" :
        st.subheader("xgboost")
        col1,col2 = st.columns([4,1])
        col3,col4 = st.columns([4,1])
        
        with col1:
            st.info("prediction avec le test_set")
            md.graph_compare_interval(pred_xgb)
                    
        with col2:
            st.info("erreur")
            md.mean_absolute_errors(pred_xgb)
            md.r_mean_squared_errors(pred_xgb)
            md.mean_absolute_percentage_error(pred_xgb.Preal,pred_xgb.pred)
            md.compare_pred(pred_xgb)
            md.coverage(pred_xgb)
        
            
        with col3:
            st.info("donnees")
            st.write(pred_xgb)
            
        with col4:
            st.info("parametres")
            
            st.write(" learning_rate = 0.1, max_depth = 5, n_estimators = 300")
            st.write("lags =[ 48,  72,  96, 120, 143, 144, 145, 166, 167, 168]")
            st.write("variables : ['year' ,'day','month','day_of_week', 'hour_sin', 'hour_cos'] " )
                
            
    elif choice == "catboost" :
        st.subheader("ridge")
        col1,col2 = st.columns([4,1])
        col3,col4 = st.columns([4,1])
        
        with col1:
            st.info("prediction avec le test_set")
            md.graph_compare_interval(pred_cat)      
        with col2:
            st.info("erreur")
            md.mean_absolute_errors(pred_cat)
            md.r_mean_squared_errors(pred_cat)
            md.mean_absolute_percentage_error(pred_cat.Preal,pred_cat.pred)
            md.compare_pred(pred_cat)
            md.coverage(pred_cat)
        
        with col3:
           
            st.info("donnees")
            st.write(pred_cat)
        with col4:
            st.info("erreur 2")
      
    elif choice == "ann" :
        st.subheader("ann")
        col1,col2 = st.columns([9,3])
        col3,col4 = st.columns([9,3])
        col5,col6 = st.columns([3,3])
        
        
        with col1:
            st.info("prediction avec le test_set")
           
            md.graph_compare_all(pred_ann1)
            
        with col2:
            st.info("erreur")
            md.compare_pred(pred_ann1)
            md.mean_absolute_errors(pred_ann1)
            md.mean_squared_errors(pred_ann1)
            md.r_mean_squared_errors(pred_ann1)
            md.mean_absolute_percentage_error(pred_ann1.Preal,pred_ann1.pred)
            
        with col3:
            st.info("prediction 1ere sem 2021")
            md.graph_compare(pred_simul_ann1)
          
        with col4:
            st.info("erreur 2")
            md.mean_absolute_errors(pred_simul_ann1)
            md.mean_squared_errors(pred_simul_ann1)
            md.r_mean_squared_errors(pred_simul_ann1)
            md.mean_absolute_percentage_error(pred_simul_ann1.Preal,pred_simul_ann1.pred)
            
        with col5:
            st.info("remarques et conclusion")
            st.write("le model ANN avec 2 couches cachees  fait des predictions avec un pas de 1 en utilisant les 24h precedentes.Pour le test sur les donnees de 2021 nous utilisons que les predictions et non les valeurs reelles.On obtient des resultats moyens.")

        with col6:
            st.info("parametres")
            st.write("batch_size : 50 ")
            st.write("epochs : 50")
            st.write("nbre neurons 1ere couche : 100")
            st.write("nbre neurons 2eme couche : 150")


    else:
        st.subheader("lstm")
        col1,col2 = st.columns([4,1])
        col3,col4 = st.columns([4,1])
       
        
        with col1:
            st.info("prediction avec le test_set")
            md.graph_compare_interval(pred_lgbm)  
        with col2:
            st.info("erreur")
            md.mean_absolute_errors(pred_lgbm)
            md.r_mean_squared_errors(pred_lgbm)
            md.mean_absolute_percentage_error(pred_lgbm.Preal,pred_lgbm.pred)
            md.compare_pred(pred_lgbm)
            md.coverage(pred_lgbm)
        
        with col3:
           
            st.info("donnees")
            st.write(pred_lgbm)
            
            
        with col4:
            st.info("parametres")
            
            st.write(" learning_rate = 0.1, max_depth = 7, n_estimators = 300")
            st.write("lags =[ 48,  72,  96, 120, 143, 144, 145, 166, 167, 168]")
            st.write("variables : ['year' ,'day','month','day_of_week', 'hour_sin', 'hour_cos'] " )
                
       
if __name__ == '__main__':
    main()
    
