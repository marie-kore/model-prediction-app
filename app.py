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


# decomposition

dd= df.resample('m').sum()
result = seasonal_decompose(dd.Preal)
### nbre de lags importants
def lag_importance(df ,seuil):
    
    import numpy as np
   
    acf_val = acf(df['Preal'])

    ind_lag = np.where(acf_val > seuil)[0]
    return(ind_lag)


def main():
    menu = ["analyse","sarimax","autoArima","ridge","ann","lstm"]
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
            seuil = st.slider('', 0.0, 1.0, 0.8, step=0.1)
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
        
    elif choice == "autoArima" :
        st.subheader("autoArima")
        col1,col2 = st.columns([4,1])
        col3,col4 = st.columns([4,1])
        
        with col1:
            st.info("prediction avec le test_set")
                    
        with col2:
            st.info("erreur")
            
        with col3:
            st.info("prediction 1ere sem 2021")
            
        with col4:
            st.info("erreur 2")
    
    elif choice == "ridge" :
        st.subheader("ridge")
        col1,col2 = st.columns([4,1])
        col3,col4 = st.columns([4,1])
        
        with col1:
            st.info("prediction avec le test_set")
                    
        with col2:
            st.info("erreur")
            
        with col3:
            st.info("prediction 1ere sem 2021")
            
        with col4:
            st.info("erreur 2")
      
    elif choice == "ann" :
        st.subheader("ann")
        col1,col2 = st.columns([4,1])
        col3,col4 = st.columns([4,1])
        
        with col1:
            st.info("prediction avec le test_set")
           
            md.graph_compare_all(pred_ann1)
            
        with col2:
            st.info("erreur")
            md.compare_pred(pred_ann1)
            md.mean_squared_errors(pred_ann1)
            md.mean_absolute_percentage_error(pred_ann1.Preal,pred_ann1.pred)
            
        with col3:
            st.info("prediction 1ere sem 2021")
            
        with col4:
            st.info("erreur 2")
        
    else:
        st.subheader("lstm")
        col1,col2 = st.columns([4,1])
        col3,col4 = st.columns([4,1])
        
        with col1:
            st.info("prediction avec le test_set")
                    
        with col2:
            st.info("erreur")
            
        with col3:
            st.info("prediction 1ere sem 2021")
            
        with col4:
            st.info("erreur 2")
        
 
if __name__ == '__main__':
    main()
    