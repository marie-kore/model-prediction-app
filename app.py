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
pred_lasso = pd.read_csv("pred_lasso1.csv")
pred_lgbm =  pd.read_csv("dt_pred_lgbm1.csv")
pred_cat = pd.read_csv("dt_pred_catboost.csv")
pred_xgb =  pd.read_csv("dt_pred_xgb1.csv")
pred_elas = pd.read_csv("pred_elas1.csv")
# decomposition

dd= df.resample('D').sum()
dd  = dd.iloc[1:100]
result = seasonal_decompose(dd.Preal)
### nbre de lags importants
def lag_importance(df ,seuil):
    
    import numpy as np
   
    acf_val = acf(df['Preal'])

    ind_lag = np.where(acf_val > seuil)[0]
    return(ind_lag)


def main():
    menu = ["analyse","lasso","elastic","xgboost","catboost","ann","lgbm"]
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
            
       

       
    elif choice == "lasso" :
        st.subheader("lasso")
        col1,col2 = st.columns([5,1])
        col3,col4 = st.columns([5,1])
        
        with col1:
            st.info("prediction avec le test_set")
            md.graph_compare_interval(pred_lasso)  
        with col2:
            st.info("erreur")
            md.mean_absolute_errors(pred_lasso)
            md.r_mean_squared_errors(pred_lasso)
            md.mean_absolute_percentage_error(pred_lasso.Preal,pred_lasso.pred)
            md.compare_pred(pred_lasso)
            md.coverage(pred_lasso)
        
        with col3:
           
            st.info("donnees")
            st.write(pred_lasso)
            
            
        with col4:
            st.info("parametres")
            st.write("lags =[ 48,  49,  50,  51,  52,  53,  54,  55,56,  57,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,74,  75,  76,  77,  78,  79,  80,  81,  87,  88,  89,  90,  91,92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,123, 124, 125, 126, 127, 128, 129, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]")
            st.write(" max_iter=1000,alpha = 1.0235896477251554e-1")
            
            st.write("variables : ['year' ,'day','month','day_of_week', 'hour_sin', 'hour_cos'] " )
                
    elif choice == "elastic" :
        st.subheader("elastic")
        col1,col2 = st.columns([5,1])
        col3,col4 = st.columns([5,1])
        
        with col1:
            st.info("prediction avec le test_set")
            md.graph_compare_interval(pred_elas)  
        with col2:
            st.info("erreur")
            md.mean_absolute_errors(pred_elas)
            md.r_mean_squared_errors(pred_elas)
            md.mean_absolute_percentage_error(pred_elas.Preal,pred_elas.pred)
            md.compare_pred(pred_elas)
            md.coverage(pred_elas)
        
        with col3:
           
            st.info("donnees")
            st.write(pred_elas)
            
            
        with col4:
            st.info("parametres")
            st.write("lags =[  48,  49,  50,  70,  71,  72,  73,  74,  94,  95,  96,  97,98, 118, 119, 120, 121, 122, 141, 142, 143, 144, 145, 146, 147,165, 166, 167, 168]")
            st.write(" max_iter=1000,alpha=0.2592943797404667, l1_ratio=1.0")
            
            st.write("variables : ['year' ,'day','month','day_of_week', 'hour_sin', 'hour_cos'] " )
                




    
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
            
            st.write("learning_rate=0.1,max_depth=5,n_estimators =300")
            st.write("lags =[    48,  49,  71,72,  73,  95,  96,  97, 119, 120, 121, 142, 143, 144, 145, 146,165, 166, 167, 168]")
            st.write("variables : ['year' ,'day','month','day_of_week', 'hour_sin', 'hour_cos'] " )
                
            
    elif choice == "catboost" :
        st.subheader("catboost")
        col1,col2 = st.columns([4,1])
        col3,col4 = st.columns([4,1])
        
        with col1:
            st.info("prediction avec le test_set")
            md.graph_compare(pred_cat)      
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
        st.subheader("lgbm")
        col1,col2 = st.columns([4,1])
        col3,col4 = st.columns([4,1])
       
        
        with col1:
            st.info("prediction avec le test_set")
            md.graph_compare(pred_lgbm)  
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
    
