import pandas as pd 
import os 
class Preprocess():
    @staticmethod
    def load_data(file):
        dossier = "../data"
        path =  os.path.abspath(os.path.join(dossier, file)).replace('\\', '/')
        df = pd.read_csv(path, parse_dates = True,sep=",")
        return df
               



    def is_holiday(df):
        hol = ['2012-01-01', '2012-04-09', '2012-05-01', '2012-05-08', '2012-05-17', '2012-05-28',
                            '2012-07-14', '2012-08-15', '2012-11-01', '2012-11-11', '2012-12-25', '2013-01-01',
                            '2013-04-01', '2013-05-01', '2013-05-08', '2013-05-09', '2013-05-20', '2013-07-14',
                            '2013-08-15', '2013-11-01', '2013-11-11', '2013-12-25', '2014-01-01', '2014-04-21',
                            '2014-05-01', '2014-05-08', '2014-05-29', '2014-06-09', '2014-07-14', '2014-08-15',
                            '2014-11-01', '2014-11-11', '2014-12-25', '2015-01-01', '2015-04-06', '2015-05-01',
                            '2015-05-08', '2015-05-14', '2015-05-25', '2015-07-14', '2015-08-15', '2015-11-01',
                            '2015-11-11', '2015-12-25', '2016-01-01', '2016-03-28', '2016-05-01', '2016-05-05',
                            '2016-05-08', '2016-05-16', '2016-07-14', '2016-08-15', '2016-11-01', '2016-11-11',
                            '2016-12-25', '2017-01-01', '2017-04-17', '2017-05-01', '2017-05-08', '2017-05-25',
                            '2017-06-05', '2017-07-14', '2017-08-15', '2017-11-01', '2017-11-11', '2017-12-25',
                            '2018-01-01', '2018-04-02', '2018-05-01', '2018-05-08', '2018-05-10', '2018-05-21',
                            '2018-07-14', '2018-08-15', '2018-11-01', '2018-11-11', '2018-12-25', '2019-01-01',
                            '2019-04-22', '2019-05-01', '2019-05-08', '2019-05-30', '2019-06-10', '2019-07-14',
                            '2019-08-15', '2019-11-01', '2019-11-11', '2019-12-25', '2020-01-01','2020-04-13',
                            '2020-05-01', '2020-05-08', '2020-05-21', '2020-06-01', '2020-07-14',
                            '2020-08-15', '2020-11-01', '2020-11-11', '2020-12-25', '2021-01-01', '2021-04-05',
                            '2021-05-01', '2021-05-08', '2021-05-13', '2021-05-24', '2021-07-14', '2021-08-15',
                            '2021-11-01', '2021-11-11', '2021-12-25']

        holidays = ['12-31', '01-01', '05-01', '08-07', '12-24', '12-25']
        
        df = df.reset_index()
        df['holidays'] = 0
        
        for index, row in df.iterrows():
        
            month_day = row['Date'].strftime('%m-%d')

           
            if month_day in holidays:
                df.at[index, 'holidays'] = 1

        for index, row in df.iterrows():
        
            month_day = row['Date'].strftime('%Y-%m-%d')

           
            if month_day in holidays:
                df.at[index, 'hol'] = 1     
            
            
        df = df.set_index('Date')
        df = df.asfreq('H')
        df = df.sort_index()
        return(df)
    
    def create_cadr(df):

            df['cadr1'] = (df.index.hour >= 0) & (df.index.hour <= 6).astype(int)
            df['cadr2'] = (df.index.hour >= 7) & (df.index.hour <= 18).astype(int)
            df['cadr3'] = (df.index.hour >= 19) & (df.index.hour <= 23).astype(int)

      
            return df 

    def create_hour_sincos(df):
        import numpy as np
        df['hour_sin'] = np.sin(df['hour'] / 23 * 2 * np.pi)
        df['hour_cos'] = np.cos(df['hour'] / 23 * 2 * np.pi)
        return df 
    
    def int_to_str(df,features):
        for i in features:
            df[i] = df[i].astype("category")


    def create_features(df):
        
        df["hour"] = df.index.hour
        df["day"] = df.index.day
        df["month"] = df.index.month
        df["year"] = df.index.year
        df["weekend"] = df.index.dayofweek.isin([5, 6, 7]).astype(int)
        df =  Preprocess.is_holiday(df)
        df = Preprocess.create_hour_sincos(df)
        df['day_of_week'] = df.index.dayofweek  #si date est pas en index df.dt.dayofweek
        df['day_of_year'] = df.index.dayofyear
        return df