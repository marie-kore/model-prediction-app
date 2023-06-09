import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class  Viz():
    @staticmethod
    def ViewSerie(df):
        fig, ax = plt.subplots(figsize=(17, 6))
        df.Preal.plot(ax=ax, label='', linewidth=1)
    
        ax.set_title('qte electricite')
        ax.legend()
        
    def serie_par_Month(df, start, end):
        mois = (start,end)
        fig = plt.figure(figsize=(12, 6))
        grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)
    
        main_ax = fig.add_subplot(grid[1:3, :])
        zoom_ax = fig.add_subplot(grid[5:, :])
    
        df.Preal.plot(ax=main_ax, c='black', alpha=0.5, linewidth=0.5)
        min_y = min(df.Preal)
        max_y = max(df.Preal)
        main_ax.fill_between(mois, min_y, max_y, facecolor='blue', alpha=0.5, zorder=0)
        main_ax.set_xlabel('')
    
        df.loc[mois[0]: mois[1]].Preal.plot(ax=zoom_ax, color='blue', linewidth=2)
    
        main_ax.set_title(f'qte electricite par an: {df.index.min()}, {df.index.max()}', fontsize=14)
        zoom_ax.set_title(f'qte electricite par mois: {mois}', fontsize=14)
        plt.subplots_adjust(hspace=1)
    
    def serie_per_Period(df, start, end):
    
        df = df.loc[start: end]
        fig, ax = plt.subplots(figsize=(17, 6))
        liste = []
        for year in range(df.index.year.min(), df.index.year.max() + 1):
            year_data = df[df.index.year == year]
            liste.append(year_data.iloc[:])
    
        # Créer les sous-graphiques
        df.Preal.plot(ax=ax, label='', linewidth=1)
    
    
    
    
        
    def serie_per_year(df):
        liste = []
        for year in range(df.index.year.min(), df.index.year.max() + 1):
            year_data = df[df.index.year == year]
            liste.append(year_data.iloc[:])

        # Créer les sous-graphiques
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
        axs = axs.ravel()

        # Tracer les graphes pour chaque année
        for i, year_data in enumerate(liste):
            axs[i].plot(year_data.index, year_data['Preal'])
            axs[i].set_xlabel('Date')
            axs[i].set_ylabel('Preal')
            axs[i].set_title(f'annee {year_data.index.year[0]}')

        # Afficher le graphe
        plt.tight_layout()
        plt.show()
    
    def box_per_year(df,start_year,end_year):
        years = df['year'].unique()

        fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
        sns.boxplot(x='year', y='Preal', data=df.loc[~df.year.isin([start_year, end_year]), :], ax=axes[0])
        sns.boxplot(x='month', y='Preal', data=df.loc[~df.year.isin([start_year, end_year]), :],ax=axes[1])


        axes[0].set_title('Distribution annuelle Tendance', fontsize=18); 
        axes[1].set_title(' Distribution mensuelle Saisonalite)', fontsize=18)

        plt.show()

    def box_per_day(df,start_year,end_year):
        fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
        sns.boxplot(x='hour', y='Preal', data=df.loc[~df.year.isin([start_year, end_year]), :],ax=axes[0])
        sns.boxplot(x='day', y='Preal', data=df.loc[~df.year.isin([start_year, end_year]), :],ax=axes[1])
        axes[0].set_title(' Distribution par heure )', fontsize=18)
        axes[1].set_title(' Distribution par jour)', fontsize=18)
        plt.show()
        