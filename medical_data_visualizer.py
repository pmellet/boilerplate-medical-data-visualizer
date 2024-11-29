import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('medical_examination.csv')

BMI = df['weight'] / ((df['height'] * 0.01) ** 2)
df['overweight'] = (BMI > 25).astype(int)

df['gluc'] = (df['gluc'] >= 2).astype(int)
df['cholesterol'] = (df['cholesterol'] >= 2).astype(int)

def draw_cat_plot():

    df_cat = pd.melt(df,
                id_vars = ['age', 'height', 'weight', 'sex',  'ap_hi','ap_lo','cardio'],
                value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                var_name = 'variable',
                value_name = 'Value'
            )

    df_cat_sorted = df_cat.sort_values(by=['cardio','variable'], ascending=[True,True]).reset_index()
    df_cat_sorted = df_cat_sorted[['cardio','variable','Value']]

    binary_possibilities = df_cat_sorted.value_counts() 
    df_messy = pd.DataFrame(binary_possibilities)

    df_ordened = df_messy.sort_values(by = ['cardio','variable','Value'], ascending = [True,True,True]).reset_index()
    df_ordened.columns.values[3] = 'total'

    facet = sns.catplot(data=df_ordened,
        x="variable",
        y="total",
        col="cardio",
        hue = 'Value',
        kind="bar"
    )

    fig = facet.fig
    fig.savefig('catplot.png')
    return fig

def draw_heat_map():

    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) &
                (df['height'] <= df['height'].quantile(0.975)) &
                (df['weight'] >= df['weight'].quantile(0.025)) &
                (df['weight'] <= df['weight'].quantile(0.975)) 
            ]

    df_heat = df_heat.round(1)
    corr = df_heat.corr()

    mask = np.ones(corr.shape, dtype =int)
    mask = np.triu(mask)

    plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.family': 'serif'})

    cmap = sns.color_palette("icefire", as_cmap=True)
    ax = sns.heatmap(corr,
                    fmt = ".1f",
                    annot = True,
                    linewidth = .5,
                    vmin = -0.16,
                    vmax = 0.3,
                    square = True,
                    mask = mask,
                    cmap = cmap,
                    cbar_kws={'ticks': [-0.08, 0.0, 0.08, 0.16, 0.24]}
                )

    fig = plt.gcf()
    fig.savefig('heatmap.png')
    return fig