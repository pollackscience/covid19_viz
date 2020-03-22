import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import holoviews as hv
from holoviews import opts


def load_and_clean_jhu_data(mode='Confirmed'):
    '''Simple function for grabbing the JHU covid data and converting it into pandas dfs
    Location data dropped, states aggregated into countries'''
    csv_path = Path('data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/')
    df = pd.concat([pd.read_csv(f) for f in csv_path.glob(f'*{mode}.csv')], ignore_index = True)
    df = df.drop(['Lat', 'Long'], axis=1)
    df = df.rename(columns={'Province/State':'state', 'Country/Region':'country'})
    df.state = df.state.fillna('none').str.replace(' ', '_').str.lower()
    df.country = df.country.fillna('none').str.replace(' ', '_').str.lower()
    df.set_index(['country', 'state'])
    df = df.groupby('country').sum()
    df = df.T
    df = df.set_index(pd.to_datetime(df.index))
    df.index.rename('date', inplace=True)
    return df

def make_xr_ds():
    '''xarray maker'''
    df_confirmed = load_and_clean_jhu_data('Confirmed')
    df_death = load_and_clean_jhu_data('Deaths')
    df_recovered = load_and_clean_jhu_data('Recovered')
    ds = xr.Dataset({'confirmed': df_confirmed, 'dead':df_death, 'recovered':df_recovered})
    ds = ds.sel(country=['us', 'italy', 'korea,_south'])
    return ds



def covid_viewer(ds):
    '''
    covid viewer, start with MRE view backbone?
    '''
    hv_ds = hv.Dataset(ds, ['date', 'country'], ['confirmed', 'dead', 'recovered'])
    layout = (hv_ds.to(hv.Curve, 'date',
                       'confirmed').overlay('country').opts(legend_position='top_left') +
              hv_ds.to(hv.Curve, 'date',
                       'dead').overlay('country').opts(show_legend=False) +
              hv_ds.to(hv.Curve, 'date',
                       'recovered').overlay('country').opts(show_legend=False)).cols(1)
    layout.opts(
        opts.Curve(width=600, height=250, framewise=True))
    return layout
