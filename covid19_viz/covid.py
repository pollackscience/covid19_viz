import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import holoviews as hv
from holoviews import opts
import panel as pn


def load_and_clean_jhu_data(mode='Confirmed'):
    '''Simple function for grabbing the JHU covid data and converting it into pandas dfs
    Location data dropped, states aggregated into countries'''
    csv_path = Path('local_data/')
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
    opts.defaults(
        opts.Curve(tools=['hover'], width=600)
    )
    logtog = pn.widgets.Toggle(name='Log (Y-axis)', button_type='default', value=False)
    xlim=(np.datetime64('2020-02-10'), np.datetime64('2020-03-25'))


    hv_ds = hv.Dataset(ds, ['date', 'country'], ['confirmed', 'dead', 'recovered'])
    confirmed = hv_ds.to(hv.Curve, 'date', 'confirmed').overlay('country').opts(
        legend_position='top_left', shared_axes=False,
        ylim=(-ds.confirmed.values.max()*0.1, ds.confirmed.values.max()*1.1),
        xlim=xlim, title='Confirmed')
    confirmed_log = hv_ds.to(hv.Curve, 'date', 'confirmed').overlay('country').opts(
        legend_position='top_left', shared_axes=False, logy=True,
        ylim=(1, ds.confirmed.values.max()*2),
        xlim=xlim, title='Confirmed (Log)')

    dead = hv_ds.to(hv.Curve, 'date', 'dead').overlay('country').opts(
        legend_position='top_left', shared_axes=False,
        ylim=(-ds.dead.values.max()*0.1, ds.dead.values.max()*1.1),
        xlim=xlim, title='Dead')
    dead_log = hv_ds.to(hv.Curve, 'date', 'dead').overlay('country').opts(
        legend_position='top_left', shared_axes=False, logy=True,
        ylim=(0.1, ds.dead.values.max()*2),
        xlim=xlim, title='Dead (Log)')

    recovered = hv_ds.to(hv.Curve, 'date', 'recovered').overlay('country').opts(
        legend_position='top_left', shared_axes=False,
        ylim=(-ds.recovered.values.max()*0.1, ds.recovered.values.max()*1.1),
        xlim=xlim, title='Recovered')
    recovered_log = hv_ds.to(hv.Curve, 'date', 'recovered').overlay('country').opts(
        legend_position='top_left', shared_axes=False, logy=True,
        ylim=(0.1, ds.recovered.values.max()*2),
        xlim=xlim, title='Recovered (Log)')

    layout = (confirmed + confirmed_log + dead + dead_log + recovered + recovered_log).cols(2)
    layout.opts(
        opts.Curve(width=400, height=250, framewise=True))
    # pn_layout = pn.pane.HoloViews(layout)
    # return pn.Row(logtog, pn_layout)
    return layout
