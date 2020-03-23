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

def get_skimmed_data():
    '''Simple function for grabbing the JHU covid data and converting it into pandas dfs
    Location data dropped, states aggregated into countries'''
    csv_path = Path('local_data/')
    df_list = []
    # for mode in ['Confirmed']:
    for mode in ['Confirmed', 'Deaths', 'Recovered']:
        df = pd.concat([pd.read_csv(f) for f in csv_path.glob(f'*{mode}.csv')], ignore_index = True)
        df = df.drop(['Lat', 'Long'], axis=1)
        df = df.rename(columns={'Province/State':'state', 'Country/Region':'country'})
        df.state = df.state.fillna('none').str.replace(' ', '_').str.lower()
        df.country = df.country.fillna('none').str.replace(' ', '_').str.lower()
        df.set_index(['country', 'state'])

        df_country = df.groupby('country').sum().T
        df_state = df.groupby('state').sum().T

        df_state.index.rename('date', inplace=True)
        df_country.index.rename('date', inplace=True)

        df_state.columns.rename('place', inplace=True)
        df_country.columns.rename('place', inplace=True)

        df_state = df_state[['pennsylvania', 'new_york', 'california', 'ohio', 'texas']]
        df_country = df_country[['us', 'italy', 'germany']]
        df = pd.concat([df_country, df_state], axis=1, sort=False)
        df = df.set_index(pd.to_datetime(df.index))
        df_list.append(df)

    ds = xr.Dataset({'confirmed': df_list[0],
                     'dead':df_list[1],
                     'recovered':df_list[2],
                     'active':df_list[0] - df_list[1] - df_list[2],
                     'active_per_beds': (['date', 'place'],
                                         np.zeros(df_list[0].shape)),
                     'beds_per_1000': (['place'], np.ones(len(df_list[0].columns))),
                     'population': (['place'], np.ones(len(df_list[0].columns))),
                     'beds': (['place'], np.ones(len(df_list[0].columns))),
                     })

    ds['beds_per_1000'].loc[{'place':'pennsylvania'}] = 2.9
    ds['beds_per_1000'].loc[{'place':'texas'}] = 2.3
    ds['beds_per_1000'].loc[{'place':'ohio'}] = 2.8
    ds['beds_per_1000'].loc[{'place':'new_york'}] = 2.7
    ds['beds_per_1000'].loc[{'place':'california'}] = 1.8
    ds['beds_per_1000'].loc[{'place':'us'}] = 2.4
    ds['beds_per_1000'].loc[{'place':'germany'}] = 8.0
    ds['beds_per_1000'].loc[{'place':'italy'}] = 3.18

    ds['population'].loc[{'place':'pennsylvania'}] = 12813969
    ds['population'].loc[{'place':'texas'}] = 29087070
    ds['population'].loc[{'place':'ohio'}] = 11718568
    ds['population'].loc[{'place':'new_york'}] = 19491339
    ds['population'].loc[{'place':'california'}] = 39747267
    ds['population'].loc[{'place':'us'}] = 328239523
    ds['population'].loc[{'place':'germany'}] = 83149300
    ds['population'].loc[{'place':'italy'}] = 60317546

    ds['beds'] = ds['population'] * ds['beds_per_1000']/1000
    ds['active_per_beds'] = (0.12*ds['active']/(ds['beds']*0.3))


    return ds

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
        opts.Curve(tools=['hover'], width=600, ylabel='')
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


def covid_viewer_v2(ds):
    '''
    covid viewer, for actives_vs_beds
    '''
    opts.defaults(
        opts.Curve(tools=['hover'], width=800, height = 600, ylabel='')
    )
    logtog = pn.widgets.Toggle(name='Log (Y-axis)', button_type='default', value=False)
    xlim=(np.datetime64('2020-03-01'), np.datetime64('2020-03-25'))


    hv_ds = hv.Dataset(ds, ['date', 'place'], ['active_per_beds'])
    avb = hv_ds.to(hv.Curve, 'date', 'active_per_beds').overlay('place').opts(
        legend_position='top_left', shared_axes=True,
        ylim=(0, 0.13),
        xlim=xlim, title='Severe Cases per Open Hospital Bed')
    avb_log = hv_ds.to(hv.Curve, 'date', 'active_per_beds').overlay('place').opts(
        legend_position='top_left', shared_axes=True, logy=True,
        ylim=(1e-6, 10),
        xlim=xlim, title='Severe Cases per Open Hospital Bed (Log Scale)')
    max_line = hv.HLine(1).opts( opts.HLine(color='red', line_width=6),
                                opts.Points(color='#D3D3D3'))


    # layout = (avb_log)
    # layout.opts(
    #     opts.Curve(width=400, height=300, framewise=True))
    # pn_layout = pn.pane.HoloViews(layout)
    # return pn.Row(logtog, pn_layout)
    return avb
