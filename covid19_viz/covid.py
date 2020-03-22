import numpy as np
import pandas as pd
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


def covid_viewer(ds)
    '''
    covid viewer, start with MRE view backbone?
    '''
    opts.defaults(
        opts.GridSpace(shared_xaxis=True, shared_yaxis=True,
                       fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12},
                       plot_size=size),
        opts.Layout(fontsize={'title': 16, 'labels': 16, 'xticks': 12, 'yticks': 12}),
        opts.Image(cmap='gray', width=size, height=size, xaxis=None,
                   yaxis=None),
        opts.Labels(text_color='white', text_font_size='20pt', text_align='left',
                    text_baseline='bottom'),
        opts.Path(color='white'),
        opts.Spread(width=600),
        opts.NdOverlay(show_legend=True, border_muted_alpha=0.1)
    )

    # Make holoviews dataset from xarray
    # xr_ds = xr_ds.sel(subject=['0006', '0384'])


    hv_ds = hv.Dataset(ds)
    print(hv_ds)

    hv_ds_mri_image = hv_ds_mri.to(hv.Image, kdims=['x', 'y'], vdims='image_mri', dynamic=True)
    hv_ds_mri_mask = hv_ds_mri.to(hv.Image, kdims=['x', 'y'], vdims='mask_mri',
                                  dynamic=True).opts(tools=[])

    hv_ds_mre_image_1 = hv_ds_mre_1.to(hv.Image, kdims=['x', 'y'], vdims='image_mre',
                                       dynamic=True).opts(cmap='viridis')
    hv_ds_mre_mask_1 = hv_ds_mre_1.to(hv.Image, kdims=['x', 'y'], vdims='mask_mre',
                                      dynamic=True).opts(tools=[])
    if not torch:
        hv_ds_mre_image_2 = hv_ds_mre_2.to(hv.Image, kdims=['x', 'y'], vdims='image_mre',
                                           dynamic=True).opts(cmap='viridis')
        hv_ds_mre_mask_2 = hv_ds_mre_2.to(hv.Image, kdims=['x', 'y'], vdims='mask_mre',
                                          dynamic=True).opts(tools=[])

    slider = pn.widgets.FloatSlider(start=0, end=1, value=0.7, name='mask transparency')
    if torch:
        cslider = pn.widgets.RangeSlider(start=-2, end=2, value=(-2, 2), name='contrast')
        cslider2 = pn.widgets.RangeSlider(start=0, end=200, value=(0, 100), name='mre contrast')
    else:
        cslider = pn.widgets.RangeSlider(start=0, end=2000, value=(0, 1000), name='contrast')
        cslider2 = pn.widgets.RangeSlider(start=0, end=12000, value=(0, 10000), name='mre contrast')

    redim_image_mri = {'image_mri': (0, 1200)}
    hv_ds_mri_image = hv_ds_mri_image.redim.range(**redim_image_mri).opts(tools=['hover'])
    hv_ds_mri_image = hv_ds_mri_image.apply.opts(clim=cslider.param.value)
    redim_mask_mri = {'mask_mri': (0.1, 2)}
    hv_ds_mri_mask = hv_ds_mri_mask.opts(cmap='Category10', clipping_colors={'min': 'transparent'},
                                         color_levels=10)
    hv_ds_mri_mask = hv_ds_mri_mask.redim.range(**redim_mask_mri)
    hv_ds_mri_mask = hv_ds_mri_mask.apply.opts(alpha=slider.param.value)

    redim_image_mre_1 = {'image_mre_1': (0, 10000)}
    # hv_ds_mre_image_1 = hv_ds_mre_image_1.redim(image_mre='image_mre_1')
    hv_ds_mre_image_1 = hv_ds_mre_image_1.apply.opts(clim=cslider2.param.value)
    hv_ds_mre_image_1 = hv_ds_mre_image_1.redim.range(**redim_image_mre_1).opts(tools=['hover'])
    redim_mask_mre = {'mask_mre': (0.1, 2)}
    hv_ds_mre_mask_1 = hv_ds_mre_mask_1.opts(cmap='Category10',
                                             clipping_colors={'min': 'transparent'},
                                             color_levels=10)
    hv_ds_mre_mask_1 = hv_ds_mre_mask_1.redim.range(**redim_mask_mre)
    hv_ds_mre_mask_1 = hv_ds_mre_mask_1.apply.opts(alpha=slider.param.value)

    if not torch:
        redim_image_mre_2 = {'image_mre': (-200, 200)}
        hv_ds_mre_image_2 = hv_ds_mre_image_2.redim.range(**redim_image_mre_2).opts(tools=['hover'])
        redim_mask_mre = {'mask_mre': (0.1, 2)}
        hv_ds_mre_mask_2 = hv_ds_mre_mask_2.opts(cmap='Category10',
                                                 clipping_colors={'min': 'transparent'},
                                                 color_levels=10)
        hv_ds_mre_mask_2 = hv_ds_mre_mask_2.redim.range(**redim_mask_mre)
        hv_ds_mre_mask_2 = hv_ds_mre_mask_2.apply.opts(alpha=slider.param.value)
        layout = (((hv_ds_mre_image_1 * hv_ds_mre_mask_1).grid('mre_type') +
                  (hv_ds_mre_image_2 * hv_ds_mre_mask_2).grid('mre_type')) +
                  (hv_ds_mri_image * hv_ds_mri_mask).layout('sequence').cols(3)
                  ).cols(2)
    else:
        layout = (((hv_ds_mre_image_1 * hv_ds_mre_mask_1).grid('mre_type')) +
                  (hv_ds_mri_image * hv_ds_mri_mask).layout('sequence').cols(3)
                  ).cols(1)
    pn_layout = pn.pane.HoloViews(layout)
    wb = pn_layout.widget_box
    wb.append(slider)
    wb.append(cslider)
    wb.append(cslider2)

    # return pn.Column(slider, cslider2, layout, cslider)
    return pn.Column(wb, pn_layout)
    # return hv_ds_mri_image
    # return hv_ds_mre_image

