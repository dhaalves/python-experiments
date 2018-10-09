import plotly

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.figure_factory as FF

import numpy as np
import pandas as pd
from IPython.display import Image
from plotly.graph_objs import Heatmap
from os import listdir
from os.path import isfile, join

# print(plotly.__version__)
# py.sign_in('dhaalves', 'mZlkGLDiBPfXDp3KGFsa')

path = '/home/daniel/Desktop/dhaa-weka/results/confusion_matrix'


def save_image(filename):
    df = pd.read_csv(path + '/' + filename + '.csv', header=None)
    # trace = go.Heatmap(z=df[df.columns[:53]].as_matrix(),
    #                    x=df[df.columns[53]].as_matrix(),
    #                    y=df[df.columns[53]].as_matrix())
    # data = [trace]
    trace1 = {
        "x": df[df.columns[53]].as_matrix(),
        "y": df[df.columns[53]].as_matrix(),
        "z": df[df.columns[:53]].as_matrix(),
        "autocolorscale": False,
        "colorbar": {
            "x": 0,
            "y": 0.55,
            "dtick": 0.1,
            "exponentformat": "e",
            "len": 0.95,
            "lenmode": "fraction",
            "nticks": 10,
            "outlinewidth": 1,
            "showticklabels": True,
            "thickness": 30,
            "thicknessmode": "pixels",
            "tick0": 0,
            "tickfont": {"size": 15},
            "tickmode": "auto",
            "ticks": "",
            "title": "<br>",
            "titleside": "right",
            "xanchor": "right",
            "yanchor": "middle"
        },
        "colorscale": [
            [0, "rgb(144,19,28)"], [0.3, "rgb(150,19,27)"], [0.4, "rgb(182,59,37)"], [0.5, "rgb(192,88,64)"],
            [0.65, "rgb(207,131,113)"], [1, "rgb(222,183,175)"]],
        "connectgaps": True,
        "name": "B",
        "opacity": 1,
        "reversescale": True,
        "showscale": True,
        "transpose": False,
        "type": "heatmap",
        "uid": "ee9d1e",
        "xgap": 0,
        "xsrc": "dhaalves:275:41a728",
        "ygap": 0,
        "ysrc": "dhaalves:275:590cbf",
        "zauto": True,
        "zmax": 1,
        "zmin": 0,
        "zsmooth": False,
        "zsrc": "dhaalves:275:57cf16,557289,d9822a,1245b6,59bb48,ac8571,abcacf,6e0258,51a5fe,70180f,773875,9fc22a,73df85,3f79be,8bf4e1,a331ed,df82d6,89f30e,1e63a9,f15f44,0b3d11,570bc8,bd8ca2,1ba2ce,e52b7e,e7488d,7fb306,17bc5f,ede16b,e0191f,a9f416,94a948,8dcc93,bab5ba,65b53d,652193,12fcf6,4c7ae0,3c6fe5,62db91,36e55f,767797,df1e77,51aa66,639e26,3f1091,392e86,abc203,303bc1,c4ed6b,19de56,b0f894,aa69f4"
    }
    data = Data([trace1])
    layout = {
        "width": 1920,
        "height": 1080,
        "autosize": False,
        "dragmode": "zoom",
        "hidesources": False,
        "hovermode": "closest",
        "legend": {
            "x": 1.02,
            "y": 1,
            "orientation": "v",
            "traceorder": "normal"
        },
        "paper_bgcolor": "#fff",
        "plot_bgcolor": "#fff",
        "separators": ".,",
        "showlegend": True,
        "smith": False,
        "title": "Click to enter Plot title",
        "xaxis": {
            "anchor": "y",
            "autorange": True,
            "categoryorder": "trace",
            "color": "#444",
            "domain": [0.01, 0.94],
            "dtick": 1,
            "exponentformat": "SI",
            "fixedrange": False,
            "gridcolor": "rgb(238, 238, 238)",
            "gridwidth": 3,
            "mirror": False,
            "nticks": 0,
            "range": [-0.5, 52.5],
            "rangemode": "normal",
            "showexponent": "all",
            "showgrid": False,
            "showline": False,
            "showspikes": False,
            "showticklabels": True,
            "showtickprefix": "last",
            "side": "bottom",
            "tick0": 0,
            "tickangle": 45,
            "tickcolor": "#444",
            "tickfont": {
                "color": "#444",
                "family": "\"Open Sans\", verdana, arial, sans-serif",
                "size": 15
            },
            "ticklen": 5,
            "tickmode": "auto",
            "tickprefix": "",
            "ticks": "",
            "ticksuffix": "",
            "tickwidth": 1,
            "title": "Click to enter X axis title",
            "titlefont": {
                "color": "#444",
                "family": "\"Open Sans\", verdana, arial, sans-serif",
                "size": 15
            },
            "type": "category",
            "zeroline": False,
            "zerolinecolor": "#444",
            "zerolinewidth": 0
        },
        "yaxis": {
            "anchor": "x",
            "autorange": True,
            "categoryorder": "trace",
            "color": "#444",
            "domain": [0.1, 1.01],
            "dtick": 1,
            "exponentformat": "e",
            "fixedrange": False,
            "gridcolor": "rgb(238, 238, 238)",
            "gridwidth": 3,
            "mirror": False,
            "nticks": 0,
            "position": 0,
            "range": [52.5, -0.5],
            "rangemode": "normal",
            "showexponent": "all",
            "showgrid": False,
            "showline": False,
            "showspikes": False,
            "showticklabels": True,
            "side": "right",
            "tick0": 0,
            "tickangle": "auto",
            "tickcolor": "#444",
            "tickfont": {
                "color": "#444",
                "family": "\"Open Sans\", verdana, arial, sans-serif",
                "size": 12
            },
            "ticklen": 5,
            "tickmode": "auto",
            "tickprefix": "",
            "ticks": "",
            "ticksuffix": "",
            "tickwidth": 1,
            "title": "Click to enter Y axis title",
            "titlefont": {
                "color": "#444",
                "family": "\"Open Sans\", verdana, arial, sans-serif",
                "size": 14
            },
            "type": "category",
            "zeroline": False,
            "zerolinecolor": "#444",
            "zerolinewidth": 0
        }
    }
    # py.iplot(data, filename='labelled-heatmap')
    # layout = go.Layout(title='A Simple Plot', width=1920, height=1080)
    # fig = Figure(data=data, layout=layout)
    # py.image.save_as(fig, filename=filename + '.png')
    # py.image.ishow(fig)
    # from IPython.display import Image
    # Image('a-simple-plot.png')
    # sample_data_table = FF.create_table(df.head())
    # py.iplot(sample_data_table, filename='sample-data-table')

    import plotly.offline as offline
    import plotly.graph_objs as go

    offline.plot({'data': data,
                  'layout': layout},
                 image='png', filename=filename,  image_filename=filename + '.png', image_width=1440, image_height=1080)


if __name__ == '__main__':

    # files = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    for f in listdir(path):
        save_image(f.split('.')[0])
