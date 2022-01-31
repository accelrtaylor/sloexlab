import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import io

def Visualiser():
    'Dashboard for producing and editing figures based on 6D tracking data'

    htitle = widgets.HTML('<h2> Upload 6D Tracking Data </h2>')
    Hdetails = widgets.Output()
    hdetails = widgets.HTML('Numpy of the format: X, Xp, Y, Yp, t, pt in a 6 x nparticles x nturns array')
    form = widgets.Dropdown(options=['Numpy Array', 'PTC Track File', 'Pandas Dataframe'], description='File Type')
    uploader = widgets.FileUpload(accept='', description='6D Track Results', layout=widgets.Layout(width='auto'))
    showdata = widgets.Button(description='Show Data')
    Trackdata = widgets.Output()
    
    trackdata = []
    
    with Hdetails: display(hdetails)
    
    def upload_form(change):
        if form.value == 'Numpy Array':
                hdetails = widgets.HTML('Numpy of the format: X, Xp, Y, Yp, t, pt in a 6 x nparticles x nturns array')
        if form.value == 'PTC Track File':
                hdetails = widgets.HTML('Output from PTCTrack.onetxt')
                if len(uploader.data) != 0:
                    TracksB = io.BytesIO( uploader.data[0] )
                    Tracks = io.TextIOWrapper(seq_file, encoding='utf-8')
                    tracks = pd.read_csv(seq, delim_whitespace=True, names=['Number', 'Turn', 'X', 'PX', 'Y', 'PY', 'T', 'PT', 'S', 'E'], header = 6, skiprows=2)
                    with Trackdata:
                        showdata.on_click(display(tracks))
        if form.value == 'Pandas Dataframe':
                hdetails = widgets.HTML('Dataframe of format Particle No, Turn No, X, PX, Y, PY, T, PT, S, E')
        Hdetails.clear_output()
        with Hdetails: display(hdetails)
        
        def show_data(b):
            with Trackdata:
                display(tracks.head())
            
    form.observe(upload_form, 'value')
    Upload = widgets.VBox([htitle, widgets.HBox([widgets.VBox([Hdetails, widgets.HBox([form, uploader]), showdata]), Trackdata])])
    
    return(Upload)