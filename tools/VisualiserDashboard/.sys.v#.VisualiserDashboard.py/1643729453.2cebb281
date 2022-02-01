import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import io

from tools.VisualiserDashboard.PhaseSpace import PhaseSpaceInputs

def Visualiser():
    'Dashboard for producing and editing figures based on 6D tracking data'

    '====================== Uploader Inputs ============================'
    htitle = widgets.HTML('<h2> Upload 6D Tracking Data </h2>')
    hdetails = widgets.HTML('Numpy of the format: X, Xp, Y, Yp, t, pt in a 6 x nparticles x nturns array')
    form = widgets.Dropdown(options=['Numpy Array', 'PTC Track File', 'Pandas Dataframe'], description='File Type')
    uploader = widgets.FileUpload(accept='.npy', description='6D Track Results', layout=widgets.Layout(width='auto'))
    
    Upload = widgets.VBox([htitle, hdetails, widgets.HBox([widgets.VBox([widgets.HBox([form, uploader])])])])
    
    trackdata = []
    
    '====================== Uploader Data Handling ============================'
    
    def data_type(change):
        if form.value == 'Numpy Array':
                hdetails.value = 'Numpy of the format: X, Xp, Y, Yp, t, pt in a 6 x nparticles x nturns array'
                uploader.accept = '.npy'
        if form.value == 'PTC Track File':
                hdetails.value ='Output from PTCTrack.onetxt'
                uploader.accept = '.onotxt'
        if form.value == 'Pandas Dataframe':
                hdetails.value = 'Dataframe of format Particle No, Turn No, X, PX, Y, PY, T, PT, S, E'
                uploader.accept = '.csv'
                
    def form_upload(change):
        TracksB = io.BytesIO(uploader.data[0])
        Tracks = np.load(TracksB)
        if form.value == 'PTC Track File':
            tracks = pd.read_csv(seq, delim_whitespace=True, names=['Number', 'Turn', 'X', 'PX', 'Y', 'PY', 'T', 'PT', 'S', 'E'], header = 6, skiprows=2)
        trackdata.append(Tracks)
    
            
    form.observe(data_type, 'value')
    uploader.observe(form_upload, 'data')
    
    '====================== Phase Space Plot ============================'
    PhaseSpaceDash = PhaseSpaceInputs(trackdata)
    TuneOut = widgets.Output()
    
    with TuneOut:
        display(trackdata)
    
    '====================== Display Dashboard ============================'
    vistab = widgets.Tab()                           #Makes a tab of Steinbach & Hamiltonian Outputs
    vistab.children = PhaseSpaceDash, TuneOut
    vistab.set_title(0, "Coordinate Space"), vistab.set_title(1, "Tune Calculation")
    
    Dashboard = widgets.VBox([Upload, vistab])
    
    return(Dashboard)