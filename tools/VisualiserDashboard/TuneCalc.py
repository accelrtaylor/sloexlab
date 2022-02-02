#PyNAFF is imported from my local site
import os
import numpy as np
import ipywidgets as widgets
import PyNAFF

from tqdm.notebook import tqdm

def tune_scroll(Tracks, n_particles, t_step, Q_step, param='X'):
    ''' 
    x_pos is a np array of number of turns T
    t_step is the number of turns per step
    '''
    X_pos = Tracks[0]
    coord = {'X': 0, 'Xp':1, 'Y':2, 'Yp':3}
    N_turns = np.shape(X_pos)[2]
    Q = np.zeros((2, n_particles, int(N_turns/t_step)))
    for p in tqdm(range(n_particles)):
        x_pos = X_pos[coord[param], p, :]
        for t in range(int(N_turns/t_step)):
            Turn_no = t * t_step
            if Turn_no - Q_step > 0:
                x_step = x_pos[Turn_no - Q_step : Turn_no]
                Q[0,p,t] = PyNAFF.naff(x_step - np.mean(x_step), Q_step, 1, 0, False)[0][1]

            Q[1,p,t] = Turn_no
    return Q


def TuneDashboard(Tracks):
    '''
    '''
    
    '----- Inputs -----'
    WindowCalc = widgets.BoundedIntText(value=128, min=2**0, max=2**15, step=64, description='Window Calc')
    WindowStep = widgets.BoundedIntText(value=10,  min=1, max=2**15, step=1, description='Window Step')
    Coordinate = widgets.ToggleButtons(options=['X', 'Y'], value='X',  description = 'Coordinate', layout=widgets.Layout(width='auto'))

    TurnMin = widgets.BoundedIntText(value=0  , min=0, max=1E10, step=100, description='Turn Min')
    TurnMax = widgets.BoundedIntText(value=100, min=0, max=1E10, step=100, description='Turn Max')
    nparticles = widgets.BoundedIntText(value=100  , min=0, max=1E10, step=100, description='Particle no.')

    Calculate = widgets.Button(description='Calculate')
    Download = widgets.Button(description='Download')
    Upload = widgets.FileUpload(description='Upload')
    
    TuneOut = widgets.Output()
    
    TuneInputs = widgets.HBox([ widgets.VBox([WindowCalc, WindowStep, Coordinate])   , widgets.VBox([TurnMin, TurnMax, nparticles])    ,   widgets.VBox([Calculate, Download, Upload])       ])
    
    def TuneCalc(change):
        with TuneOut:
            Qx = tune_scroll(Tracks, nparticles.value, WindowStep.value, WindowCalc.value, Coordinate.value)
        return Qx
    
    Calculate.on_click(TuneCalc)
    
    
    TuneDash = widgets.VBox([TuneInputs, TuneOut])
    return TuneDash