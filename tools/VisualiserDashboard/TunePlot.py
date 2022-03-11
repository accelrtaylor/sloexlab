import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize as Norm
import matplotlib.cm as cm
import numpy as np
from tqdm.notebook import tqdm
import ipywidgets as widgets

def TuneTurnPlot(tunes):
    Tunes = tunes[0]
    plt.figure(figsize=(10,7))
    plt.plot(Tunes[1,:,:], Tunes[0,:,:], 'g.');

    xmin = np.min(Tunes[0,:,:][Tunes[0,:,:] != 0])
    xmax = np.max(Tunes[0,:,:])
    Tmin = np.min(Tunes[1,:,:][Tunes[0,:,:] != 0])
    Tmax = np.max(Tunes[1,:,:])

    plt.ylim(xmin - xmin/100, xmax + xmax/100)
    plt.xlim(Tmin, Tmax)
    plt.xlabel('Turns')
    plt.ylabel('Tune')
    plt.show()
    
def TuneOneParticle(tunes, trackdata):
    Tunes = tunes[0]
    Tracks = trackdata[0]
    
    tuneparticle = widgets.Output()
    
    Xmin = np.min(Tracks[0,:,:][Tracks[0,:,:] != 0])
    Xmax = np.max(Tracks[0,:,:])
    Qmin = np.min(Tunes[0,:,:][Tunes[0,:,:] != 0])
    Qmax = np.max(Tunes[0,:,:])
    
    qmin = widgets.FloatText(value=Qmin, description='Tune min', step=0.01, layout=widgets.Layout(width='200px'))
    qmax = widgets.FloatText(value=Qmax, description='Tune max', step=0.01, layout=widgets.Layout(width='200px'))
    xmin = widgets.FloatText(value=Xmin, description='X min', step=0.01, layout=widgets.Layout(width='200px'))
    xmax = widgets.FloatText(value=Xmax, description='X max', step=0.01, layout=widgets.Layout(width='200px'))
    num_p = widgets.BoundedIntText(value=0  , min=0, max=1E10, step=1, description='Particle', layout=widgets.Layout(width='auto'))
    oneplot = widgets.Button(description='Plot')
    
    PlotBoard = widgets.VBox([ widgets.HBox([qmin, qmax])    , widgets.HBox([xmin, xmax]),   num_p,  oneplot])
    
    TuneOneDashboard = widgets.HBox([tuneparticle, PlotBoard])
    
    with tuneparticle:
        fig, ax1 = plt.subplots(figsize=(10,7))

        ax1.plot(Tunes[1,num_p.value,:], Tunes[0,num_p.value,:], 'g.')
        ax1.set_xlabel('Turn'), ax1.set_ylabel('Tune')
        ax1.set_ylim(qmin.value, qmax.value)

        ax2 = ax1.twinx()
        ax2.set_ylabel('X [m]', color='orange')
        ax2.plot(Tracks[0,num_p.value,:], '.', color='orange')
        ax2.set_ylim(xmin.value - xmin.value/1000, xmax.value + xmax.value/1000)
        

        fig.tight_layout()
        plt.show()
    
    def tuneoneplot(b):
        ax1.clear(), ax2.clear()
        with tuneparticle:

            ax1.plot(Tunes[1,num_p.value,:], Tunes[0,num_p.value,:], 'g.')
            ax1.set_ylim(qmin.value, qmax.value)

            ax2.plot(Tracks[0,num_p.value,:], '.', color='orange')
            ax2.set_ylim(xmin.value - xmin.value/1000, xmax.value + xmax.value/1000)
            
    
    oneplot.on_click(tuneoneplot)

    
    return TuneOneDashboard
        

def TunePlotDash(tunedata, trackdata):
    ''''''
    tuneturn = widgets.Output()
    tuneparticle = TuneOneParticle(tunedata, trackdata)
    
    with tuneturn:
        TuneTurnPlot(tunedata)


    TuneTabs = widgets.Tab()
    TuneTabs.children = tuneturn, tuneparticle
    TuneTabs.set_title(0, 'Tune per Turn'), TuneTabs.set_title(1, "Tune for particle")
    return TuneTabs