import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

from tools.helperhamiltonian import *

import io
import warnings
        
def Hamiltonian_Output(tdf, QX, QX_r, ES, ele_pos, Hamilton_err):
    '''
    Inputs
    -------
        tdf             : Twiss dataframe uploader widget
        QX              : Horizontal tune widget
        QX_r            : Resonant tune widget
        ES              : Electrostatic septa widget
        ele_pos         : Element where hamilton is measured
        Hamiltonian_err : Error output widget
        
    Uses functions in tools.Ham_tools to calculate the Hamiltonian from a Twiss Dataframe file.
    Plots as a matplotlib contour plot.
    
    Returns
    -------
        Hamilton_out  : widget output showing contour plot of Hamiltonian
    '''
    # Defining output, containing figure and axis
    Hamilton_out = widgets.Output()
    with Hamilton_out:
        figH, axH = plt.subplots(figsize=(9,8))
        
    def HamiltonPlot(change):
        '''
        Inputs
        -------
            tdf             : Twiss dataframe uploader widget
            QX              : Horizontal tune widget
            QX_r            : Resonant tune widget
            ES              : Electrostatic septa widget
            ele_pos         : Element where hamilton is measured
            
        If input values change, redraws Hamiltonian & updates plot
        '''
        import matplotlib.cm as cm
        
        if tdf.data != []:                                       #If non-empty Twiss dataframe uploader
            tdf_bytes = io.BytesIO( tdf.data[0] )                # Converts uploaded file into binary
            TDF = io.TextIOWrapper(tdf_bytes, encoding='utf-8')  # Unwraps binary values
            header, twiss_df = readtfs(TDF)                      # Extracts header information and dataframe.

            axH.clear()                                          #Removes previous plot to avoid overlapping contours
            xx, xpxp, hh = HamiltonianContour(twiss_df, QX.value, QX_r.value, ele_pos.value, Hamilton_err)

            with warnings.catch_warnings():                      # Ignores errors of empty contour lines
                warnings.simplefilter("ignore")
                axH.contour(xx, xpxp, hh, 500, colors=[cm.get_cmap("coolwarm")(0)], linestyles='solid')

            axH.axvline(ES.value, color='black')                 # Draws on position of aperture limit
            axH.set_xlabel('x [m]')
            axH.set_ylabel('px')
            axH.set_xlim(-ES.value-0.01, ES.value+0.01)
            axH.set_ylim(-0.004, 0.004)
            axH.set_title(f'Hamiltonian - nu={QX.value} ')

            figH.canvas.draw()
            
    # If either of these change, update Hamiltonian Plot
    tdf.observe(HamiltonPlot, 'data')
    QX.observe(HamiltonPlot, 'value')
    QX_r.observe(HamiltonPlot, 'value')
    ES.observe(HamiltonPlot, 'value')
    ele_pos.observe(HamiltonPlot, 'value')
    
    return Hamilton_out