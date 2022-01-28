import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import io

from tools.helpers import *

def TrackDashboard():
    '''
    Dashboard for generating particle beam and tracking it
    '''

    style = {'description_width': 'initial'}
    
    '==================== Input Beam ============================='
    #Input parameters
    h1 = widgets.HTML("<h2>Initial Beam Parameters</h2>")
    Np = widgets.IntText(value=100, description='Np', step=100, layout=widgets.Layout(width='auto'))
    DPP = widgets.FloatText(value=1E-3, description='dpp', step=1E-4, layout=widgets.Layout(width='auto'))
    m = widgets.FloatText(value=11.17467, description='m [GeV]', step=1E-4, layout=widgets.Layout(width='auto'))
    p = widgets.FloatText(value=0.951303*12, description='p [GeV]', step=1E-4, layout=widgets.Layout(width='auto'))
    
    betx = widgets.FloatText(value=8.75, description='beta x', step=0.1, layout=widgets.Layout(width='auto'))
    bety = widgets.FloatText(value=3.38, description='beta y', step=0.1, layout=widgets.Layout(width='auto'))
    alfx = widgets.FloatText(value=-0.132, description='alpha x', step=0.01, layout=widgets.Layout(width='auto'))
    alfy = widgets.FloatText(value=-0.374, description='alpha y', step=0.01, layout=widgets.Layout(width='auto'))
    dx = widgets.FloatText(value=0.116, description='Disp x', step=0.1, layout=widgets.Layout(width='auto'))
    dy = widgets.FloatText(value=0, description='Disp y', step=0.1, layout=widgets.Layout(width='auto'))
    dpx = widgets.FloatText(value=0.0104, description="Disp x'", step=0.01, layout=widgets.Layout(width='auto'))
    dpy = widgets.FloatText(value=0, description="Disp y'", step=0.01, layout=widgets.Layout(width='auto'))
    ex = widgets.FloatText(value=1E-6, description='ex', step=1E-6, layout=widgets.Layout(width='auto'))
    ey = widgets.FloatText(value=1E-6, description='ey', step=1E-6, layout=widgets.Layout(width='auto'))

    beam_params = [Np, DPP, betx, alfx, dx, dpx, ex, bety, alfy, dy, dpy, ey]
    beam = widgets.VBox([h1, m, p, Np, DPP])
    beam_x = widgets.VBox([betx, alfx, dx, dpx, ex])
    beam_y = widgets.VBox([bety, alfy, dy, dpy, ey])
    PBeam = widgets.VBox([beam, widgets.HBox([beam_x, beam_y])], layout=widgets.Layout(width='350px'))
    
    def BeamGen(change):
        'Observes if any changes were made to initial beam parameters and updates plot'
        plt.clf()
        Beam = make_beam_dist(Np.value, DPP.value,
                              betx.value, bety.value,
                              alfx.value, alfy.value,
                              dx.value, dy.value, dpx.value, dpy.value,
                              ex.value, ey.value)
        ax1 = figure.add_subplot(1,3,1)
        ax2 = figure.add_subplot(1,3,2)
        ax3 = figure.add_subplot(1,3,3)

        ax1.plot(Beam[0], Beam[1],'.')
        ax1.set_xlabel('X'), ax1.set_ylabel('XP')

        ax2.plot(Beam[0], Beam[5],'.')
        ax2.set_xlabel('X'), ax2.set_ylabel('DPP')

        ax3.plot(Beam[0], Beam[2],'.')
        ax3.set_xlabel('X'), ax3.set_ylabel('Y')
    
    
    BeamOut = widgets.Output()
    with BeamOut:
        figure = plt.figure(figsize=(11, 3), constrained_layout=True)
        figure.canvas.toolbar_position = 'bottom'
        BeamGen(Np.value)
    
    [beamparams.observe(BeamGen, 'value') for beamparams in beam_params]
    
    '==================== MADX-PTC Set-up ============================='
    h2 = widgets.HTML("<h2>MADX Set-up</h2>")
    programme = widgets.RadioButtons(options=['MADX-PTC', 'Maptrack'], description='Tracking Software:', style=style)
    upload = widgets.FileUpload(accept='.seq', description='MADX .seq', multiple=False, style=style)
    sequence = widgets.Text(value='PIMMS', description='Seq name', layout=widgets.Layout(width='200px'))
    twissplot = widgets.Button(description="Plot Twiss")
    twisssave = widgets.Button(description="Save PTC.tfs")
    
    def FileUploader(change):
        'Set up for MADX PTC or Maptrack tracking'
        if programme.value == 'MADX-PTC':
            upload.description = 'MADX .seq'
        if programme.value == 'Maptrack':
            upload.description='Maptrack .zip'
            
    def MADX_Track(change):
        'Changes if button pushed or file uploaded'
        if programme.value == 'MADX-PTC':
            seq_file = io.BytesIO( upload.data[0] )
            seq = io.TextIOWrapper(seq_file, encoding='utf-8')
            seqfile = seq.read()
            
            '---Begin MADX---'            
            madx.input(f'BEAM, PARTICLE=POSITRON, PC={p.value}, ex={ex.value*1E6}, ey={ey.value*1E6}, DELTAP={DPP.value};')
            madx.input(seqfile)
            madx.use(sequence.value)
            twiss = madx.twiss(rmatrix=True).dframe()
            
            def twiss_plot_button(b):
                with twissout:
                    figT = plt.figure(figsize=(11, 5))
                    figT.canvas.toolbar_position = 'bottom'
                    plot_twiss(figT, twiss)

            twissplot.on_click(twiss_plot_button)
            
            def twiss_save_button(b):            
                madx.input('ptc_create_universe;')
                madx.input('ptc_create_layout,model=2,method=6,nst=5, exact;')
                madx.input(f'PTC_TWISS, icase=5, no=5, FILE="{sequence.value}_Twiss_sloexlab.tfs";')
                madx.input('PTC_END;')
                with twisssaveout:
                    display(HTML(f'Saved "{sequence.value}_Twiss_sloexlab.tfs" to {os.getcwd()}'))

            twisssave.on_click(twiss_save_button)
    
    from cpymad.madx import Madx
    madx = Madx()        
    twissout = widgets.Output()
    twisssaveout = widgets.Output()
    programme.observe(FileUploader, 'value')
    prog = widgets.VBox([h2, programme, widgets.HBox([sequence, upload]), widgets.HBox([twissplot, twisssave]), twisssaveout])
    upload.observe(MADX_Track, 'data')
    sequence.observe(MADX_Track, 'value')
    
    
    '==================== Track Set-up ============================='
    htrack = widgets.HTML('<h2>Track Settings</h2>')
    nturns = widgets.FloatText(value=1000, description='No. Turns')
    trackobs = widgets.Text(value='[#start,ES]', description='Observe at:')
    TRACK = widgets.Button(description='TRACK!')
    trackout = widgets.Output()
    
    Beams = make_beam_dist(Np.value, DPP.value,
              betx.value, bety.value,
              alfx.value, alfy.value,
              dx.value, dy.value, dpx.value, dpy.value,
              ex.value, ey.value)

    def PTCTrack(b):
        with trackout:
            for i in tqdm(range(1), desc='PTC Track Time'):
                madx.input('ptc_create_universe;')
                madx.input('ptc_create_layout,model=2,method=2,nst=5, exact;')
                madx.input('PTC_ALIGN;')
                for n in range(Np.value):
                    madx.input(f'ptc_start, x = {Beams[0][n]}, px={Beams[1][n]}, y={Beams[2][n]}, py={Beams[3][n]}, t=0, pt={Beams[5][n]};')
                obs_list = trackobs.value.strip('[]').split(',')
                for obs in obs_list:
                    madx.input(f'ptc_observe, place={obs};')
                madx.input(f'''ptc_track, turns={nturns.value}, element_by_element=True, file="{sequence.value}_Track_sloexlab.txt", ONETABLE=True, icase=5;
                ptc_track_end;
                ptc_end;''')
                with trackout:
                    print(f'Saving "{sequence.value}_Track_sloexlab.txtone" to {os.getcwd()}')
    
    with trackout:
        TRACK.on_click(PTCTrack)
    trackwidgets = widgets.VBox([htrack, widgets.HBox([widgets.VBox([nturns, trackobs, TRACK]), trackout])])
    
    Dashboard = widgets.VBox([widgets.HBox([PBeam, BeamOut]), widgets.HBox([prog, twissout]), trackwidgets])
    return(Dashboard)