import matplotlib.pyplot as plt
import numpy as np

from tqdm.notebook import tqdm

def Steinbach(S, QX, Q_res, dpp, dQX,ex, Np):
    '''
    Observes baseline parameters of the dashboard and returns values to plot steinbach diagram
    '''
    Q_range = np.linspace(-0.5, +0.5, 5000)                        # Tune range to plot line
    A_stopb = (48 * np.pi * 3**0.5)**0.5 * np.abs(Q_range / S )  # Amplitude due to virtual sextupole
    # Ensures particle distribution does not regenerate each time a parameter changes
    np.random.seed(1)
    DPP = np.random.uniform(-dpp, dpp, Np)                       # Beam momentum spread
    EX = np.random.normal(0, ex, Np)                             # Beam Emittance
    An = (abs(EX)/np.pi)**0.5                                    # Converts emittance to amplitude
    #Returs particle tune range, particle amplitudes, resonant line in Qx and A
    return([QX + DPP*dQX, An, Q_res+Q_range, A_stopb ])


def plot_twiss(fig, twiss, title=''):
    '''
    Written by Y. Dutheil
    Prints schematic of accelerator lattice as boxes.
    Plots beta in x and  y and dispersion in x and y
    '''
    import matplotlib as mpl
    gs = mpl.gridspec.GridSpec(3, 1, height_ratios=[1, 3,3])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    #ax4 = fig.add_subplot(gs[3], sharex=ax1)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # top plot is synoptic
    ax1.axis('off')
    ax1.set_ylim(-1.2, 1)
    ax1.plot([0, twiss['s'].max()], [0, 0], 'k-')

    for _, row in twiss[twiss['keyword'].str.contains('quadrupole|rbend|sbend')].iterrows():
        if row['keyword'] == 'quadrupole':
            _ = ax1.add_patch(
                mpl.patches.Rectangle(
                    (row['s']-row['l'], 0), row['l'], np.sign(row['k1l']),
                    facecolor='k', edgecolor='k'))
        elif (row['keyword'] == 'rbend' or 
              row['keyword'] == 'sbend'):
            _ = ax1.add_patch(
                mpl.patches.Rectangle(
                    (row['s']-row['l'], -1), row['l'], 2,
                    facecolor='None', edgecolor='k'))

    #2nd plot is beta functions
    ax2.set_ylabel(r'$\beta$ (m)')
    ax2.plot(twiss['s'], twiss['betx'], 'r-')
    ax2.plot(twiss['s'], twiss['bety'], 'b-')        

    #3rd plot is dispersion functions
    ax3.set_ylabel('D (m)')
    ax3.plot(twiss['s'], twiss['dx'], 'r-')
    ax3.plot(twiss['s'], twiss['dy'], 'b-')

    axnames = ax1.twiny()
    axnames.spines['top'].set_visible(False)
    axnames.spines['left'].set_visible(False)
    axnames.spines['right'].set_visible(False)
    ax1.get_shared_x_axes().join(ax1, axnames)

    ticks, ticks_labels = list(), list()
    for keyword in ['quadrupole', 'rbend', 'sbend']:
        sub_twiss = twiss[twiss['keyword'] == keyword]
        ticks += list(sub_twiss['s'])
        ticks_labels += list(sub_twiss.index)

    axnames.set_xticks(ticks)
    axnames.set_xticklabels(ticks_labels, rotation=90)

    ax3.set_xlabel('s (m)')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    ax1.set_xlim(twiss['s'].min(), twiss['s'].max())
    plt.title(title)
    plt.show()
    
def make_beam_dist(N, dpp, betx, bety, alfx, alfy, dx, dy, dpx, dpy, ex, ey):
    '''
    Produces a beam of N particles following a Gaussian distribution.    
    
    P is a dataframe including all beta, alphas etc.
    '''
    gamx, gamy = (1+alfx**2)/betx, (1+alfy**2)/bety  #Calculation of Twiss gamma from alpha and beta
    
    # RMS parameters
    x_rms  = np.sqrt(ex * betx)
    y_rms  = np.sqrt(ey * bety)
    xp_rms = np.sqrt(ex * gamx)
    yp_rms = np.sqrt(ey * gamy)
    
    # Ensures the particle distribution does not change each time a variable is chaged
    np.random.seed(1)                               
    x0  = np.random.normal(0, x_rms,  N)
    px0 = np.random.normal(0, xp_rms, N)
    y0  = np.random.normal(0, y_rms,  N)
    py0 = np.random.normal(0, yp_rms, N)
    
    # Non-normalises the distribution
    xp0 = (px0 - alfx * x0)/betx
    yp0 = (py0 - alfy * y0)/bety

    Del = np.random.normal(0, dpp, N)
    
    # Adding dispersive effects to match beam to lattice
    x  = x0  + dx  * Del
    y  = y0  + dy  * Del
    px = xp0 + dpx * Del
    py = yp0 + dpy * Del
    
    xp = alfx*x + betx*px 
    yp = alfy*y + bety*py
    
    return(x, xp, y, yp, 0, Del)


def rad(x):
    'Defining radians'
    theta = x * np.pi / 180
    return theta