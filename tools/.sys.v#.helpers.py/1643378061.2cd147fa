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
    ax1._shared_x_axes.join(ax1, axnames)

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


#ResTheory functions written by P. Arrutia

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import os

def get_p3rtilde(twiss_df, nu):
    """
    Sextupole Component
    This function computes p3rtilde from the Wiedemann convention of sextupole str.

    Arguments:
      - twiss_df: twiss pandas dataframe
    Returns:
      - (totVStr, totVPhase): normalized strength and phase location of the virtual sextupole.
                              N.B. All the active sextupoles are used for the computations, also the
                              chromatic ones.
    """
    dfRStr = twiss_df.copy()
    lsTotDf = dfRStr[ dfRStr["k2l"] != 0 ].copy()
    
    strongest_sexts = lsTotDf.k2l.abs().sort_values(ascending=False).head(10).index
    #print(twiss_df.loc[strongest_sexts][['k2l', 'betx', 's', 'mux', 'l']])
    
    factor = np.sqrt(2)/(24*np.pi*np.sqrt(nu))
    normTotSeStr = factor*np.array(np.array(lsTotDf["k2l"])*np.array(lsTotDf["betx"].pow(3/2)) )
    totSePhases = 2*np.pi*np.array(lsTotDf["mux"])

    totSeSinSum = np.sum( normTotSeStr*np.sin(3*totSePhases) )
    totSeCosSum = np.sum( normTotSeStr*np.cos(3*totSePhases) )

    totVStr = np.sqrt( np.power(totSeSinSum,2) + np.power(totSeCosSum,2) )
    totVPhase = np.arctan2((totSeSinSum ),( totSeCosSum ))/3

    return totVStr, totVPhase    

def get_p40tilde(twiss_df, nu):
    """
    This function computes the p40tilde from the wiedemann convention for octupole str.

    Arguments:
      - twiss_df: twiss pandas dataframe
      - nu : tune
    Returns:
      - totVStr: normalized strength of the virtual octupole.
    """
    
    dfRStr = twiss_df.copy()
    lsTotDf = dfRStr[ dfRStr["k3l"] != 0 ].copy()

    strongest_octs = lsTotDf.k3l.abs().sort_values(ascending=False).head(10).index
    #print(twiss_df.loc[strongest_octs][['k3l', 'betx', 's', 'mux', 'l']])
    
    factor = 1/(32*nu*np.pi)
    normTotOcStr = factor*np.array(np.array(lsTotDf["k3l"])*np.array(lsTotDf["betx"].pow(2)) )
    totVStr = np.sum(normTotOcStr)

    return totVStr

def j_to_w(j, phi, nu):
    w = np.sqrt(2*j/nu)*np.cos(phi)
    wdot = np.sqrt(2*nu*j)*np.sin(phi)
    return w, wdot

def phi1_to_phi(j, phi1, hh, nu, dnu, mux):
    return phi1 + (nu-dnu)/nu * mux, hh+(nu-dnu)*j

def w_to_x(w, wdot, nu, alpha, beta):
    x = np.sqrt(beta)*w
    xp = wdot/(nu*np.sqrt(beta)) - alpha*w/np.sqrt(beta)
    return x, xp

def hamiltonian(w, wdot, nu, dnu, p40tilde, p3rtilde):
    term1 = dnu*nu/2 * (w**2 + wdot**2/nu**2)
    term2 = p40tilde*(nu/2 * (w**2 + wdot**2/nu**2))**2
    term3 = p3rtilde*nu**(3/2)/2**(3/2) * (w**3 - 3*w*wdot**2/nu**2)
    return term1 + term2 + term3

def hamiltonian_radial(r, phi, delta, omega, n, phi0=0):
    return delta*r + omega*r**2 + r**(n/2)*np.cos(n*phi + phi0)

def get_delta_omega(dnu, p40tilde, pnrtilde, j0, n):
    delta = dnu/(pnrtilde*j0**(n/2-1))
    omega = p40tilde/(pnrtilde*j0**(n/2-2))
    return delta, omega


def readtfs(filename, usecols=None, index_col=0, check_lossbug=True):
    '''Reads twiss file into pandas df.'''
    header = {}
    nskip = 0
    closeit = False
    try:
        datafile = open(filename, 'r')
        closeit = True
    except TypeError:
        datafile = filename

    for line in datafile:
        nskip += 1
        if line.startswith('@'):
            entry = line.strip().split()
            header[entry[1]] = eval(' '.join(entry[3:]))
        elif line.startswith('*'):
            colnames = line.strip().split()[1:]
            break

    if closeit:
        datafile.close()

    table = pd.read_csv(filename, delim_whitespace = True,
                        skipinitialspace = True,
                        names = colnames, usecols = usecols,
                        index_col = index_col)

    if check_lossbug:
        try:
            table['ELEMENT'] = table['ELEMENT'].apply(lambda x: str(x).split()[0])
        except KeyError:
            pass
        try:
            for location in table['ELEMENT'].unique():
                if not location.replace(".","").replace("_","").replace('$','').isalnum():
                    print("WARNING: some loss locations in "+filename+
                          " don't reduce to alphanumeric values. For example "+location)
                    break
                if location=="nan":
                    print("WARNING: some loss locations in "+filename+" are 'nan'.")
                    break
        except KeyError:
            pass
        
    table = table.drop(index='$')
    cols = table.columns[1:]
    table = table.drop('N1', 1)
    table.columns = cols
    for col in table.columns[1:]:
        table[col] = pd.to_numeric(table[col],errors = 'coerce')
    table.columns= table.columns.str.strip().str.lower()
    return header, table

def HamiltonianContour(tdf, nu, nu_res, ele, title=''):   
    import matplotlib.cm as cm
    cmap = cm.get_cmap("coolwarm")
    # Compute contours
    dnu = nu - nu_res
    npoints = 500
    j0 = 0.0002

    beta = tdf.loc[ele]['beta11']
    alpha = tdf.loc[ele]['alfa11']
    mux_seh = 2*np.pi*tdf.loc[ele]['mu1']

    p3rtilde, mux_sext = get_p3rtilde(tdf, nu)
    p40tilde = get_p40tilde(tdf, nu)

    mux = -(mux_seh - mux_sext)

    #r = np.linspace(0, 4, npoints)
    r = 10**np.linspace(-10, 0.5, npoints)
    phi = np.linspace(-np.pi, np.pi, npoints)
    rr, phiphi = np.meshgrid(r, phi)

    factor = 1.2
    delta, omega = get_delta_omega(dnu, factor*p40tilde, p3rtilde, j0, n=3)
    hh = hamiltonian_radial(rr, phiphi, delta, omega, 3)

    jj = rr*j0
    phiphi, hh = phi1_to_phi(jj, phiphi, hh, nu, dnu, mux)
    ww, wdotwdot = j_to_w(jj, phiphi, nu)

    xx, xpxp = w_to_x(ww, wdotwdot, nu, alpha, beta)

    # Get contours in increasing order (the stable region is a valley)
    flat_h = hh.flatten()
    sorted_h = np.unique(np.sort(flat_h))
    size = sorted_h.shape[0]
    step = int(size/50)
    
    return(xx, xpxp, hh)


