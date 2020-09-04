import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emcee
#from IPython.display import display, Math
import corner
from scipy.optimize import minimize
import glob
from astroML.time_series import \
    lomb_scargle
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from astroML.stats import sigmaG
from scipy import stats
import sys
import argparse
from astropy.io import ascii
from astropy.table import Table, Column
from astropy import constants as c

'''
	Purpose
	--------
	Package to fit the TESS light curves from the FFIs on the PiGs project
	The needed folder structure is:
	plots/
		corner/
		chains/
		ajuste0/
		ajustemcmc/
	
	results/
		chains/
		chi2norm

	Example
	--------
		> python MCMCTICs_2parte.py 000000089770
	
	or if only the plots are wanted (once it has been run):
		> python MCMCTICs_2parte.py 000000089770 --PLOTS
'''

#------------------------funciones------------------------------------

def cli():
    """command line inputs 
    
    Get parameters from command line
        
    Returns
    -------
    Arguments passed by command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("tic", help="TIC Object name")
    parser.add_argument("-P", "--PLOTS", help="Only do plots", action="store_true", default=False)
    args = parser.parse_args()
    return args

#------------------------funciones------------------------------------

def limites(y):
    median_setlim = np.median(y)
    error_setlim = sigmaG(y)
    lim_sup = median_setlim + 5.*error_setlim
    lim_inf = median_setlim - 5.*error_setlim
    return lim_sup, lim_inf

def out_outliers(x,y,dy, lim_sup, lim_inf): 
    keep = np.where((y<lim_sup) & (y>lim_inf))[0]
    x, y, dy = x[keep], y[keep], dy[keep]
    return x,y,dy

def periodo_significativo(x, y, dy):
    period = 10 ** np.linspace(0, 2, 10000)  
    omega = 2 * np.pi / period  
    PS = lomb_scargle(x, y, dy, omega, generalized=True)
    power_p = period[np.argmax(PS)]
    double_power_p = 2. * power_p
    P_value = power_p 
    return P_value

def ajuste_t0(x, A, t0, P_value):  #A: amplitud, t0: fase
    flux_0_value = 1.0 #esta normalizado
    return flux_0_value + A * np.sin(2 * np.pi * (x - t0) / P_value)  

def ajuste_fase(x, y, P_value):
    Ampl_value = (np.max(y)-np.min(y))/2
    popt, pcov = curve_fit(lambda x,A,t0: ajuste_t0(x,A,t0,P_value) , x, y, 
                           p0=[Ampl_value, x[0]],
                          bounds=([0.0,x[0]-P_value],[2*Ampl_value,x[0]+P_value]))
    T0_value = popt[1]
    phi_value = (((x - T0_value) % P_value) / P_value)
    
    #----------------------------------------------------------
    if 1: #Para hacer los plots del ajuste0
        points = np.array([0, 0.25, 0.5, 0.75, 1])
        points_err = np.array(0.02 * np.ones(len(points)))
        #a, b = puntos_clave(y, dy, phi_value)
        
        fig = plt.figure(figsize=(6.9, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[1])
        gs.update(left=0.1, right=0.95, bottom=0.12, top=0.93, wspace=0.12, hspace=0.5)
    
        ax1 = plt.subplot(gs[0,0])
        ax1.set_title('Initial' + ' ' + 'lk'+ ' ' + 'TIC'+ TIC)
        ax1.plot(x,y,'o')
        ax1.plot(x,ajuste_t0(x,popt[0],popt[1],P_value))
        ax1.set_xlabel('Time (JD)')
        ax1.set_ylabel('Flux')
    
        ax2 = plt.subplot(gs[1, 0])  # flujos en funcion de la fase 
        ax2.set_title('Phase Diagram' + ' ' + 'TIC'+ TIC)
        ax2.errorbar(phi_value, y, 0, fmt='.', lw=1, c='gray', ecolor='gray', alpha=0.5)
        #ax2.errorbar(points, a, b, points_err, fmt='o', c='k', lw=1, ecolor='k')
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Flux')

    
        plt.savefig('plots/ajuste0/{}.png'.format(TIC))
        plt.close(fig)
    #----------------------------------------------------------
        
    return Ampl_value, T0_value, phi_value 


def binningstat(fase,y,dy,nbins, method='median', weighted=False, offphase=0.0, novoid=False):

    ybin, bin_edges, binnumber = stats.binned_statistic(fase, y, 'median', bins=nbins)
    sumeybin, _, _ = stats.binned_statistic(fase, dy**2, 'sum', bins=nbins)
    Nb, _, _ = stats.binned_statistic(fase, dy, 'count', bins=nbins)
    eybin = 1./Nb * np.sqrt(sumeybin)
    bin_width = (bin_edges[1] - bin_edges[0])
    xbin = bin_edges[1:] - bin_width/2

    return xbin, ybin, eybin

def priors_info (TIC, P_value, T0_value, Ampl_value):
    flux_0_value = 1.0
    P_elow = P_eup = 0.01
    T0_elow, T0_eup = T0_value-1, T0_value+1 
    Ampl_elow, Ampl_eup = 0 , +np.inf
    flux_0_eup = flux_0_elow = 0.01
    prior_values = [P_value, T0_value, Ampl_value, Ampl_value, Ampl_value, flux_0_value]
    prior_elow = [P_elow, T0_elow, Ampl_elow, Ampl_elow, Ampl_elow, flux_0_elow]
    prior_eup = [P_eup, T0_eup, Ampl_eup, Ampl_eup, Ampl_eup, flux_0_eup]
    param = ['P' , 'T0', 'A_r', 'A_e', 'A_b', 'flux_0']
    prior_form = ['G1', 'U1', 'U2', 'U3', 'U4', 'G2']
    prior_list = pd.DataFrame({'': param, 'values': prior_values,'elow': prior_elow,'eup': prior_eup, 'type': prior_form})
    if 0: 
        prior_txt = open('results/priors' + ' ' + TIC +'.txt', 'w')
        prior_txt.write(str(prior_list))
        prior_txt.close()
    return prior_list, param
            

def funcion_ajuste(theta, x): 
    P, T0, A_r, A_e, A_b, flux_0 = theta
    phi = 2*np.pi*(((x - T0) % P) / P)
    model = flux_0 - A_b*np.sin(phi+np.pi) + A_e*np.cos(2*phi+np.pi) + A_r*np.cos(phi+np.pi)
    return model

def log_likelihood(theta, x, y, dy): 
    P, T0, A_r, A_e, A_b, flux_0 = theta
    model = funcion_ajuste(theta, x)
    sigma2 = dy ** 2               
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(theta, prior_values, prior_max, prior_min, prior_type):
    P, T0, A_r, A_e, A_b, flux_0 = theta
    v_lp = []
    for types in prior_type: 
        #------------------------Priors uniformes--------------------
        if types == 'U1' : 
            if prior_min[1] < T0 < prior_max[1]:
                lp = 0.0
                v_lp = np.append(lp,v_lp)
            else: 
                lp = -np.inf
                v_lp =np.append(lp,v_lp)
        elif types =='U2': 
            if prior_min[2] < A_r < prior_max[2]: 
                lp = 0.0
                v_lp = np.append(lp,v_lp)
            else: 
                lp = -np.inf
                v_lp = np.append(lp,v_lp)
        elif types == 'U3': 
            if prior_min[3] < A_e < prior_max[3]: 
                lp = 0.0
                v_lp = np.append(lp,v_lp)
            else: 
                lp = -np.inf
                v_lp = np.append(lp,v_lp)
        elif types =='U4': 
            if prior_min[4] < A_b < prior_max[4]: 
                lp = 0.0
                v_lp = np.append(lp,v_lp)
            else: 
                lp = -np.inf
                v_lp = np.append(lp,v_lp)
        #---------------Priors gausianos-------------------------
        elif types == 'G1':
            sigma_P = prior_min[0]
            sigma_flux_0 = prior_min[5]
            lp_P = 1.0/(sigma_P*np.sqrt(2.0*np.pi)) * np.exp(-(P-prior_values[0])**2/(2.0*sigma_P**2))
            lp = lp_P
            v_lp =np.append(lp,v_lp)
        elif types =='G2': 
            sigma_flux_0 = prior_min[5]
            lp_flux_0 = 1.0/(sigma_flux_0*np.sqrt(2.0*np.pi)) * np.exp(-(flux_0-prior_values[5])**2/(2.0*sigma_flux_0**2))
            lp = lp_flux_0
            v_lp = np.append(lp,v_lp)
    return sum(v_lp)
            

def log_probability(theta, x, y, dy, prior_values, prior_max, prior_min, prior_type):
    lp = log_prior(theta, prior_values, prior_max, prior_min, prior_type)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, dy)


def plots_ajparam (samples, ndim): 
    fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
    #samples = sampler.chain#[:, :, :].reshape((-1, ndim)) #sampler.get_chain()
    #samples = sampler.get_chain()
    labels = ["P", "T0", "A_r", "A_e", "A_b", "flux_0"]
    print np.shape(samples)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(np.transpose(samples[:, :, i]), "k", alpha=0.3)
        #ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.savefig('plots/chains/{}.png'.format(TIC))
    plt.close()
    return labels

def plots_prob(chain, labels, prior_values, ndim,burning): 
    #flat_samples = sampler.get_chain(discard=burning, thin=15, flat=True)
    flat_samples = chain[:, burning:, :].reshape((-1, ndim))
    fig = corner.corner(flat_samples, labels=labels); 
    media = np.mean(flat_samples, axis=0)
    mediana = np.median(flat_samples, axis=0)
    axes = np.array(fig.axes).reshape((ndim,ndim))
    for i in range(ndim): 
        ax = axes[i,i]
        ax.axvline(media[i], color = 'g')
        ax.axvline(mediana[i], color = 'r')
    for yi in range(ndim): 
        for xi in range(yi): 
            ax = axes[yi, xi]
            ax.axvline(media[xi], color="g")
            ax.axvline(mediana[xi], color="r")
            ax.axhline(media[yi], color="g")
            ax.axhline(mediana[yi], color="r")
            ax.plot(media[xi], media[yi], "sg")
            ax.plot(mediana[xi], mediana[yi], "sr")
    plt.savefig('plots/corner/{}.png'.format(TIC))
    plt.close()
    return flat_samples

def final_param(ndim, flat_samples, labels, param): 
    ajuste_values = []
    ajuste_eup = []
    ajuste_edown = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        ajuste_values = np.append(ajuste_values, mcmc[1]) 
        ajuste_eup = np.append(ajuste_eup, q[0])
        ajuste_edown = np.append(ajuste_edown, q[1])        
    ajuste_list = pd.DataFrame({'Parametros': param, 'values': ajuste_values, 'eup': ajuste_eup, 'edown': ajuste_edown })
    
    if 0: 
        ajuste_txt = open('results/ajustes' + ' ' + TIC +'.txt', 'w')
        ajuste_txt.write(str(ajuste_list))
        ajuste_txt.close()
        
    return ajuste_list, ajuste_values

def plot_ajuste_mcmc(x,y,ajuste_values):
    #Ajuste mcmc: 
    P = ajuste_values[0]
    t0 = ajuste_values[1]
    fase = ((x - t0) % P)/P
    theta = 2*np.pi*fase
    F0 = ajuste_values[5]
    A_b = ajuste_values[4]
    A_e = ajuste_values[3]
    A_r = ajuste_values[2]
    ajuste = F0 - A_b*np.sin(theta+np.pi) + A_e*np.cos(2*theta+np.pi) + A_r*np.cos(theta+np.pi)
    z = np.linspace(0,1,len(y))
    ajuste_total = F0 - A_b * np.sin(2*np.pi *z + np.pi) + A_e * np.cos(4*np.pi*z + np.pi) + A_r * np.cos(2*np.pi*z + np.pi)
    ajuste_e = F0 + A_e * np.cos(4 * np.pi * z +np.pi)
    ajuste_b = F0 - A_b * np.sin(2 * np.pi * z +np.pi)
    ajuste_r = F0 + A_r * np.cos(2 * np.pi * z +np.pi)
    #Puntos en fase: 
    points = np.array([0, 0.25, 0.5, 0.75, 1])
    points_err = np.array(0.02 * np.ones(len(points)))
    xbin1, ybin1, eybin1 = binningstat(fase, y, dy, 15)
    
    fig = plt.figure(figsize = (6.9, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios = [1, 1, 1], width_ratios = [1])
    gs.update(left = 0.1, right = 0.95, bottom = 0.10, top = 0.95, wspace = 0.12, hspace = 0.3)
    
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_title('MCMC' + ' ' + 'lk' + ' ' + 'TIC' + TIC)
    ax1.errorbar(x, y, yerr=dy, fmt = 'o', c = 'gray', lw = 1, alpha = 0.2)
    xb, yb, eyb = binningstat(x, y, dy, int(len(x)/10))
    ax1.errorbar(xb, yb, eyb, fmt = 'o', c = 'k', lw = 1)
    ax1.plot(x, ajuste, c = 'red', lw = 2)
    ax1.set_xlabel('Time (JD)')
    ax1.set_ylabel('Flux')

    ax2 = plt.subplot(gs[1, 0])
    ax2.errorbar(fase, y, 0, fmt = '.', lw = 1, c = 'gray', ecolor = 'gray', alpha = 0.2)
    ax2.errorbar(xbin1, ybin1, yerr=eybin1, fmt = 'o', c = 'k', lw = 1, ecolor = 'k', alpha = 1)
    ax2.plot(z, ajuste_total, c ='red', lw = 2)
    ax2.plot(z, ajuste_e, c = 'mediumvioletred', linestyle ='--', lw = 1)
    ax2.plot(z, ajuste_b, c = 'cornflowerblue', linestyle ='--', lw = 1)
    ax2.plot(z, ajuste_r, c = 'mediumaquamarine', linestyle ='--', lw = 1)
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Flux')
 
    ax3 = plt.subplot(gs[2, 0])
    ax3.errorbar(xbin1, ybin1, yerr=eybin1, fmt = 'o', c = 'k', lw = 1, ecolor = 'k', alpha = 1)
    ax3.plot(z, ajuste_total, c ='red', lw = 2)
    ax3.plot(z, ajuste_e, c = 'mediumvioletred', linestyle ='--', lw = 1)
    ax3.plot(z, ajuste_b, c = 'cornflowerblue', linestyle ='--', lw = 1)
    ax3.plot(z, ajuste_r, c = 'mediumaquamarine', linestyle ='--', lw = 1)
    ax3.set_xlabel('Phase')
    ax3.set_ylabel('Flux')
   
    plt.savefig('plots/ajustemcmc/{}.png'.format(TIC))
    plt.close(fig)

    chi2norm = 1./len(x) * np.sum( (y-ajuste)**2/dy**2 )
    max_mplanet = A_e * 1.*c.M_sun * 2.**3 / c.M_jup # Assuming Ms = 1Msun, a/Rs = 2
    
    data = Table([np.atleast_1d(np.array(chi2norm)), np.atleast_1d(np.array(max_mplanet))], names=['# chi2norm', 'max_mplanet'])
    ascii.write(data, 'results/chi2/' + TIC +'_chi2norm.txt',format ='tab')


    
def results_txt(prior_list, ajuste_list, param):
    difer = diferencia_param(prior_list, ajuste_list)
    final_list = pd.DataFrame({'Parametros': param, 'priors': prior_list['values'],
                               'MCMC_values': ajuste_list['values'], 'DIF': difer,
                               'MCMC_eup': ajuste_list['eup'], 'MCMC_edown': ajuste_list['edown']})
    data = Table([param, ajuste_list['values'], ajuste_list['edown'], ajuste_list['eup']], names=['# Param', 'Value','e_low','e_upp'])
    ascii.write(data, 'results/' + TIC +'.txt',format ='tab')

#     final_txt = open('results/' + TIC +'.txt', 'w')
#     final_txt.write(str(final_list))
#     final_txt.close()
    return final_list

def diferencia_param(prior_list, ajuste_list): 
    difer = np.zeros(6)
    for i in range(0,6): 
        difer[i] = abs(prior_list['values'][i]-ajuste_list['values'][i])
    return difer    
    
#-------------------------------Comienza la ejecucion-------------------------------

args = cli()

TIC = args.tic
path_to_data = './data/PIGS-LCS/'
x, y, dy = [], [], []
for files in glob.glob(path_to_data+"lc_" + str(TIC) + "*.npz"):
    lc = np.load(files)
    x.append(lc['time_flat'])
    y.append(lc['flux_flat'])
    dy.append(lc['ferr_flat'])  
    

x  = np.concatenate(x).ravel()   
y  = np.concatenate(y).ravel()  
dy = np.concatenate(dy).ravel()
srt = np.argsort(x)
x, y, dy = x[srt], y[srt], dy[srt]

lim_sup, lim_inf = limites(y)
x,y,dy = out_outliers(x,y,dy,lim_sup,lim_inf)
norm = np.median(y)
y /= norm
dy /= norm
P_value = periodo_significativo(x, y, dy)
Ampl_value, T0_value, phi_value = ajuste_fase(x, y, P_value) 
prior_list, param = priors_info (TIC, P_value, T0_value, Ampl_value) 

prior_values = np.array(prior_list['values'])
prior_max = np.array(prior_list['eup'])
prior_min = np.array(prior_list['elow'])
prior_type = np.array(prior_list['type'])

nwalkers, ndim, nsteps = 30, 6, 5000

if args.PLOTS == False:
	pos = prior_values + 1e-4 * np.random.randn(nwalkers, ndim)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
									args=(x, y, dy, prior_values, prior_max, prior_min, prior_type))	
	burning_phase = True							
	
	if burning_phase: 
		#Runing 1st burn-in:
		p0,lnp,_ = sampler.run_mcmc(pos,nsteps)
		sampler.reset()
		
		#Runing 2nd burn-in: 
		p = p0[np.argmax(lnp)]
		p0 = [p + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]
		p0,_,_ = sampler.run_mcmc(p0, nsteps/2)
		sampler.reset()
		
		#Runing production: 
		p0,_,_ = sampler.run_mcmc(p0, nsteps/2)
		burning = 0
	else: 
		p0,lnp,_ = sampler.run_mcmc(pos,nsteps)
		burning = int(0.5*nsteps)
		
	chains = sampler.chain
	np.savez('results/chains/' + TIC ,chains=chains, burning=burning)

else:
	tmp = np.load('results/chains/' + TIC+'.npz')
	chains = tmp['chains']
	burning = tmp['burning']

labels = plots_ajparam(chains,ndim)
flat_samples = plots_prob(chains, labels, prior_values, ndim,burning)
ajuste_list, ajuste_values = final_param(ndim, flat_samples, labels, param)
plot_ajuste_mcmc(x,y,ajuste_values)
final_list = results_txt(prior_list, ajuste_list, param)





