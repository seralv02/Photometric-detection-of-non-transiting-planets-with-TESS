#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from astroML.time_series import     lomb_scargle
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from astroML.stats import sigmaG

#-------------------------Functions definitions:---------------------------------

#1-signal limits: 
def limites(y):                          
    median_setlim = np.median(y)
    error_setlim = sigmaG(y)
    lim_sup = median_setlim + 5.*error_setlim
    lim_inf = median_setlim - 5.*error_setlim
    return lim_sup, lim_inf

#2-Removing ouliers:
def out_outliers(x,y,dy, lim_sup, lim_inf): 
    keep = np.where((y<lim_sup) & (y>lim_inf))[0]
    x, y, dy = x[keep], y[keep], dy[keep]
    return x,y,dy

#3-significative period of the LC:
def periodo_significativo(x, y, dy):
    period = 10 ** np.linspace(0, 2, 10000)  # starting around 1 day until 10^2 for find the 2nd period
    omega = 2 * np.pi / period   
    PS = lomb_scargle(x, y, dy, omega, generalized=True)
    power_p = period[np.argmax(PS)]
    double_power_p = 2. * power_p
    return PS, power_p, double_power_p, period

#4-Fiting the time of conjunction
def ajuste_t0(x, A, t0, power_p):  #A: amplitude, t0: time of conjunction
    return 1.0 + A * np.sin(2 * np.pi * (x - t0) / power_p)  

#5- orbital phase
def ajuste_fase(x, y, power_p):
    ampl = (np.max(y)-np.min(y))/2
    popt, pcov = curve_fit(lambda x,A,t0: ajuste_t0(x,A,t0,power_p) , x, y, 
                           p0=[ampl, x[0]],
                          bounds=([0.0,x[0]-power_p],[2*ampl,x[0]+power_p]))
    phi1 = (((x - popt[1]) % power_p) / power_p)
    phi2 = (((x - popt[1]) % double_power_p) / double_power_p)
    return phi1, phi2

#6-flux at the four orbital phase main locations (0,0.25,0.5,0.75)
def puntos_clave(y, dy, phi1):
    elem1 = np.where((phi1 < 0.02))[0]
    f1 = np.median(y[elem1])
    e1 = 1. / len(elem1) * np.sqrt(np.sum(dy[elem1] ** 2))  

    elem2 = np.where((phi1 > 0.23) & (phi1 < 0.27))[0]
    f2 = np.median(y[elem2])
    e2 = 1. / len(elem2) * np.sqrt(np.sum(dy[elem2] ** 2))

    elem3 = np.where((phi1 > 0.48) & (phi1 < 0.52))[0]
    f3 = np.median(y[elem3])
    e3 = 1. / len(elem3) * np.sqrt(np.sum(dy[elem3] ** 2))

    elem4 = np.where((phi1 > 0.73) & (phi1 < 0.77))[0]
    f4 = np.median(y[elem4])
    e4 = 1. / len(elem4) * np.sqrt(np.sum(dy[elem4] ** 2))

    elem5 = np.where((phi1 > 0.98) & (phi1 < 1.02))[0]   #equivalent to f at orbital phase 0 (f1)
    f5 = np.median(y[elem5])
    e5 = 1. / len(elem5) * np.sqrt(np.sum(dy[elem5] ** 2))

    a = np.array([f1, f2, f3, f4, f5])  #flux vector at the four (five*) orbital phase locations
    b = np.array([e1, e2, e3, e4, e5])  #flux error 
    return a, b

#7- The same as 6 but for 2P, so double the orbital phase locations
def puntos_clave2(y, dy, phi2):
    elem1 = np.where((phi2 < 0.01))[0]
    f1 = np.median(y[elem1])
    e1 = 1. / len(elem1) * np.sqrt(np.sum(dy[elem1] ** 2))  # puntos de fase relevantes con 2P

    elem2 = np.where((phi2 > 0.115) & (phi2 < 0.135))[0]
    f2 = np.median(y[elem2])
    e2 = 1. / len(elem2) * np.sqrt(np.sum(dy[elem2] ** 2))

    elem3 = np.where((phi2 > 0.365) & (phi2 < 0.385))[0]
    f3 = np.median(y[elem3])
    e3 = 1. / len(elem3) * np.sqrt(np.sum(dy[elem3] ** 2))

    elem4 = np.where((phi2 > 0.615) & (phi2 < 0.635))[0]
    f4 = np.median(y[elem4])
    e4 = 1. / len(elem4) * np.sqrt(np.sum(dy[elem4] ** 2))

    elem5 = np.where((phi2 > 0.8025) & (phi2 < 0.8225))[0]
    f5 = np.median(y[elem5])
    e5 = 1. / len(elem5) * np.sqrt(np.sum(dy[elem5] ** 2))

    elem6 = np.where((phi2 > 0.99) & (phi2 < 1.01))[0]
    f6 = np.median(y[elem6])
    e6 = 1. / len(elem5) * np.sqrt(np.sum(dy[elem5] ** 2))

    a2 = np.array([f1, f2, f3, f4, f5, f6])  
    b2 = np.array([e1, e2, e3, e4, e5, e6])
    return a2, b2

#8-preliminary analisis for TICs classification
def test_candidatos(a, b):  # candidates or discards
    candidatos = np.array([])

    if (a[3] + b[3] < a[4] - b[4]) or (a[1]-b[1]>a[0]+b[0]):      #ya que podemos tener excentricidad en las orbitas
        respuesta1 = 1.0  # opcion 1: A candidato le asigno una respuesta=1
        respuesta = respuesta + respuesta1
    else:
        if a[3] - b[3] > a[4] + b[4]:
            respuesta3 = 1.1  # opcion3: A candidato con P1 le asigno una respuesta=1.1
            respuesta = respuesta + respuesta3
        else:
            respuesta4 = 0.1  # opcion4: No es candidato y le asigno una respuesta 0.1
            respuesta = respuesta + respuesta4
    return respuesta

#9-candidates plots
def plots_candidatos(TIC, y, dy, phi1, a, b, period, power_p, double_power_p):  
    fase = np.array([0, 0.25, 0.5, 0.75, 1])
    fase_err = np.array(0.02 * np.ones(len(fase)))

    fig = plt.figure(figsize=(6.9, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[1])
    gs.update(left=0.1, right=0.95, bottom=0.12, top=0.93, wspace=0.12, hspace=0.5)

    ax1 = plt.subplot(gs[0, 0], xscale='log')  # periodograma en escala logarítmica (ejex)
    ax1.set_title('Periodograma TIC' + TIC)
    ax1.plot(period, PS, '-', color='black', lw=1, zorder=1)
    ax1.axvline(power_p, c='red', linestyle='--')
    ax1.text(power_p, max(PS), str(power_p)[0:5])
    ax1.axvline(double_power_p, color='green', linestyle='--')
    ax1.text(double_power_p, max(PS), str(double_power_p)[0:5])
    ax1.set_xlabel(r'Period (days)')
    ax1.set_ylabel('Power')
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
    ax1.xaxis.set_major_locator(plt.LogLocator(10))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5g'))

    ax2 = plt.subplot(gs[1, 0])  # flujos en función de la fase 
    ax2.set_title('D.Fase P:' + str(power_p)[0:5] + '_TIC_' + TIC)
    ax2.errorbar(phi1, y, 0, fmt='.', lw=1, c='gray', ecolor='gray', alpha=0.5)
    ax2.errorbar(fase, a, b, fase_err, fmt='o', c='k', lw=1, ecolor='k')

    plt.savefig('plots/candidatos/TIC{}.png'.format(TIC))
    plt.close(fig)

#10-2P candidates 
def plots_candidatosP1(TIC, y, dy, phi1, phi2, a, b, a2, b2, period, power_p, double_power_p):  
    fase = np.array([0, 0.25, 0.5, 0.75, 1])  # puntos de fase con P
    fase2 = np.array([0, 0.125, 0.375, 0.625, 0.8125, 1])  # Puntos de fase con 2P
    fase_err = np.array(0.02 * np.ones(len(fase)))
    fase_err2 = np.array(0.02 * np.ones(len(fase2)))

    fig = plt.figure(figsize=(6.9, 6))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], width_ratios=[1])
    gs.update(left=0.1, right=0.95, bottom=0.12, top=0.93, wspace=0.12, hspace=0.7)

    ax1 = plt.subplot(gs[0, 0], xscale='log')  # periodograma en escala logarítmica (ejex)
    ax1.set_title('Periodograma TIC' + TIC)
    ax1.plot(period, PS, '-', color='black', lw=1, zorder=1)
    ax1.axvline(power_p, c='red', linestyle='--')
    ax1.text(power_p, max(PS), str(power_p)[0:5])
    ax1.axvline(double_power_p, color='green', linestyle='--')
    ax1.text(double_power_p, max(PS), str(double_power_p)[0:5])
    ax1.set_xlabel(r'Period (days)')
    ax1.set_ylabel('Power')
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
    ax1.xaxis.set_major_locator(plt.LogLocator(10))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5g'))

    ax2 = plt.subplot(gs[1, 0])  # flujos en fase para P
    ax2.set_title('D.Fase P:' + str(power_p)[0:5] + ' TIC ' + TIC)
    ax2.errorbar(phi1, y, 0, fmt='.', lw=1, c='gray', ecolor='gray', alpha=0.5)
    ax2.errorbar(fase, a, b, fase_err, fmt='o', c='k', lw=1, ecolor='k')

    ax3 = plt.subplot(gs[2, 0])  # flujos en fase con 2P
    ax3.set_title('D.Fase 2xP:' + str(double_power_p)[0:5] + ' TIC ' + TIC)
    ax3.errorbar(phi2, y, 0, fmt='.', lw=1, c='gray', ecolor='gray', alpha=0.5)
    ax3.errorbar(fase2, a2, b2, fase_err2, fmt='o', c='k', lw=1, ecolor='k')

    plt.savefig('plots/candidatosP1/TIC{}.png'.format(TIC))
    plt.close(fig)

#11. discarded plots    
def plots_descartados(TIC, y, dy, phi1, a, b, period, power_p, double_power_p):
    fase = np.array([0, 0.25, 0.5, 0.75, 1])
    fase_err = np.array(0.02 * np.ones(len(fase)))

    fig = plt.figure(figsize=(6.9, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[1])
    gs.update(left=0.1, right=0.95, bottom=0.12, top=0.93, wspace=0.12, hspace=0.5)

    ax1 = plt.subplot(gs[0, 0], xscale='log')
    ax1.set_title('Periodograma TIC' + TIC)
    ax1.plot(period, PS, '-', color='black', lw=1, zorder=1)
    ax1.axvline(power_p, c='red', linestyle='--')
    ax1.text(power_p, max(PS), str(power_p)[0:5])
    ax1.axvline(double_power_p, color='green', linestyle='--')
    ax1.text(double_power_p, max(PS), str(double_power_p)[0:5])
    ax1.set_xlabel(r'Period (days)')
    ax1.set_ylabel('Power')
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
    ax1.xaxis.set_major_locator(plt.LogLocator(10))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5g'))

    ax2 = plt.subplot(gs[1, 0])
    ax2.set_title('D.Fase P:' + str(power_p)[0:5] + ' TIC ' + TIC)
    ax2.errorbar(phi1, y, 0, fmt='.', lw=1, c='gray', ecolor='gray', alpha=0.5)
    ax2.errorbar(fase, a, b, fase_err, fmt='o', c='k', lw=1, ecolor='k')
    
    plt.savefig('plots/descartados/TIC{}.png'.format(TIC))
    plt.close(fig)

#----------------------------------------------------------------------------------------

#lists, arrays and variables initialization
elementos = []
fail1 = np.array([])
fail2 = np.array([])
candidatos = np.array([])
candidatosP1 = np.array([])
errorlectura = np.array([])
error_concatenate = np.array([])
c_candidatos = 0
c_candidatosP1 = 0
c_descartados = 0
c_fail1 = 0
c_fail2 = 0
c_errorlectura = 0
c_error_concatenate = 0
d = open('results/TICSdescartados.txt', 'w')

#---------------------------------------------Code execution----------------------------------------

for files in glob.glob("*.npz"):
    elementos.append(files.split('_')[1])  
targets = np.array(elementos)
TICS = np.unique(targets)

for i,TIC in enumerate(TICS):
    print(i,TIC)
    x, y, dy = [], [], []
    for files in glob.glob("lc_" + str(TIC) + "*.npz"):
        try:
            lc = np.load(files)
            x.append(lc['time_flat'])
            y.append(lc['flux_flat'])
            dy.append(lc['ferr_flat'])
        except:
            errorlectura = np.append(errorlectura,TIC)
            continue
    #x, y, dy son listas de arrays, los convierto arrays unicos:  
    try:
        x  = np.concatenate(x)   
        y  = np.concatenate(y)   
        dy = np.concatenate(dy)  
        lim_sup, lim_inf = limites(y)
        x,y,dy = out_outliers(x,y,dy,lim_sup,lim_inf)
        norm = np.median(y)
        y /= norm
        dy /= norm
    except:
        error_concatenate = np.append(error_concatenate, TIC)
        c_error_concatenate = c_error_concatenate+1
        continue
    #timespan = max(x)-min(x)
    #print('timespan:', timespan,TIC)
    PS, power_p, double_power_p, period = periodo_significativo(x, y, dy)
    phi1, phi2 = ajuste_fase(x, y, power_p)  
    try:
        a, b = puntos_clave(y, dy, phi1)
        respuesta = test_candidatos(a, b)
    except:
        c_fail1 = c_fail1 + 1
        fail1 = np.append(fail1, TIC)
        continue
        
    if respuesta == 1.0:  # candidatos
        candidatos = np.append(candidatos, TIC)
        c_candidatos = c_candidatos + 1
        plots_candidatos(TIC, y, dy, phi1, a, b, period, power_p, double_power_p)

    elif respuesta == 1.1:  # candidatos con P1
        try:
            a2, b2 = puntos_clave2(y, dy, phi2)
            plots_candidatosP1(TIC, y, dy, phi1, phi2, a, b, a2, b2, period, power_p, double_power_p)
            candidatosP1 = np.append(candidatosP1, TIC)
            c_candidatosP1 = c_candidatosP1 + 1
        except:
            c_fail2 = c_fail2 + 1
            fail2 = np.append(fail2, TIC)
            continue
    else:  # descartados
        try:
            plots_descartados(TIC, y, dy, phi1, a, b, period, power_p, double_power_p)
        except: 
            _tmp=2
        c_descartados = c_descartados + 1
        d.write(str(TIC) + ' ' + str(respuesta) + '\n')

d.close()
       
Total = c_fail1 + c_fail2 + c_errorlectura + c_candidatos + c_candidatosP1 + c_descartados
print('Total:', Total)
print('Nº Candidatos :', c_candidatos, '|', float((c_candidatos / Total) * 100), '%')
print('Nº Candidatos con P1:', c_candidatosP1, '|', float((c_candidatosP1 / Total) * 100), '%')
print('Nº Descartados:', c_descartados, '|', float((c_descartados / Total) * 100), '%')
print('Nº de errores_lectura:', c_errorlectura, '|', float((c_errorlectura / Total) * 100), '%')
print('Nº de fails1:', c_fail1, '|', float((c_fail1 / Total) * 100), '%')
print('Nº de fails2:', c_fail2, '|', float((c_fail2 / Total) * 100), '%')
print('Nº de error_concatenate:', c_error_concatenate, '|', float((c_error_concatenate / Total) * 100), '%')

e = open('results/TICSerrorlectura.txt', 'w')
for el in errorlectura:
    e.write(str(el) + '\n')
e.close()
econ = open('results/TICSerror_concatenate.txt', 'w')
for ec in error_concatenate:
    econ.write(str(ec) + '\n')
econ.close()
f = open('results/TICSfail1.txt', 'w')
for f1 in fail1:
    f.write(str(f1) + '\n')
f.close()
c = open('results/TICScandidatos.txt', 'w')
for cd in candidatos:
    c.write(str(cd) + '\n')
c.close()
cP1 = open('results/TICScandidatosP1.txt', 'w')
for cdP1 in candidatosP1:
    cP1.write(str(cdP1) + '\n')
cP1.close()
ff = open('results/TICSfail2.txt', 'w')
for f2 in fail2:
    ff.write(str(f2) + '\n')
ff.close()

