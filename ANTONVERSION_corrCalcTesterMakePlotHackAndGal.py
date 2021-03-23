#!/Users/antonbaleatolizancos/anaconda/envs/lensing_biases/bin/python
import numpy
import pickle
import matplotlib.pyplot as plt
import numpy
from   matplotlib.pylab import *
from   matplotlib.font_manager import FontProperties

def find_nearest(xout,xin,yin):
    idx = (numpy.abs(xin-xout)).argmin()
    return yin[idx]

filedir = 'filesForAntonFinal/'
mat = numpy.loadtxt(filedir+'simderiv_del_CBB_del_Cphiphi_theo.dat')
lmatA = numpy.loadtxt(filedir+'simderiv_del_CBB_del_Cphiphi_theo.dat_lvec')

lmat = pickle.load(open(filedir+'lKappaI.pkl'))
lmati = lmat.copy()
deltaL = lmat[3]-lmat[2]
lmat += deltaL/2.

X = numpy.loadtxt(filedir+'planckFINAL_lensedCls.dat')
Y = numpy.loadtxt(filedir+'planckFINAL_scalCls.dat')
Z = numpy.loadtxt(filedir+'planckFINALTensors_tensCls.dat')

errorRatio = pickle.load(open(filedir+'errorRatioPlanck.pkl'))

TCMB = 2.726e6

lr = Z[:,0]
lsqClOver2piBBr = Z[:,3]
clBBr = lsqClOver2piBBr*(2*numpy.pi)/(lr*(lr+1.0))
clBBr /= TCMB**2
ll = X[:,0]
l = ll
lsqClOver2piBB = X[:,3]
clBB = lsqClOver2piBB*(2*numpy.pi)/(l*(l+1.0))
clBB /=TCMB**2
lyy = Y[:,0]
clkap = Y[:,4]/(4.*TCMB**2)
clphi = clkap/lyy**(4.)*4.
phiBin = numpy.interp(lmatA,lyy,clphi)


plt.clf()
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3,3))
fig_width_pt = 246.0*1.4  # Get this from LaTeX using \the\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch       


golden_mean = (numpy.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean*1.4      # height in inches
fig_size =  [fig_width,fig_height]
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
params = {'axes.labelsize': 11,
          'font.size':  11,
          'legend.fontsize': 11,
          'xtick.labelsize': 8.5,
          'ytick.labelsize': 8.5,
          'figure.figsize': fig_size,
          'font.family':'serif'}
plt.rcParams.update(params)
plt.clf()
fig = plt.figure()
axes = plt.Axes(fig,[.15,.17,.7,.7])
fig.add_axes(axes)
axes.yaxis.set_major_formatter(formatter)


clBBA = numpy.dot((mat),phiBin/TCMB**2.)

intEr =lmatA.copy()
intEr2 =lmatA.copy()

# Added by Anton. Initialise b-type errors
b_intEr =lmatA.copy()
b_intEr2 =lmatA.copy()

cross_spectra_of_tracers_gals = numpy.load('cross_spectra_w_phi_galaxiesCIBinternal.npy')[0] #First entry is cross with galaxies, second is with CIB, third with internal
cross_spectra_of_tracers_cib = numpy.load('cross_spectra_w_phi_galaxiesCIBinternal.npy')[1] #First entry is cross with galaxies, second is with CIB, third with internal
cross_spectra_of_tracers_int = numpy.load('cross_spectra_w_phi_galaxiesCIBinternal.npy')[2] #First entry is cross with galaxies, second is with CIB, third with internal
#print(3,3001)
clKI = numpy.interp(lmatA,numpy.arange(3001),cross_spectra_of_tracers_cib)
clKg = numpy.interp(lmatA,numpy.arange(3001),cross_spectra_of_tracers_gals)
clKK = numpy.interp(lmatA,numpy.arange(3001),cross_spectra_of_tracers_int)

print()
#print('worked')
#sys.exit()
multitracer_weights = numpy.load('matrix_of_tracer_weights.npy') #First index refers to tracer, second to multipole
cg = numpy.interp(lmatA,numpy.arange(3001),multitracer_weights[0])
cI = numpy.interp(lmatA,numpy.arange(3001),multitracer_weights[1])
cK = numpy.interp(lmatA,numpy.arange(3001),multitracer_weights[2])

#totalCross = cg*clKg+cI*clKI+cK*clKK

print(clKg,clKI,clKK,'spectra <')

print(cg,cI,cK,'coefficients <')

totalCross = cg*clKg+cI*clKI+cK*clKK
print(totalCross,'total cross is <')

multiCorrection = numpy.nan_to_num(cI/totalCross*clKI)
print(multiCorrection,"this is the coefficient", numpy.mean(multiCorrection))

multiCorrectionKg = numpy.nan_to_num(cg/totalCross*clKg)

# ANTON: replace the CIB correlation with the correlation of the co-added tracer. We use the approximation that C^{\kappa I^{tot}} \approx C^{I^{tot} I^{tot}}
rho = np.sqrt(totalCross/clKK)#pickle.load(open(filedir+'corr545FINAL.pkl'))
lowF = np.ones(rho.shape)
print 'these are the ells'
print lmatA
lowF[lmatA<60.] = 0.
rho *= lowF
rhoAlt = rho.copy()

# ANTON: Change the fiducial delensed power as well (from CIB only to co-added)
clBBAD545 = numpy.dot((mat),(1.- rho**2.)*phiBin/TCMB**2.)#numpy.dot((mat),(1.-numpy.interp(lmatA,rho[0],lowF*rho[1]**2.))*phiBin/TCMB**2.)

# ANTON: plot in units of uK
K_to_uk = 1e6

#sys.exit()
for i in xrange(5000):
    # ANTON: right now, all measurements have the same error (from Planck)
    errReals = numpy.random.randn(len(errorRatio[0]))*errorRatio[1]
    errReals2 = numpy.random.randn(len(errorRatio[0]))*errorRatio[1]
    errReals3 = numpy.random.randn(len(errorRatio[0]))*errorRatio[1]
    errReals4 = numpy.random.randn(len(errorRatio[0]))*errorRatio[1]

    for j in xrange(len(lmatA)):
        intEr[j] = find_nearest(lmatA[j],errorRatio[0],errReals)
        intEr2[j] = find_nearest(lmatA[j],errorRatio[0],errReals2)

        b_intEr[j] = find_nearest(lmatA[j],errorRatio[0],errReals3)
        b_intEr2[j] = find_nearest(lmatA[j],errorRatio[0],errReals4)
#        print intErs
    intEr[lmatA>1500.] = 0.
    # ANTON: changed this line
    clBBADRand = numpy.dot(mat,(1.-rhoAlt**2. *(1.+multiCorrection*(intEr*2.-b_intEr)+multiCorrectionKg*(intEr2*2.-b_intEr2)))*phiBin/TCMB**2.)
    #plt.plot(lmatA,K_to_uk**2 *(clBBADRand-clBBAD545),color='b', lw=1)
    np.save('residuals_after_delensing_for_propagating_to_r_bias/alltracers_residual_spec_'+str(i)+'_in_uK', K_to_uk**2 *(clBBADRand-clBBAD545))

# ANTON: edit legend to clarify that we're varying both auto- and cross- spectra
plt.plot(lmatA,K_to_uk**2 * (clBBADRand-clBBAD545),label=r'perturbing $C_l^{\kappa I}$ and $C_l^{II}$',color='b', lw=1)
# ANTON: plot x=0 line
plt.axhline(0, lw=0.5, color='gray')

plt.plot(lr,K_to_uk**2 * clBBr*0.001,label=r'$C_l^{BB,\mathrm{primordial}}(r=0.001)$',linewidth=2,color='k')
plt.legend(loc='best')
print clBBA.shape
print lmatA.shape
plt.ylim(-1.5e-8,3.e-8)#plt.ylim((-1.5e-7,2e-7))
plt.xlim(20,450)
plt.ylabel(r'$C_l^{BB,\mathrm{res}}-C_l^{BB,\mathrm{res},\mathrm{fiducial}}$ [$\mu\mathrm{K}^2$]')
plt.xlabel(r'$l$')

#plt.savefig('ANTONlensBcompareErrorFINALPostHackAndGal.pdf',bbox_inches='tight', dpi=600)

plt.clf()
print lmat.shape, intEr.shape
plt.plot(lmatA,intEr)
plt.plot(rho[0],rho[1])
plt.xlim(50,3000.)
#plt.savefig('ANTONrhoAndErr.png')
