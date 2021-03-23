#!/Users/antonbaleatolizancos/anaconda/envs/B_temp/bin/python
import numpy
import pickle
import pylab
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


pylab.clf()
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3,3))
fig_width_pt = 246.0*1.4  # Get this from LaTeX using \the\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch       

golden_mean = (numpy.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean*1.4      # height in inches
fig_size =  [fig_width,fig_height]
params = {'axes.labelsize': 11,
          'font.size':  11,
          'legend.fontsize': 11,
          'xtick.labelsize': 8.5,
          'ytick.labelsize': 8.5,
          'figure.figsize': fig_size,
          'font.family':'serif'}
pylab.rcParams.update(params)
pylab.clf()
fig = pylab.figure()
axes = pylab.Axes(fig,[.15,.17,.7,.7])
fig.add_axes(axes)
axes.yaxis.set_major_formatter(formatter)


clBBA = numpy.dot((mat),phiBin/TCMB**2.)
rho = pickle.load(open(filedir+'corr545FINAL.pkl'))
lowF = rho[0]*0.+1.
lowF[rho[0]<60.] = 0.
rho[1] *= lowF
rhoAlt = rho[1].copy()

clBBAD545 = numpy.dot((mat),(1.-numpy.interp(lmatA,rho[0],lowF*rho[1]**2.))*phiBin/TCMB**2.)

intEr =lmatA.copy()
intEr2 =lmatA.copy()

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

#sys.exit()
for i in xrange(10):
    errReals = numpy.random.randn(len(errorRatio[0]))*errorRatio[1]
    errReals2 = numpy.random.randn(len(errorRatio[0]))*errorRatio[1]

    for j in xrange(len(lmatA)):
        intEr[j] = find_nearest(lmatA[j],errorRatio[0],errReals)
        intEr2[j] = find_nearest(lmatA[j],errorRatio[0],errReals2)

#        print intEr
    intEr[lmatA>1500.] = 0.
    clBBADRand = numpy.dot(mat,(1.-numpy.interp(lmatA,rho[0],rhoAlt**2.)*(1.+multiCorrection*intEr*2.+multiCorrectionKg*intEr2*2.))*phiBin/TCMB**2.)
    pylab.plot(lmatA,(clBBADRand-clBBAD545),color='b')

pylab.plot(lmatA,(clBBADRand-clBBAD545),label=r'effect of $\sigma(C_l^{\kappa I})$',color='b')
pylab.plot(lmatA,(clBBADRand)*0.)

pylab.plot(lr,clBBr*0.001,label=r'$C_l^{BB,\mathrm{primordial}}(r=0.001)$',linewidth=2,color='k')
pylab.legend(loc='best')
print clBBA.shape
print lmatA.shape
pylab.ylim(-1e-20,3.e-20)
pylab.xlim(20,450)
pylab.ylabel(r'$C_l^{BB,\mathrm{res}}-C_l^{BB,\mathrm{res},\mathrm{fiducial}}$',fontsize=14.)
pylab.xlabel(r'$l$',fontsize=14.)
pylab.savefig('lensBcompareErrorFINALPostHackAndGal.png',bbox_inches='tight')
pylab.savefig('lensBcompareErrorFINALPostHackAndGal.eps',bbox_inches='tight')

pylab.clf()
print lmat.shape, intEr.shape
pylab.plot(lmatA,intEr)
pylab.plot(rho[0],rho[1])
pylab.xlim(50,3000.)
pylab.savefig('rhoAndErr.png')
