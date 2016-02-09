import numpy as np
from chroma.geometry import Material, Surface
from chroma.demo.optics import vacuum, r7081hqe_photocathode
import math

glass = Material('glass')
glass.set('refractive_index', 1.49)
glass.absorption_length = \
    np.array([(200, 0.1e-6), (300, 1000), (330, 1000.0), (500, 2000.0), (600, 1000.0), (770, 500.0), (800, 0.1e-6)])
glass.set('scattering_length', 1e6)

silica =   np.array([(180, .9), (200, .93), (220,.94), (240, .95), (260, .95), (280, .95), (800,.95)])
gel    =   np.array([(180, 0.0), (260, 0.0),(280, .09),(300.,.4), (320, .83), (365,.98), (404.7,0.99), (480,1.0), (800,1.0)])
mirror =   np.array([(180, 0), (220, 0), (240,.0), (260, .1), (280, .4), (300, .7), (340,.88), (360, .95), (400, .97), (550, .97), (600, .95), (700, .92), (800, .87)])

mcp_boro_photocathode = Surface('mcp_boro_photocathode')
mcp_silica_photocathode = Surface('mcp_silica_photocathode')

'''
mcp_boro_photocathode.detect = \
    np.array([(240.0,  0.00), (250.0,  0.04), (260.0,  2), 
              (270.0,  8.77), (280.0,  24), (290.0,  26), (300.0,  27),
              (310.0, 28.00), (320.0, 28.8), (330.0, 29.7), (340.0, 30.1),
              (350.0, 30.52), (360.0, 31.0), (370.0, 31.30), (380.0, 31.20),
              (390.0, 31.00), (400.0, 30.90), (410.0, 30.50), (420.0, 30.16),
              (430.0, 29.24), (440.0, 28.31), (450.0, 27.41), (460.0, 26.25),
              (470.0, 24.90), (480.0, 23.05), (490.0, 21.58), (500.0, 19.94),
              (510.0, 18.48), (520.0, 17.01), (530.0, 15.34), (540.0, 12.93),
              (550.0, 10.17), (560.0,  7.86), (570.0,  6.23), (580.0,  5.07),
              (590.0,  4.03), (600.0,  3.18), (610.0,  2.38), (620.0,  1.72),
              (630.0,  0.95), (640.0,  0.71), (650.0,  0.44), (660.0,  0.25),
              (670.0,  0.14), (680.0,  0.07), (690.0,  0.03), (700.0,  0.02),
              (710.0,  0.00)])
'''

attnlen= np.load('geometry/attnlen1.npy')
waves= np.load('geometry/waves.npy')

waves = np.concatenate([np.array([200,210,220,230,240,250]),waves])
attnlen = np.concatenate([np.array([.01,.02,.04,.07,.1,.2]),attnlen])

mcp_boro_photocathode.detect = \
    np.array([(180, 0.00), (200, 2), (220,5.5),
              (230, 8.4), (240.0,  11), (250.0,  13.5), (260.0,  15), 
              (270.0,  17), (280.0,  17.5), (290.0,  18.2), (300.0,  19),
              (310.0, 19.4), (320.0, 20), (330.0, 20.3), (340.0, 20.7),
              (350.0, 21.0), (360.0, 21.2), (370.0, 21.5), (380.0, 21.6),
              (390.0, 21.5), (400.0, 21.0), (410.0, 20.5), (420.0, 19.8),
              (430.0, 18.0), (440.0, 17.0), (450.0, 16.0), (460.0, 15.0),
              (470.0, 13.0), (480.0, 12.5), (490.0, 11.7), (500.0, 11.0),
              (510.0, 9.8), (520.0, 8), (530.0, 7), (540.0, 4.3),
              (550.0, 3.36), (560.0,  2.55), (570.0,  2.2), (580.0,  1.9),
              (590.0,  1.5), (600.0,  .8), (610.0,  0.0), (620.0,  0.00),
              (630.0,  0.00), (640.0,  0.00), (650.0,  0.00), (660.0,  0.00),
              (670.0,  0.00), (680.0,  0.00), (690.0,  0.00), (700.0,  0.00),
              (710.0,  0.00)])


mcp_boro_photocathode.detect[:,1] = mcp_boro_photocathode.detect[:,1] * 0.75
#mcp_boro_photocathode.detect[:,1] = hqe_photocathode.detect[:,1] * 0.6
# convert percent -> fraction
#mcp_boro_photocathode.detect[:,1] /= 2
mcp_boro_photocathode.detect[:,1] /= 100.0
# roughly the same amount of detected photons are absorbed without detection
mcp_boro_photocathode.absorb = mcp_boro_photocathode.detect
# remaining photons are diffusely reflected
mcp_boro_photocathode.set('reflect_diffuse', 1.0 - mcp_boro_photocathode.detect[:,1] - mcp_boro_photocathode.absorb[:,1], wavelengths=mcp_boro_photocathode.detect[:,0])

mcp_silica_photocathode.detect = \
    np.array([(180, 18.6), (200, 17.4), (220,16.6),
              (230, 15.9), (240.0, 15.6 ), (250.0,  15.7), (260.0,  16.3), 
              (270.0,  17.0), (280.0,  17.5), (290.0,  18.2), (300.0,  19),
              (310.0, 19.4), (320.0, 20), (330.0, 20.3), (340.0, 20.7),
              (350.0, 21.0), (360.0, 21.2), (370.0, 21.5), (380.0, 21.6),
              (390.0, 21.5), (400.0, 21.0), (410.0, 20.5), (420.0, 19.8),
              (430.0, 18.0), (440.0, 17.0), (450.0, 16.0), (460.0, 15.0),
              (470.0, 13.0), (480.0, 12.5), (490.0, 11.7), (500.0, 11.0),
              (510.0, 9.8), (520.0, 8), (530.0, 7), (540.0, 4.3),
              (550.0, 3.36), (560.0,  2.55), (570.0,  2.2), (580.0,  1.9),
              (590.0,  1.5), (600.0,  .8), (610.0,  0.0), (620.0,  0.00),
              (630.0,  0.00), (640.0,  0.00), (650.0,  0.00), (660.0,  0.00),
              (670.0,  0.00), (680.0,  0.00), (690.0,  0.00), (700.0,  0.00),
              (710.0,  0.00)])

mcp_silica_photocathode.detect[:,1] /= 100.0

badwater = Material('badwater')
badwater.density = 1.0 # g/cm^3
badwater.composition = { 'H' : 0.1119, 'O' : 0.8881 } # fraction by mass
hc_over_GeV = 1.2398424468024265e-06 # h_Planck * c_light / GeV / nanometer
wcsim_wavelengths = hc_over_GeV /np.array([ 1.56962e-09, 1.58974e-09, 1.61039e-09, 1.63157e-09, 
       1.65333e-09, 1.67567e-09, 1.69863e-09, 1.72222e-09, 
       1.74647e-09, 1.77142e-09, 1.7971e-09, 1.82352e-09, 
       1.85074e-09, 1.87878e-09, 1.90769e-09, 1.93749e-09, 
       1.96825e-09, 1.99999e-09, 2.03278e-09, 2.06666e-09,
       2.10169e-09, 2.13793e-09, 2.17543e-09, 2.21428e-09, 
       2.25454e-09, 2.29629e-09, 2.33962e-09, 2.38461e-09, 
       2.43137e-09, 2.47999e-09, 2.53061e-09, 2.58333e-09, 
       2.63829e-09, 2.69565e-09, 2.75555e-09, 2.81817e-09, 
       2.88371e-09, 2.95237e-09, 3.02438e-09, 3.09999e-09,
       3.17948e-09, 3.26315e-09, 3.35134e-09, 3.44444e-09, 
       3.54285e-09, 3.64705e-09, 3.75757e-09, 3.87499e-09, 
       3.99999e-09, 4.13332e-09, 4.27585e-09, 4.42856e-09, 
       4.59258e-09, 4.76922e-09, 4.95999e-09, 5.16665e-09, 
       5.39129e-09, 5.63635e-09, 5.90475e-09, 6.19998e-09 ])[::-1] #reversed

badwater.set('refractive_index', 
                wavelengths=wcsim_wavelengths,
                value=np.array([1.32885, 1.32906, 1.32927, 1.32948, 1.3297, 1.32992, 1.33014, 
                          1.33037, 1.3306, 1.33084, 1.33109, 1.33134, 1.3316, 1.33186, 1.33213,
                          1.33241, 1.3327, 1.33299, 1.33329, 1.33361, 1.33393, 1.33427, 1.33462,
                          1.33498, 1.33536, 1.33576, 1.33617, 1.3366, 1.33705, 1.33753, 1.33803,
                          1.33855, 1.33911, 1.3397, 1.34033, 1.341, 1.34172, 1.34248, 1.34331,
                          1.34419, 1.34515, 1.3462, 1.34733, 1.34858, 1.34994, 1.35145, 1.35312,
                          1.35498, 1.35707, 1.35943, 1.36211, 1.36518, 1.36872, 1.37287, 1.37776,
                          1.38362, 1.39074, 1.39956, 1.41075, 1.42535])[::-1] #reversed
)
badwater.set('absorption_length',
                wavelengths=waves,
                value=attnlen * 1000
                )
      
badwater.set('scattering_length',
                wavelengths=wcsim_wavelengths,
                value=np.array([167024.4, 158726.7, 150742,
                          143062.5, 135680.2, 128587.4,
                          121776.3, 115239.5, 108969.5,
                          102958.8, 97200.35, 91686.86,
                          86411.33, 81366.79, 76546.42,
                          71943.46, 67551.29, 63363.36,
                          59373.25, 55574.61, 51961.24,
                          48527.00, 45265.87, 42171.94,
                          39239.39, 36462.50, 33835.68,
                          31353.41, 29010.30, 26801.03,
                          24720.42, 22763.36, 20924.88,
                          19200.07, 17584.16, 16072.45,
                          14660.38, 13343.46, 12117.33,
                          10977.70, 9920.416, 8941.407,
                          8036.711, 7202.470, 6434.927,
                          5730.429, 5085.425, 4496.467,
                          3960.210, 3473.413, 3032.937,
                          2635.746, 2278.907, 1959.588,
                          1675.064, 1422.710, 1200.004,
                          1004.528, 833.9666, 686.1063])[::-1] * 10.0 * 0.625 # reversed, cm -> mm, * magic tuning constant
          )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import rc

    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('font', size=13)


    plt.figure(1)
    plt.plot(mcp_boro_photocathode.detect[:,0], mcp_boro_photocathode.detect[:,1]/.75, color='red', linewidth=3.5, label='QE (typ.)+borosilicate window')
    plt.plot(mcp_silica_photocathode.detect[:,0], mcp_silica_photocathode.detect[:,1], color='blue', linewidth=3.5, label='QE (typ.)+fused-silica window', ls='--')
    plt.plot(silica[:,0], silica[:,1],color='green', linewidth=3, label='Transmission, OTPC port',ls='-.')
    plt.plot(gel[:,0], gel[:,1],':k',linewidth=4.5,label='Transmission, 1 mm opt. gel')
    plt.plot(mirror[:,0], mirror[:,1],linestyle='--', color='orange',linewidth=4.5,label='Reflectance, mirror')

    plt.xlim([170,700])
    plt.ylim([-.05, 1.1])
    #plt.yscale('log', nonposy='clip')
    plt.legend( loc=(.35,.4), numpoints = 1)
    #plt.legend( loc=1, numpoints = 1)
    plt.xlabel('Wavelength [nm]', fontsize=17)
    plt.ylabel('Spectral Efficiency', fontsize=17)
    

    plt.figure(2)
    #plt.plot(waves, badwater.absorption_length[:,1])

    n1 = 1.333
    n2 = 1.458
    
    x = np.arange(0,np.pi/2, .1)
    R_p_hi = n1*np.sqrt(1-np.power((n1*np.sin(x)/n2),2))-n2*np.cos(x)
    R_p_lo = n1*np.sqrt(1-np.power((n1*np.sin(x)/n2),2))+n2*np.cos(x)

    plt.plot(x, np.power(np.absolute(R_p_hi/R_p_lo), 2))
    plt.ylim([-.05, .05])

    convolve = []
    print len(gel), len(mcp_silica_photocathode.detect)
    for i in range(0,len(silica)):
        convolve.append(silica[i,1]*gel[i,1]*mcp_silica_photocathode.detect[i,1])

    plt.figure(3)
    plt.plot(convolve)

    plt.show()
    
