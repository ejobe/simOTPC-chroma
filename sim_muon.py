from chroma import make, view
import numpy as np
from geometry.detector import (build_detector, get_tube_height, get_tube_radius,
                               build_tube)

if __name__ == '__main__':
    from chroma.sim import Simulation
    from chroma.sample import uniform_sphere
    from chroma.event import Photons
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    import matplotlib.pyplot as plt

    plot_last_event = False

    g = build_detector()
    g.flatten()
    g.bvh = load_bvh(g)
    
    sim = Simulation(g, geant4_processes=1)
    
    from chroma.io.root import RootWriter
    f = RootWriter('muon.root')

    n = 0
    m = 0
    n_prime = 0.0
    m_prime = 0.0
    
    gun = vertex.particle_gun(['mu-'],
                              vertex.constant((n,m, get_tube_height())), #
                              vertex.constant((n_prime, m_prime, -1)),  #
                              vertex.constant(1000))      #     
            

    for ev in sim.simulate(gun, keep_photons_beg=True,keep_photons_end=True,
                           run_daq=False,max_steps=1000):
        f.write_event(ev)                      

        detected = (ev.photons_end.flags & (0x1 << 2)).astype(bool)        

        print "photons, generated: ", len(ev.photons_beg.t), " ..detected: ", len(ev.photons_end.t[detected])

    f.close()

    if(plot_last_event):
        fig1 = plt.figure(1)
        plt.plot(ev.photons_end.t[detected], ev.photons_end.pos[:,2][detected],'p')
        plt.xlabel('time [ns]')
        plt.ylabel('z [mm]')
          
        fig2 = plt.figure(2)
        plt.subplot(311)
        plt.plot(ev.photons_beg.t,ev.photons_beg.pos[:,0],"p")
        plt.subplot(312)
        plt.plot(ev.photons_beg.t,ev.photons_beg.pos[:,1],"p")
        plt.subplot(313)
        plt.plot(ev.photons_beg.t,ev.photons_beg.pos[:,2],"p")
    
        print "press enter to exit"
        plt.show(block=False)
        if raw_input():
            plt.close('all')
