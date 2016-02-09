import os, os.path
import shutil
import numpy as np
import event as event

import ROOT

# Check if we have already imported the ROOT class due to a user's
# rootlogon.C script
if not hasattr(ROOT, 'Vertex') or not hasattr(ROOT, 'Channel'):
    # Create .chroma directory if it doesn't exist
    chroma_dir = os.path.expanduser('~/.chroma')
    if not os.path.isdir(chroma_dir):
        if os.path.exists(chroma_dir):
            raise Exception('$HOME/.chroma file exists where directory should be')
        else:
            os.mkdir(chroma_dir)
    # Check if latest ROOT file is present
    package_root_C = os.path.join(os.path.dirname(__file__), 'root.C')
    home_root_C = os.path.join(chroma_dir, 'root.C')
    if not os.path.exists(home_root_C) or \
            os.stat(package_root_C).st_mtime > os.stat(home_root_C).st_mtime:
        shutil.copy2(src=package_root_C, dst=home_root_C)
    # Import this C file for access to data structure
    ROOT.gROOT.ProcessLine('.L '+home_root_C+'+')


def tvector3_to_ndarray(vec):
    '''Convert a ROOT.TVector3 into a numpy np.float32 array'''
    return np.array((vec.X(), vec.Y(), vec.Z()), dtype=np.float32)

def make_photon_with_arrays(size):
    '''Returns a new chroma.event.Photons object for `size` number of
    photons with empty arrays set for all the photon attributes.'''
    return event.Photons(pos=np.empty((size,3), dtype=np.float32),
                         dir=np.empty((size,3), dtype=np.float32),
                         pol=np.empty((size,3), dtype=np.float32),
                         wavelengths=np.empty(size, dtype=np.float32),
                         t=np.empty(size, dtype=np.float32),
                         flags=np.empty(size, dtype=np.uint32),
                         last_hit_triangles=np.empty(size, dtype=np.int32))

def root_vertex_to_python_vertex(vertex):
    "Returns a chroma.event.Vertex object from a root Vertex object."
    return event.Vertex(str(vertex.particle_name),
                        pos=tvector3_to_ndarray(vertex.pos),
                        dir=tvector3_to_ndarray(vertex.dir),
                        ke=vertex.ke,
                        t0=vertex.t0,
                        pol=tvector3_to_ndarray(vertex.pol))

def root_event_to_python_event(ev):
    '''Returns a new chroma.event.Event object created from the
    contents of the ROOT event `ev`.'''
    pyev = event.Event(ev.id)
    pyev.primary_vertex = root_vertex_to_python_vertex(ev.primary_vertex)

    for vertex in ev.vertices:
        pyev.vertices.append(root_vertex_to_python_vertex(vertex))

    # photon begin
    if ev.photons_beg.size() > 0:
        photons = make_photon_with_arrays(ev.photons_beg.size())
        ROOT.get_photons(ev.photons_beg,
                         photons.pos.ravel(),
                         photons.dir.ravel(),
                         photons.pol.ravel(),
                         photons.wavelengths,
                         photons.t,
                         photons.last_hit_triangles,
                         photons.flags)
        pyev.photons_beg = photons

    # photon end
    if ev.photons_end.size() > 0:
        photons = make_photon_with_arrays(ev.photons_end.size())
        ROOT.get_photons(ev.photons_end,
                         photons.pos.ravel(),
                         photons.dir.ravel(),
                         photons.pol.ravel(),
                         photons.wavelengths,
                         photons.t,
                         photons.last_hit_triangles,
                         photons.flags)
        pyev.photons_end = photons

    # channels
    if ev.nchannels > 0:
        hit = np.empty(ev.nchannels, dtype=np.int32)
        t = np.empty(ev.nchannels, dtype=np.float32)
        q = np.empty(ev.nchannels, dtype=np.float32)
        flags = np.empty(ev.nchannels, dtype=np.uint32)

        ROOT.get_channels(ev, hit, t, q, flags)
        pyev.channels = event.Channels(hit.astype(bool), t, q, flags)
    else:
        pyev.channels = None

    return pyev
    
class RootReader(object):
    '''Reader of Chroma events from a ROOT file.  This class can be used to 
    navigate up and down the file linearly or in a random access fashion.
    All returned events are instances of the chroma.event.Event class.

    It implements the iterator protocol, so you can do

       for ev in RootReader('electron.root'):
           # process event here
    '''

    def __init__(self, filename):
        '''Open ROOT file named `filename` containing TTree `T`.'''
        self.f = ROOT.TFile(filename)
        print "opening root file ", filename 
        self.T = self.f.T
        self.i = -1
        
    def __len__(self):
        '''Returns number of events in this file.'''
        print "num of entries: ", T.GetEntries()
        return self.T.GetEntries()

    def __iter__(self):
        for i in xrange(self.T.GetEntries()):
            self.T.GetEntry(i)
            yield root_event_to_python_event(self.T.ev)

    def next(self):
        '''Return the next event in the file. Raises StopIteration
        when you get to the end.'''
        if self.i + 1 >= len(self):
            raise StopIteration

        self.i += 1
        self.T.GetEntry(self.i)
        return root_event_to_python_event(self.T.ev)

    def prev(self):
        '''Return the next event in the file. Raises StopIteration if
        that would go past the beginning.'''
        if self.i <= 0:
            self.i = -1
            raise StopIteration

        self.i -= 1
        self.T.GetEntry(self.i)
        return root_event_to_python_event(self.T.ev)

    def current(self):
        '''Return the current event in the file.'''
        self.T.GetEntry(self.i) # in case we were iterated over elsewhere
        return root_event_to_python_event(self.T.ev)

    def jump_to(self, index):
        '''Return the event at `index`.  Updates current location.'''
        if index < 0 or index >= len(self):
            raise IndexError
        
        self.i = index

        self.T.GetEntry(self.i)
        return root_event_to_python_event(self.T.ev)

    def index(self):
        '''Return the current event index'''
        return self.i
