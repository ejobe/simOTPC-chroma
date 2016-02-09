import numpy as np
from chroma import make, loader, geometry, transform, view, mesh_from_stl
from chroma.demo.optics import (water, vacuum, r7081hqe_photocathode,
                                black_surface, shiny_surface, glass,
                                glossy_surface, lambertian_surface)
from custom_optics import mcp_boro_photocathode, badwater#, glass
from chroma.tools import profile_if_possible
from normal import get_normals, make_axes

# detector dimensions (mm)
mcp_height = 25.4
mcp_length = 50.8#100.5
mirror_length = 1.5*mcp_length
mcp_glass_thickness = 2.0

window_thickness = 3.2

plug_offset = 6

mirror_thickness = 5.84

tube_height = 771.53 #30 inches tall
tube_inner_radius = 250.8/2
tube_thickness = 30
tube_outer_radius = tube_inner_radius + tube_thickness
water_level_from_top = 0

start_of_detectors_from_top = 220

mcp_plane = tube_inner_radius + mcp_height/2

mcp_0_displace = start_of_detectors_from_top
mcp_1_displace = mcp_0_displace - 3.5*25.4 #2.5
mcp_1_abs_displace = start_of_detectors_from_top + 3.5*25
mcp_2_displace = mcp_0_displace - 7*25.4#7 
mcp_2_abs_displace = start_of_detectors_from_top + 7*25.4 
mcp_3_displace = mcp_0_displace - 10.5*25.4#9.5
mcp_3_abs_displace = start_of_detectors_from_top + 10.5*25.4
mcp_4_displace = mcp_0_displace - 14.5*25.4 #2.5
mcp_4_abs_displace = start_of_detectors_from_top + 14.5*25
mcp_5_displace = mcp_0_displace - 17.5*25.4#7 
mcp_5_abs_displace = start_of_detectors_from_top + 18*25.4 


mirror_plane = tube_inner_radius

#new: theta actually 32.2 degrees
#OLD theta 30 degrees
th_degrees = 32.6
th_radians = np.pi * th_degrees/180
cos_th = np.cos(th_radians)  #0.8434#0.866
sin_th = np.sin(th_radians)  #0.5373#0.5
cos_2th = np.cos(2*th_radians) #0.4320 #0.5
sin_2th = np.sin(2*th_radians) #0.90183 #0.866

sin_th_ck = 0.40673664307
cos_th_ck = 0.91354545764

rotate_matrix_x = np.matrix([[1,0,0],[0,0,-1],[0,1,0]])
rotate_matrix_y0 = np.matrix([[cos_th,0,sin_th],[0,1,0],[-sin_th,0,cos_th]])
rotate_matrix_y1 = np.matrix([[cos_th,0,-sin_th],[0,1,0],[sin_th,0,cos_th]])
rotate_matrix_z0 = np.matrix([[cos_2th,-sin_2th,0],[sin_2th,cos_2th,0],[0,0,1]])
identity_matrix = np.matrix([[1,0,0],[0,1,0],[0,0,1]])

#rotate_matrix_0 = rotate_matrix_x * rotate_matrix_y0 
rotate_matrix_0 = rotate_matrix_x * rotate_matrix_y0 * rotate_matrix_y0 
#rotate_matrix_1 = rotate_matrix_x * rotate_matrix_y1
rotate_matrix_1 = rotate_matrix_x 

rotate_matrix_invX = np.matrix([[-1,0,0],[0,1,0],[0,0,1]])
rotate_matrix_invZ = np.matrix([[1,0,0],[0,1,0],[0,0,-1]])
rotate_matrix_invY = np.matrix([[1,0,0],[0,-1,0],[0,0,1]])

rotate_cherenkov = np.matrix([[1,0,0],[0,cos_th_ck,sin_th_ck],[0, -sin_th_ck, cos_th_ck]])

rotate_mirror_matrix = rotate_matrix_x * rotate_cherenkov
rotate_mirror_0 = rotate_matrix_x * rotate_matrix_y0

rotate_tube = rotate_matrix_x * rotate_matrix_invY * rotate_matrix_invZ

# mcp resolution (mm, ps)
mcp_x_resolution = 10
mcp_y_resolution = 10
mcp_t_resolution = 1000/1e3 # 1 ns, set to approximate PMT

reflect0 = geometry.Surface('reflect0')
reflect0.set('reflect_specular',0.05)
reflect0.set('absorb',0.95)

reflect90 = geometry.Surface('reflect90')
reflect90.set('reflect_specular', 0.9)
reflect90.set('absorb', 0.1)

window_thickness = 3.175  ##0.125" window

def get_normals(mesh):
    triangles = mesh.assemble()

    v1 = triangles[:,1] - triangles[:,0]
    v2 = triangles[:,2] - triangles[:,1]

    return normalize(np.cross(v1,v2))

def build_mcp(length):
    """Returns a simple photodetector Solid. The photodetector is a cube of
    size `size` constructed out of a glass envelope with a photosensitive
    face on the inside of the glass envelope facing up."""
    glass_thickness = 1
 
    # outside of the glass envelope
    outside_mesh = make.box(length+3, length+3, mcp_height)
    # inside of the glass envelope
    inside_mesh = make.box(length-glass_thickness,length-glass_thickness,mcp_height-0*window_thickness-glass_thickness)

    # outside solid with water on the outside, and glass on the inside
    outside_solid = geometry.Solid(outside_mesh,glass,badwater)    

    # now we need to determine the triangles which make up
    # the top face of the inside mesh, because we are going to place
    # the photosensitive surface on these triangles
    # do this by seeing which triangle centers are at the maximum z
    # coordinate
    z = inside_mesh.get_triangle_centers()[:,2]
    top = z == -max(z)

    # see np.where() documentation
    # Here we make the photosensitive surface along the top face of the inside
    # mesh. The rest of the inside mesh is perfectly absorbing.
    inside_surface = np.where(top,mcp_boro_photocathode,black_surface)
    inside_color = np.where(top,0x00ff00,0x33ffffff)

    # construct the inside solid
    inside_solid = geometry.Solid(inside_mesh,vacuum,glass,surface=inside_surface,
                         color=inside_color)

    # you can add solids and meshes!
    return outside_solid + inside_solid


def segment_path(xy, dx):
    xy = np.array(xy)

    xy_segmented = []
    for xy1, xy2 in zip(xy[:-1],xy[1:]):
        N = transform.norm(xy2-xy1)//dx

        if N < 1:
            N = 1

        xy_segmented += (xy1 + np.linspace(0,1,N,endpoint=False)[:,np.newaxis]*(xy2-xy1)).tolist()

    xy_segmented += [list(xy[-1])]

    return xy_segmented

def get_tube_height():
    return tube_height

def get_tube_radius():
    return tube_inner_radius

def build_tube():   
    tube_mesh = mesh_from_stl('/net/users/eric/optTPC_sim/detector/geometry/stl_files/OTPC_Tank_mm.stl')
    inside_solid = geometry.Solid(tube_mesh,glass,badwater,surface=black_surface, color=0x666666)   
    #inside_solid = geometry.Solid(inside_mesh,water,vacuum)   #make shiny

    return inside_solid

def build_plug():
    plug_mesh = mesh_from_stl('/net/users/eric/optTPC_sim/detector/geometry/stl_files/OTPC_Plug_mm.stl')
    inside_solid = geometry.Solid(plug_mesh, glass,badwater, surface=black_surface, color = 0xffffff)

    return inside_solid

def build_glass_window():
    window_mesh = make.cylinder(50.8, window_thickness, radius2=None, nsteps=64)
    window_solid = geometry.Solid(window_mesh, glass, badwater, color=0xd1efff)

    return window_solid


def build_mirror(mirror_x, mirror_y):
    glass_thickness = .1

    outside_mesh = make.box(mirror_x, mirror_y, glass_thickness)
    inside_mesh = make.box(mirror_x - glass_thickness, mirror_y-glass_thickness, glass_thickness/2)
  
    outside_solid = geometry.Solid(outside_mesh,glass,badwater)  

    z = inside_mesh.get_triangle_centers()[:,2]
    top = z == max(z)
    
    inside_surface = np.where(top, reflect90, None)
    inside_color = np.where(top,0xff3232,0)
    
    inside_solid = geometry.Solid(inside_mesh,vacuum,glass,surface=inside_surface,
                         color=inside_color)

    return outside_solid + inside_solid

def build_detector():
    g = geometry.Geometry(badwater)

    mcp_solid_0 = build_mcp(mcp_length)   
    mcp_solid_0.mesh.vertices[:,2] += mcp_plane + plug_offset + window_thickness
    mcp_solid_0.mesh.vertices[:,1] += - mcp_0_displace + tube_height
   
    mcp_solid_1 = build_mcp(mcp_length)
    mcp_solid_1.mesh.vertices[:,2] += mcp_plane + plug_offset + window_thickness
    mcp_solid_1.mesh.vertices[:,1] += -mcp_1_abs_displace + tube_height
   
    mcp_solid_2 = build_mcp(mcp_length)
    mcp_solid_2.mesh.vertices[:,2] += mcp_plane + plug_offset + window_thickness
    mcp_solid_2.mesh.vertices[:,1] += -mcp_2_abs_displace + tube_height

    mcp_solid_3 = build_mcp(mcp_length)
    mcp_solid_3.mesh.vertices[:,2] += mcp_plane + plug_offset + window_thickness 
    mcp_solid_3.mesh.vertices[:,1] += -mcp_3_abs_displace + tube_height
    
    mcp_solid_4 = build_mcp(mcp_length)
    mcp_solid_4.mesh.vertices[:,2] += mcp_plane + plug_offset + window_thickness 
    mcp_solid_4.mesh.vertices[:,1] += -mcp_4_abs_displace + tube_height

    mcp_solid_5 = build_mcp(mcp_length)
    mcp_solid_5.mesh.vertices[:,2] += mcp_plane + plug_offset + window_thickness
    mcp_solid_5.mesh.vertices[:,1] += -mcp_5_abs_displace + tube_height

    #mirror_solid_1 = build_mirror(2*tube_inner_radius*sin_th, 1*mcp_length)
    mirror_solid_0 = build_mirror(mirror_length, mirror_length)

    mirror_solid_1 = build_mirror(mirror_length, mirror_length)
    #mirror_solid_1.mesh.vertices[:,2] += -tube_inner_radius - mcp_length*sin_th_ck
    #mirror_solid_1.mesh.vertices[:,1] += -mcp_1_abs_displace  + mcp_length/2 * sin_th_ck + tube_height
    
    mirror_solid_2 = build_mirror(mirror_length, mirror_length)
    mirror_solid_3 = build_mirror(mirror_length, mirror_length)
    mirror_solid_4 = build_mirror(mirror_length, mirror_length)
    mirror_solid_5 = build_mirror(mirror_length, mirror_length)
    ######

    tube_solid = build_tube()
    tube_solid.mesh.vertices[:,1] -= tube_height#/2
    #print tube_solid.mesh.vertices
 
    plug_solid_0 = build_plug()
    plug_solid_0.mesh.vertices[:,1] += -mcp_plane - plug_offset 
    plug_solid_0.mesh.vertices[:,2] += - mcp_0_displace + tube_height
    plug_solid_1 = build_plug()
    plug_solid_1.mesh.vertices[:,1] += -mcp_plane - plug_offset
    plug_solid_1.mesh.vertices[:,2] += - mcp_1_abs_displace + tube_height
    plug_solid_2 = build_plug()
    plug_solid_2.mesh.vertices[:,1] += -mcp_plane - plug_offset
    plug_solid_2.mesh.vertices[:,2] += - mcp_2_abs_displace + tube_height
    plug_solid_3 = build_plug()
    plug_solid_3.mesh.vertices[:,1] += -mcp_plane - plug_offset
    plug_solid_3.mesh.vertices[:,2] += - mcp_3_abs_displace + tube_height
    plug_solid_4 = build_plug()
    plug_solid_4.mesh.vertices[:,1] += -mcp_plane - plug_offset
    plug_solid_4.mesh.vertices[:,2] += - mcp_4_abs_displace + tube_height
    plug_solid_5 = build_plug()
    plug_solid_5.mesh.vertices[:,1] += -mcp_plane - plug_offset
    plug_solid_5.mesh.vertices[:,2] += - mcp_5_abs_displace + tube_height

    window_0 = build_glass_window()
    window_0.mesh.vertices[:,1] += -mcp_plane + window_thickness
    window_0.mesh.vertices[:,2] += -mcp_0_displace + tube_height
    window_1 = build_glass_window()
    window_1.mesh.vertices[:,1] += -mcp_plane + window_thickness
    window_1.mesh.vertices[:,2] += -mcp_1_abs_displace + tube_height
    window_2 = build_glass_window()
    window_2.mesh.vertices[:,1] += -mcp_plane + window_thickness
    window_2.mesh.vertices[:,2] += -mcp_2_abs_displace + tube_height
    window_3 = build_glass_window()
    window_3.mesh.vertices[:,1] += -mcp_plane + window_thickness
    window_3.mesh.vertices[:,2] += -mcp_3_abs_displace + tube_height
    window_4 = build_glass_window()
    window_4.mesh.vertices[:,1] += -mcp_plane + window_thickness
    window_4.mesh.vertices[:,2] += -mcp_4_abs_displace + tube_height
    window_5 = build_glass_window()
    window_5.mesh.vertices[:,1] += -mcp_plane + window_thickness
    window_5.mesh.vertices[:,2] += -mcp_5_abs_displace + tube_height    

    g.add_solid(tube_solid, rotation=rotate_tube)
    
    g.add_solid(plug_solid_0, rotation=None)
    g.add_solid(plug_solid_1, rotation=rotate_matrix_z0)
    g.add_solid(plug_solid_2, rotation=None)
    g.add_solid(plug_solid_3, rotation=rotate_matrix_z0)
    g.add_solid(plug_solid_4, rotation=None)
    g.add_solid(plug_solid_5, rotation=rotate_matrix_z0)

    g.add_solid(window_0, rotation=None)
    g.add_solid(window_1, rotation=rotate_matrix_z0)
    g.add_solid(window_2, rotation=None)
    g.add_solid(window_3, rotation=rotate_matrix_z0)
    g.add_solid(window_4, rotation=None)
    g.add_solid(window_5, rotation=rotate_matrix_z0)

    g.add_solid(mcp_solid_0, rotation=rotate_matrix_1)
    g.add_solid(mcp_solid_1, rotation=rotate_matrix_0)
    g.add_solid(mcp_solid_2, rotation=rotate_matrix_1)
    g.add_solid(mcp_solid_3, rotation=rotate_matrix_0)
    g.add_solid(mcp_solid_4, rotation=rotate_matrix_1)
    #g.add_solid(mcp_solid_5, rotation=rotate_matrix_0)

    
    g.add_solid(mirror_solid_0, displacement = (0,tube_inner_radius - mirror_length * sin_th_ck, \
                                                     -mcp_0_displace + mcp_length/2 * sin_th_ck + tube_height ), \
                    rotation= rotate_matrix_1 * rotate_cherenkov)
    g.add_solid(mirror_solid_1, displacement = ((-tube_inner_radius+ mirror_length*sin_th_ck)*sin_2th, \
                                                    (tube_inner_radius - mirror_length*sin_th_ck)*cos_2th, \
                                                     -mcp_1_abs_displace + mcp_length/2 * sin_th_ck + tube_height), \
                    rotation= rotate_matrix_0 * rotate_cherenkov)

    g.add_solid(mirror_solid_2, displacement = (0,tube_inner_radius - mirror_length*sin_th_ck, \
                                                     -mcp_2_abs_displace + mcp_length/2 * sin_th_ck + tube_height), \
                    rotation= rotate_matrix_1 * rotate_cherenkov)
    g.add_solid(mirror_solid_3, displacement = ((-tube_inner_radius+ mirror_length*sin_th_ck)*sin_2th, \
                                                    (tube_inner_radius - mirror_length*sin_th_ck)*cos_2th, \
                                                     -mcp_3_abs_displace + mcp_length/2 * sin_th_ck + tube_height),\
                    rotation= rotate_matrix_0 * rotate_cherenkov)
    g.add_solid(mirror_solid_4, displacement = (0,tube_inner_radius - mirror_length*sin_th_ck, \
                                                    -mcp_4_abs_displace + mcp_length/2 * sin_th_ck + tube_height), \
                    rotation= rotate_matrix_1 * rotate_cherenkov)
    #g.add_solid(mirror_solid_5, displacement = ((-tube_inner_radius+ mirror_length*sin_th_ck)*sin_2th, \
    #                                                (tube_inner_radius - mirror_length*sin_th_ck)*cos_2th, \
    #                                                 -mcp_5_abs_displace + mcp_length/2 * sin_th_ck + tube_height),\
    #                rotation= rotate_matrix_0 * rotate_cherenkov)
    #g.flatten()
    #g.bvh = loader.load_bvh(g)

    return g


if __name__ == '__main__':
    from chroma import view

    view(build_detector())
