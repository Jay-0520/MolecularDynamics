import mdtraj as md
from numpy import linalg as LA
import numpy as np
import math
from eigen_tiltrot_argu_final import *
from scipy.spatial import distance

def get_min_distan_bet_hlx_ha(frame,c1,c2,d1,d2):
    ### get x,y,z coordinates of ca_atoms
    ### define starting and ending index of two helices, "c2" or "b2" must be the index of the last residue of helix, "c1" or "b1" must be the index of the first residue of helix 
    ###x_axis,y_axis,z_axis are xyz coordinates of c_alpha atoms between selected index 

    ca_atoms = frame.topology.select("name CA")
    x_axis = frame.xyz[0,ca_atoms,0]
    y_axis = frame.xyz[0,ca_atoms,1]
    z_axis = frame.xyz[0,ca_atoms,2]

    S=np.arange(0,1,0.005)
   
    hpa_hlx_c=eigen(c1,c2-2,x_axis,y_axis,z_axis)[2] 
    ca_num_c1_z=frame.xyz[0,ca_atoms,2][c1]
    ca_num_c2_z=frame.xyz[0,ca_atoms,2][c2-2]
    if ca_num_c1_z > ca_num_c2_z: ##original: small index is always on the top
        if vangle(hpa_hlx_c,[0,0,1]) > 90:
            norm_hpa_hlx_c=-hpa_hlx_c/ vlen(hpa_hlx_c)
        else:
            norm_hpa_hlx_c=hpa_hlx_c/ vlen(hpa_hlx_c)
    else:
        if vangle(hpa_hlx_c,[0,0,1]) < 90:
            norm_hpa_hlx_c=-hpa_hlx_c / vlen(hpa_hlx_c)
        else:
            norm_hpa_hlx_c=hpa_hlx_c / vlen(hpa_hlx_c)

    cent_ca_c=frame.xyz[0,ca_atoms[c1:c2]].mean(axis=0)
    r_1_c=frame.xyz[0,ca_atoms[c1]]
    r_n_c=frame.xyz[0,ca_atoms[c2]]
    # Equation (2-3) from Lee, Im JCC 2006
    begin_c = cent_ca_c+np.dot(norm_hpa_hlx_c,(r_1_c-cent_ca_c))*norm_hpa_hlx_c
    end_c   = cent_ca_c+np.dot(norm_hpa_hlx_c,(r_n_c-cent_ca_c))*norm_hpa_hlx_c
    arbi_point_c = np.asarray([begin_c + ss*(end_c-begin_c) for ss in S])
    
    hpa_hlx_d=eigen(d1,d2-2,x_axis,y_axis,z_axis)[2]
    ca_num_d1_z=frame.xyz[0,ca_atoms,2][d1]
    ca_num_d2_z=frame.xyz[0,ca_atoms,2][d2-2]
    if ca_num_d1_z > ca_num_d2_z: ##original: small index is always on the top
        if vangle(hpa_hlx_d,[0,0,1]) > 90:
            norm_hpa_hlx_d=-hpa_hlx_d/ vlen(hpa_hlx_d)
        else:
            norm_hpa_hlx_d=hpa_hlx_d/ vlen(hpa_hlx_d)
    else:
        if vangle(hpa_hlx_d,[0,0,1]) < 90:
            norm_hpa_hlx_d=-hpa_hlx_d / vlen(hpa_hlx_d)
        else:
            norm_hpa_hlx_d=hpa_hlx_d / vlen(hpa_hlx_d)

    cent_ca_d=frame.xyz[0,ca_atoms[d1:d2]].mean(axis=0)
    r_1_d=frame.xyz[0,ca_atoms[d1]]
    r_n_d=frame.xyz[0,ca_atoms[d2]]
    # Equation (2-3) from Lee, Im JCC 2006
    begin_d = cent_ca_d+np.dot(norm_hpa_hlx_d,(r_1_d-cent_ca_d))*norm_hpa_hlx_d
    end_d   = cent_ca_d+np.dot(norm_hpa_hlx_d,(r_n_d-cent_ca_d))*norm_hpa_hlx_d
    arbi_point_d = np.asarray([begin_d + ss*(end_d-begin_d) for ss in S])
   
    return distance.cdist(arbi_point_c, arbi_point_d, 'euclidean').min() 
