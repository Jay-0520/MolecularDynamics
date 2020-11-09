# ---------------------------------------------------------------------------
# This script processes gromacs trajectory of molecular dynamics (MD)
# simulations
#
# Specifically, this script centers the cylinder of water/the pore formed 
# during simulations of membrane systems. The strategy is: 1. compute
# the center of mass (COM) of the membrane (e.g., the octane slab in this 
# script); 2. define a 2D x-y plane (or a very thin rectangular), with z set
#  at the z value of the COM of the membrane; 3. find xy values of water 
# molecules on the plane (within in the rectangular); 4. define 41 points on 
# the plane to place an ion or an atom of any other type that is defined in 
# force field; 5. define a cutoff to compute number of water molecules within
# the defined cutoff distance of each of the 41 points and find the one that 
# has the largest number of neighboring water molecules; 6. the point from step
# 5 is the point representing the center of the slice of the water cylinder 
# on the 2D x-y plance and will be used for centering the the water cylinder 
# in the MD simulation box. 
#
# Writting by Jingjing Huang on Nov, 2020
# email jjhuang0520@outlook.com
# ---------------------------------------------------------------------------

import mdtraj as md
import numpy as np
import itertools
import math
import heapq
from sys import argv
from subprocess import call
from collections import Counter
import pickle
import os, sys, errno
import re
import argparse
import multiprocessing
from collections import defaultdict
import time

def CAL_CofM(traj_,atom_ndx):
    '''
    Return the center of mass of the selected atoms in 
    a MD trajectory
    '''
    top_ = traj_.topology
    masses = np.array([top_.atom(a).element.mass for a in atom_ndx])
    masses /= masses.sum()
    return np.einsum('ijk,k->ij',np.swapaxes(traj_.xyz[:,atom_ndx,:].astype('float64'),1,2), masses)

def get_dist(a,b):
    '''
    Compute the distance between atoms
    ''' 
    #ab_dist = np.sum((a - b)**2,axis=2)**(1/2.)
    if a.ndim == 3:
        return np.sum((a - b)**2,axis=2)**(1/2.)
    elif a.ndim == 2:
        return np.sum((a - b)**2,axis=1)**(1/2.)
    else:
        return math.sqrt(np.array([i ** 2 for i in (np.array(a)-np.array(b))]).sum())

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot random data in parallel')
    parser.add_argument('-o', '--outputDir', required=True,
                        help='The directory to which files should be saved')

    parser.add_argument('-np', '--numProcessors', required=False, type=int,
                                        default=multiprocessing.cpu_count(),
                                        help='Number of processors to use. ' + \
                                        "Default for this machine is %d" % (multiprocessing.cpu_count(),) )

    parser.add_argument('-ix', '--inputXTC', required=True,
                        help='Input XTC file')

    parser.add_argument('-ig', '--inputGRO', required=True,
                        help='Input GRO file')

    #args = parser.parse_args()
    #parser.set_defaults()
    args=parser.parse_args()

    gro_input = args.inputGRO
    xtc_input = args.inputXTC

    traj = md.load(xtc_input, top = gro_input)
    top  = traj.topology

    # define variables for setting the 41 points on the 2D x-y plane
    global  point_Y0X0,  point_Y0X1,  point_Y0X2,  point_Y0X3,  point_Y0X4
    global  point_Y1X0,  point_Y1X1,  point_Y1X2,  point_Y1X3,  point_Y1X4
    global  point_Y2X0,  point_Y2X1,  point_Y2X2,  point_Y2X3,  point_Y2X4
    global  point_Y3X0,  point_Y3X1,  point_Y3X2,  point_Y3X3,  point_Y3X4
    global  point_Y4X0,  point_Y4X1,  point_Y4X2,  point_Y4X3,  point_Y4X4
    global  point_W1, point_W2, point_W3, point_W4, point_W5, point_W6, point_W7, point_W8
    global  point_W9, point_W10, point_W11, point_W12, point_W13, point_W14, point_W15, point_W16
    
    # define some other variables
    global  mask_z,      wat_xy,      oct_z

    # select atoms
    octane = top.select("resname OCT and mass > 2")
    water  = top.select("water and mass > 2")
    
    # compute the center of mass of the membrane for the entire trajectory
    oct_com = CAL_CofM(traj, octane)
 
    # find the z-values of the center of mass for all frames
    oct_z = np.swapaxes(np.swapaxes(oct_com,0,1)[-1:],0,1)

    # get box dimensions for all frames
    box_dim = traj.unitcell_vectors
    box_x = np.array([ x[0] for x in np.swapaxes(box_dim,0,1)[0]])
    box_y = np.array([ x[1] for x in np.swapaxes(box_dim,0,1)[1]])

    pZERO = np.zeros(len(box_x))  # for easy formatting in output
 
    # these are 5*5 points on the 2D grid
    point_Y0X0 = np.vstack((pZERO,      pZERO)).T
    point_Y0X1 = np.vstack((box_x*0.25, pZERO)).T
    point_Y0X2 = np.vstack((box_x*0.50, pZERO)).T
    point_Y0X3 = np.vstack((box_x*0.75, pZERO)).T
    point_Y0X4 = np.vstack((box_x*1.00, pZERO)).T
   
    point_Y1X0 = np.vstack((pZERO,      box_y*0.25)).T
    point_Y1X1 = np.vstack((box_x*0.25, box_y*0.25)).T
    point_Y1X2 = np.vstack((box_x*0.50, box_y*0.25)).T
    point_Y1X3 = np.vstack((box_x*0.75, box_y*0.25)).T
    point_Y1X4 = np.vstack((box_x*1.00, box_y*0.25)).T

    point_Y2X0 = np.vstack((pZERO,      box_y*0.50)).T
    point_Y2X1 = np.vstack((box_x*0.25, box_y*0.50)).T
    point_Y2X2 = np.vstack((box_x*0.50, box_y*0.50)).T
    point_Y2X3 = np.vstack((box_x*0.75, box_y*0.50)).T
    point_Y2X4 = np.vstack((box_x*1.00, box_y*0.50)).T

    point_Y3X0 = np.vstack((pZERO,      box_y*0.75)).T
    point_Y3X1 = np.vstack((box_x*0.25, box_y*0.75)).T
    point_Y3X2 = np.vstack((box_x*0.50, box_y*0.75)).T
    point_Y3X3 = np.vstack((box_x*0.75, box_y*0.75)).T
    point_Y3X4 = np.vstack((box_x*1.00, box_y*0.75)).T

    point_Y4X0 = np.vstack((pZERO,      box_y*1.00)).T
    point_Y4X1 = np.vstack((box_x*0.25, box_y*1.00)).T
    point_Y4X2 = np.vstack((box_x*0.50, box_y*1.00)).T
    point_Y4X3 = np.vstack((box_x*0.75, box_y*1.00)).T
    point_Y4X4 = np.vstack((box_x*1.00, box_y*1.00)).T
    
    # add 16 more points
    point_W1 = np.vstack((box_x*0.125, box_y*0.875)).T
    point_W2 = np.vstack((box_x*0.375, box_y*0.875)).T
    point_W3 = np.vstack((box_x*0.625, box_y*0.875)).T
    point_W4 = np.vstack((box_x*0.875, box_y*0.875)).T

    point_W5 = np.vstack((box_x*0.125, box_y*0.625)).T
    point_W6 = np.vstack((box_x*0.375, box_y*0.625)).T
    point_W7 = np.vstack((box_x*0.625, box_y*0.625)).T
    point_W8 = np.vstack((box_x*0.875, box_y*0.625)).T
    
    point_W9 = np.vstack((box_x*0.125, box_y*0.375)).T
    point_W10 = np.vstack((box_x*0.375, box_y*0.375)).T
    point_W11 = np.vstack((box_x*0.625, box_y*0.375)).T
    point_W12 = np.vstack((box_x*0.875, box_y*0.375)).T

    point_W13 = np.vstack((box_x*0.125, box_y*0.125)).T
    point_W14 = np.vstack((box_x*0.375, box_y*0.125)).T
    point_W15 = np.vstack((box_x*0.625, box_y*0.125)).T
    point_W16 = np.vstack((box_x*0.875, box_y*0.125)).T

    # get x, y, z for waters
    wat_x   = traj.xyz[:,water,:1]
    wat_y   = traj.xyz[:,water,1:2]
    wat_z   = traj.xyz[:,water,-1:]
    
    wat_xyz = traj.xyz[:,water,:]
    wat_xy = traj.xyz[:,water,:-1]
    
    wat_x_r = np.array([ax.ravel() for ax in wat_x])
    wat_y_r = np.array([ay.ravel() for ay in wat_y])
    wat_z_r = np.array([az.ravel() for az in wat_z])

    # define a boundary along the z-axis
    mask_z = np.logical_and((oct_z-0.1 < wat_z_r),(wat_z_r < oct_z+0.1))

    def Find_Psudo_Dot(frameID):

        ndx = frameID; cutoff = 2.85
        # get water molecules on the xy-plane approximately at the z of COM of OCT
        wat_xyon2D = wat_xy[ndx][mask_z[ndx]] 
           
        wat_to_Y0X0 = get_dist(wat_xyon2D, point_Y0X0[frameID][np.newaxis])
        wat_to_Y0X1 = get_dist(wat_xyon2D, point_Y0X1[frameID][np.newaxis])
        wat_to_Y0X2 = get_dist(wat_xyon2D, point_Y0X2[frameID][np.newaxis])
        wat_to_Y0X3 = get_dist(wat_xyon2D, point_Y0X3[frameID][np.newaxis])
        wat_to_Y0X4 = get_dist(wat_xyon2D, point_Y0X4[frameID][np.newaxis]) 

        wat_to_Y1X0 = get_dist(wat_xyon2D, point_Y1X0[frameID][np.newaxis])
        wat_to_Y1X1 = get_dist(wat_xyon2D, point_Y1X1[frameID][np.newaxis])
        wat_to_Y1X2 = get_dist(wat_xyon2D, point_Y1X2[frameID][np.newaxis])
        wat_to_Y1X3 = get_dist(wat_xyon2D, point_Y1X3[frameID][np.newaxis])
        wat_to_Y1X4 = get_dist(wat_xyon2D, point_Y1X4[frameID][np.newaxis])
        
        wat_to_Y2X0 = get_dist(wat_xyon2D, point_Y2X0[frameID][np.newaxis])
        wat_to_Y2X1 = get_dist(wat_xyon2D, point_Y2X1[frameID][np.newaxis])
        wat_to_Y2X2 = get_dist(wat_xyon2D, point_Y2X2[frameID][np.newaxis])
        wat_to_Y2X3 = get_dist(wat_xyon2D, point_Y2X3[frameID][np.newaxis])
        wat_to_Y2X4 = get_dist(wat_xyon2D, point_Y2X4[frameID][np.newaxis])

        wat_to_Y3X0 = get_dist(wat_xyon2D, point_Y3X0[frameID][np.newaxis])
        wat_to_Y3X1 = get_dist(wat_xyon2D, point_Y3X1[frameID][np.newaxis])
        wat_to_Y3X2 = get_dist(wat_xyon2D, point_Y3X2[frameID][np.newaxis])
        wat_to_Y3X3 = get_dist(wat_xyon2D, point_Y3X3[frameID][np.newaxis])
        wat_to_Y3X4 = get_dist(wat_xyon2D, point_Y3X4[frameID][np.newaxis])

        wat_to_Y4X0 = get_dist(wat_xyon2D, point_Y4X0[frameID][np.newaxis])
        wat_to_Y4X1 = get_dist(wat_xyon2D, point_Y4X1[frameID][np.newaxis])
        wat_to_Y4X2 = get_dist(wat_xyon2D, point_Y4X2[frameID][np.newaxis])
        wat_to_Y4X3 = get_dist(wat_xyon2D, point_Y4X3[frameID][np.newaxis])
        wat_to_Y4X4 = get_dist(wat_xyon2D, point_Y4X4[frameID][np.newaxis])


        wat_to_W1 = get_dist(wat_xyon2D, point_W1[frameID][np.newaxis]); wat_to_W2 = get_dist(wat_xyon2D, point_W2[frameID][np.newaxis]); wat_to_W3 = get_dist(wat_xyon2D, point_W3[frameID][np.newaxis]); wat_to_W4 = get_dist(wat_xyon2D, point_W4[frameID][np.newaxis]);
        wat_to_W5 = get_dist(wat_xyon2D, point_W5[frameID][np.newaxis]); wat_to_W6 = get_dist(wat_xyon2D, point_W6[frameID][np.newaxis]); wat_to_W7 = get_dist(wat_xyon2D, point_W7[frameID][np.newaxis]); wat_to_W8 = get_dist(wat_xyon2D, point_W8[frameID][np.newaxis]);
        wat_to_W9 = get_dist(wat_xyon2D, point_W9[frameID][np.newaxis]); wat_to_W10 = get_dist(wat_xyon2D, point_W10[frameID][np.newaxis]); wat_to_W11 = get_dist(wat_xyon2D, point_W11[frameID][np.newaxis]); wat_to_W12 = get_dist(wat_xyon2D, point_W12[frameID][np.newaxis]);
        wat_to_W13 = get_dist(wat_xyon2D, point_W13[frameID][np.newaxis]); wat_to_W14 = get_dist(wat_xyon2D, point_W14[frameID][np.newaxis]); wat_to_W15 = get_dist(wat_xyon2D, point_W15[frameID][np.newaxis]); wat_to_W16 = get_dist(wat_xyon2D, point_W16[frameID][np.newaxis]);

        # count the number of water molecules that are within the cutoff distance of each of the 41 points on the 2D x-y plane; cutoff nm is the distance cut-off 
        count_Y0X0 = np.sum((wat_to_Y0X0 < cutoff),axis=0); count_Y1X0 = np.sum((wat_to_Y1X0 < cutoff),axis=0)
        count_Y0X1 = np.sum((wat_to_Y0X1 < cutoff),axis=0); count_Y1X1 = np.sum((wat_to_Y1X1 < cutoff),axis=0)
        count_Y0X2 = np.sum((wat_to_Y0X2 < cutoff),axis=0); count_Y1X2 = np.sum((wat_to_Y1X2 < cutoff),axis=0)
        count_Y0X3 = np.sum((wat_to_Y0X3 < cutoff),axis=0); count_Y1X3 = np.sum((wat_to_Y1X3 < cutoff),axis=0)
        count_Y0X4 = np.sum((wat_to_Y0X4 < cutoff),axis=0); count_Y1X4 = np.sum((wat_to_Y1X4 < cutoff),axis=0) 

        count_Y2X0 = np.sum((wat_to_Y2X0 < cutoff),axis=0); count_Y3X0 = np.sum((wat_to_Y3X0 < cutoff),axis=0)
        count_Y2X1 = np.sum((wat_to_Y2X1 < cutoff),axis=0); count_Y3X1 = np.sum((wat_to_Y3X1 < cutoff),axis=0)
        count_Y2X2 = np.sum((wat_to_Y2X2 < cutoff),axis=0); count_Y3X2 = np.sum((wat_to_Y3X2 < cutoff),axis=0)
        count_Y2X3 = np.sum((wat_to_Y2X3 < cutoff),axis=0); count_Y3X3 = np.sum((wat_to_Y3X3 < cutoff),axis=0)
        count_Y2X4 = np.sum((wat_to_Y2X4 < cutoff),axis=0); count_Y3X4 = np.sum((wat_to_Y3X4 < cutoff),axis=0) 

        count_Y4X0 = np.sum((wat_to_Y4X0 < cutoff),axis=0)
        count_Y4X1 = np.sum((wat_to_Y4X1 < cutoff),axis=0)
        count_Y4X2 = np.sum((wat_to_Y4X2 < cutoff),axis=0)
        count_Y4X3 = np.sum((wat_to_Y4X3 < cutoff),axis=0)
        count_Y4X4 = np.sum((wat_to_Y4X4 < cutoff),axis=0)

        count_W1 = np.sum((wat_to_W1 < cutoff),axis=0); count_W5 = np.sum((wat_to_W5 < cutoff),axis=0)
        count_W2 = np.sum((wat_to_W2 < cutoff),axis=0); count_W6 = np.sum((wat_to_W6 < cutoff),axis=0)
        count_W3 = np.sum((wat_to_W3 < cutoff),axis=0); count_W7 = np.sum((wat_to_W7 < cutoff),axis=0)
        count_W4 = np.sum((wat_to_W4 < cutoff),axis=0); count_W8 = np.sum((wat_to_W8 < cutoff),axis=0)

        count_W9 = np.sum((wat_to_W9 < cutoff),axis=0); count_W13 = np.sum((wat_to_W13 < cutoff),axis=0)
        count_W10 = np.sum((wat_to_W10 < cutoff),axis=0); count_W14 = np.sum((wat_to_W14 < cutoff),axis=0)
        count_W11 = np.sum((wat_to_W11 < cutoff),axis=0); count_W15 = np.sum((wat_to_W15 < cutoff),axis=0)
        count_W12 = np.sum((wat_to_W12 < cutoff),axis=0); count_W16 = np.sum((wat_to_W16 < cutoff),axis=0)

        # due to PBC, some points are overlapped
        counts_ = [count_Y0X0+count_Y0X4+count_Y4X0+count_Y4X4, count_Y0X1+count_Y4X1, count_Y0X2+count_Y4X2, count_Y0X3+count_Y4X3,
                   count_Y1X0+count_Y1X4, count_Y1X1, count_Y1X2, count_Y1X3, count_Y1X4+count_Y1X0,
                   count_Y2X0+count_Y2X4, count_Y2X1, count_Y2X2, count_Y2X3, count_Y2X4+count_Y2X0,
                   count_Y3X0+count_Y3X4, count_Y3X1, count_Y3X2, count_Y3X3, count_Y3X4+count_Y3X0,
                   count_Y4X1+count_Y0X1, count_Y4X2+count_Y0X2, count_Y4X3+count_Y0X3,
                   count_W1,count_W2,count_W3,count_W4,
                   count_W5,count_W6,count_W7,count_W8,
                   count_W9,count_W10,count_W11,count_W12,
                   count_W13,count_W14,count_W15,count_W16]

        points_ = [point_Y0X0[ndx], point_Y0X1[ndx], point_Y0X2[ndx], point_Y0X3[ndx],
                   point_Y1X0[ndx], point_Y1X1[ndx], point_Y1X2[ndx], point_Y1X3[ndx], point_Y1X4[ndx],
                   point_Y2X0[ndx], point_Y2X1[ndx], point_Y2X2[ndx], point_Y2X3[ndx], point_Y2X4[ndx],
                   point_Y3X0[ndx], point_Y3X1[ndx], point_Y3X2[ndx], point_Y3X3[ndx], point_Y3X4[ndx],
                   point_Y4X1[ndx], point_Y4X2[ndx], point_Y4X3[ndx],
                   point_W1[ndx],point_W2[ndx],point_W3[ndx],point_W4[ndx],
                   point_W5[ndx],point_W6[ndx],point_W7[ndx],point_W8[ndx],
                   point_W9[ndx],point_W10[ndx],point_W11[ndx],point_W12[ndx],
                   point_W13[ndx],point_W14[ndx],point_W15[ndx],point_W16[ndx]]

        # find the point that has the largest number of water molecules within the cutoff distance
        which_points = counts_.index(np.asarray(counts_).max())
        
        Psudo = points_[which_points]
            
        Psudo_z = round(oct_z[ndx][0],3)
       
        # output one line as in the GRO file format
        outputfile = open("./ndx_sets/index_PBC_"+str(frameID)+".txt","w")
        # 4 an 6 are empty varibale for formatting
        outputfile.write(str('{0:<8}   {1:>4}{2:<7}{3:>6}{4:<2}{5:>6}{6:<2}{7:>6}'.format("27537CL","CL", "19185",format(Psudo[0],'.3f'),"",format(Psudo[1],'.3f'),"",format(Psudo_z,'.3f')))+"\n")
        outputfile.close()
        
        return Psudo 

    Numframe = traj.n_frames

    for fndx in range(Numframe):
        Find_Psudo_Dot(fndx)
