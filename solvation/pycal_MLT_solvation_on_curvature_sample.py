#-------------------------------------------------------------------------
# This script compute number of H-bonds for gromacs trajectory of molecular 
# dynamics (MD) simulations
#
# This script takes as input the H-bonds profile generated from the MDAnalysis
# package. This script computes H-bonds of only peptides within a certain 
# boundaries along the z-axis. For example, in this version, only the peptides
# in the membrane region are considered. In this version, the MD system consists
# of melittins in octane/water membrane, and we focused on H-bonds of melittins 
# to water molecules.
#
# Writting by Jingjing Huang on Nov, 2020
# email jjhuang0520@outlook.com
# ---------------------------------------------------------------------------


import mdtraj as md
import numpy as np
from heapq import nlargest, nsmallest
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict

prefix='/scratch/jjhuang/projects/melittin_poration/melittin_surf_tm/transmembrane/symmetric_essemble/files_for_analysis_whole/'

MLT_group=['XXX']
RPA_group=['1', '2', '3', '4', '5']

# the total number of atoms in a melittin peptide is 26
mlt_IDs = []
for num in range(0,9):
    mlt_IDs.append([num*26,num*26+25])

for numM in MLT_group:
    for numR in RPA_group:
        # load gromacs trajectories using the MDtraj package
        location=str(numM)+'_melittin/'+str(numM)+'_melittin_'+str(numR)+'_replica'
        xtcfile=prefix+location+'/melittin_'+str(numM)+'_replica_'+str(numR)+'_03_cont5time_skip10_noPBC.xtc'
        grofile=prefix+location+'/melittin_'+str(numM)+'_replica_'+str(numR)+'_03_cont5time_rename_Add.gro'

        traj=md.load(xtcfile,top=grofile)
        top=traj.topology
        protein=top.select('protein')
        octane=top.select('resname OCT')

        # compute the xyz for all octane molecules
        oct_z = traj.xyz[:,octane,2] # note the difference in array shape if we use traj.xyz[:,octane,2:]

        # find 200 largest and 200 smallest z-values of octane atoms
        Large_list=[]; Small_list=[]
        for fz in oct_z:
            Large_list.append(np.ravel(np.asarray(nlargest(200, fz))))
            Small_list.append(np.ravel(np.asarray(nsmallest(200, fz))))

        # define the upbound based on the average value of the 200 largest z-values of octane atoms
        upbound = np.mean(np.asarray(Large_list),axis=1)  -0.5
        # define the lowerbound based on the averaged value of the 200 smallest z-values of octane atoms
        lowbound = np.mean(np.asarray(Small_list),axis=1) +0.5
        
        # 434 is the total atom number for one MLT peptide
        # 51 is the frame number
        upbound_2D=np.reshape(np.repeat(upbound, 434, axis=0),(51,434)) 
        lowbound_2D=np.reshape(np.repeat(lowbound, 434, axis=0),(51,434))
        
        # dict that links MLT ID to frameID
        mlt_ndx_mask_dict=defaultdict(list)
        # dict that links MLT ID to MLT resID
        mlt_atoms_ndx_dict=defaultdict(list)

        # here, we're trying to get residue IDs of melittins that are in the membrane region (e.g., the z-value of melittin residues are smaller than the upper bound and larger than the lower bound) 
        for ndx,IDs in enumerate(mlt_IDs[:XXX]): # 5 for 5 MLTs 
            mlt_atoms=top.select('protein and resid '+str(IDs[0])+' to '+str(IDs[1]))
            mlt_atoms_ndx_dict[str(ndx)].append(range(int(IDs[0]),int(IDs[1])+1))
            
            # get z values for all mlt atoms
            mlt_z = traj.xyz[:,mlt_atoms,2] 
            
            # consider a MLT to be in the membrane region (along the water cylinder) 
            # if half of the peptide is within the region lowbound-upbound
            check_up = mlt_z < np.asarray(upbound_2D)
            check_low = mlt_z > np.asarray(lowbound_2D)
        
            # MlT kinks at the PRO, and there are 203 atoms in the sequence from the N-terminal to the PRO
            # 200 is approximately half the length of a MLT
            mask_up = np.sum(check_up, axis=1) > 200
            mask_low = np.sum(check_low, axis=1) > 200
            mask = [all(m_) for m_ in zip(mask_up, mask_low)]
            
            # get the frames that have MLTs in the membrane region
            frameID=np.where(mask)
            mlt_ndx_mask_dict[str(ndx)].append(np.ravel(frameID).tolist())
        
        # get the dict that links frameID to MLT resID
        # each key (frameID) of this dictionary is linked to residue IDs of MLTs that in the membrane region  
        mlt_frame_atoms_dict=defaultdict(list)
        for frameID in range(0,51):
            for keys_, items_ in mlt_ndx_mask_dict.items():
                if frameID in items_[0]: # add "[0]" to get the list
                    mlt_frame_atoms_dict[str(frameID)].append(mlt_atoms_ndx_dict[str(keys_)][0]) # add "[0]" to get the list
        
       
        # load the h-bonds profile generated by the MDAnalysis package 
        file_name='../mlt_'+str(numM)+'_rpt_'+str(numR)+'_data.dat'
        hbonds_lines=open(file_name,"r").readlines()


        # group h-bond instances by frameID
        hbonds_frameID_dict = defaultdict(list)
        for l in hbonds_lines:
            hbond_frameID = int(float(l.split()[0])/200)
            hbonds_frameID_dict[hbond_frameID].append(l.split())

        # check each frame and find h-bond instances only for the residues of the melittins that are in the membrane region
        hbonds_store=[]
        for frameID in hbonds_frameID_dict.keys():
            hbonds_counter = defaultdict(int)
            hbonds_store_per_frame=[]
            for line in hbonds_frameID_dict[frameID]:
                
                # line[4] is 1-indexed but mlt_frame_atoms_dict is 0-indexed
                # line[4] is a string but mlt_frame_atoms_dict[key] outputs ints
                if line[3] != 'SOL':
                    hbonds_counter[line[4]] += 1
                if line[6] != 'SOL':
                    hbonds_counter[line[7]] += 1
                        
            for i in range(1,XXX*26+1): # the output from MDAnalysis is 1-indexed
                hbonds_store_per_frame.append(hbonds_counter[str(i)])
            hbonds_store.append(hbonds_store_per_frame)
        
         
        # here we assign nan values to the residues of those melittins that are not in the membrane region
        hbonds_store_filt = []
        for frameID, hbonds in enumerate(hbonds_store):
            
            nan_positions = list(itertools.chain.from_iterable(mlt_frame_atoms_dict[str(frameID)]))
            for ndx, hb in enumerate(hbonds):
                if ndx not in nan_positions:
                    hbonds[ndx] = np.nan
                    
            hbonds_reshape = np.reshape(hbonds,(XXX,26))
            hbonds_store_filt.append(hbonds_reshape)
            
        hbonds_store_filt = np.asarray(hbonds_store_filt).reshape(XXX*51,26) # 51*5 = 255
       
        hbonds_mean = np.nanmean(hbonds_store_filt,axis=0) 

        np.save('mlt_'+str(numM)+'_rpt_'+str(numR)+'_hbonds_mean.npy', hbonds_mean)
