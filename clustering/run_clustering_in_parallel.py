#
# This script loads MD trajectories and clusters frame/snapshots
#
# trajectories were loaded by using mdtraj 
# clustering was performed by using msmbuilder
# python package multiprocessing was used to implement OpenMP 
#
#
# In this script, hundreds of simulation trajectories were loaded and were first atom-sliced to make
# sure that all trajectories have same number atoms; clustering was performed later.
#
# by Jingjing Huang; email at jjhuang0520@gmail.com
#

import mdtraj as md
import pandas as pd
import numpy as np
from msmbuilder.featurizer import RMSDFeaturizer
#import matplotlib.pyplot as plt
from msmbuilder.dataset import dataset
from msmbuilder.decomposition import PCA
from msmbuilder.cluster import KMedoids
from collections import defaultdict
import multiprocessing 
import os, sys, errno
import re
import argparse
import time
import gc
from msmbuilder.cluster import MiniBatchKMedoids

def intersect(*d):
    sets = iter(map(set, d))
    result = sets.next()
    for s in sets:
        result = result.intersection(s)
    return result

def load_xtc(files):
    gro, xtc = files

    traj_load = md.load(xtc, top=gro,stride=10)  # for cont1time
    top_load = traj_load.topology
    wat_resid = np.asarray([int(str(res)[3:]) \
                for res in top_load.residues if (str(res)[:3] == 'HOH')])
    oct_num = len(top_load.select('resname OCT'))

    return traj_load, top_load, wat_resid, oct_num

def load_tar_xtc(files):
    gro, xtc = files

    traj_load = md.load(xtc, top=gro,stride=5) ## for cont1time
    top_load = traj_load.topology
                        
    return traj_load, top_load

def traj_slice(files):
    traj, atom_ndx = files
    return traj.atom_slice(atom_ndx)


if __name__ == '__main__':
    #outputDir, group, repeat)

    parser = argparse.ArgumentParser(description='Plot random data in parallel')
    parser.add_argument('-o', '--outputDir', required=True,
                        help='The directory to which plot files should be saved')

    parser.add_argument('-g', '--group_list', required=True, nargs='+', default=1,
                                                    help='The list of groupID')
    parser.add_argument('-nc', '--cluster_num', required=True, type=int, default=1,
                                                    help='The number of clusters')
    parser.add_argument('-np', '--numProcessors', required=False, type=int,
                                        default=multiprocessing.cpu_count(),
                                        help='Number of processors to use. ' + \
                                        "Default for this machine is %d" % (multiprocessing.cpu_count(),) )
    args = parser.parse_args()

    if not os.path.isdir(args.outputDir) or not os.access(args.outputDir, os.W_OK):
        sys.exit("Unable to write to output directory %s" % (args.outputDir,) )

    if args.numProcessors < 1:
        sys.exit('Number of processors to use must be greater than 0')

       
    prefix="/scratch/jjhuang/projects/project_2/m2_virusa_cut_random_real/m2_virusa_cut_random_real_2loj/files_for_analysis_cont1time/"

    gro = prefix+ 'group_%s/m2_virusa_npt_tiltrots_%s_rots_2LOJ_03_cont1time_rename.gro'
    xtc = prefix+ 'group_%s/m2_virusa_npt_tiltrots_%s_rots_2LOJ_03_cont1time_noPBC_3steps.xtc'
    
    # just making filenames for the first 10 trajectories
    inputGro = [ gro % (g_pg,g_fg) for g_pg,g_fg in zip(args.group_list,args.group_list) ]
    inputXtc = [ xtc % (x_pg,x_fg) for x_pg,x_fg in zip(args.group_list,args.group_list) ]
 
    # make a pool for loading
    t1=time.time()
    pool_load = multiprocessing.Pool( args.numProcessors )

    results = pool_load.map(load_xtc, zip(inputGro, inputXtc))

    pool_load.close()
    pool_load.join()

    ref_traj_list = [t[0] for t in results]
    ref_top_list =  [t[1] for t in results]
    ref_wat_resid_list = [t[2] for t in results]
    ref_oct_num_list = [t[3] for t in results]
    
    t2=time.time()
    print ("Load ref trajs is done", t2-t1)
    # make a pool for loading

    gc.collect()

    ref_wat_resid_intersect=np.asarray(list(set(ref_wat_resid_list[0]).intersection(*ref_wat_resid_list)))
    for index,num in enumerate(ref_oct_num_list):
        if num == np.asarray(ref_oct_num_list).min():
            ref_oct_index=ref_top_list[index].select('resname OCT')
    ref_pro_index=ref_top_list[0].select('protein')
  
    #print ("ref_wat_resid_intersect")

    ref_slice_pro_oct_wat_list=[]
    for top in ref_top_list:
        ref_wat_index_rd_one_rpa=top.select('resid '+str(ref_wat_resid_intersect[0])+' to '+str(ref_wat_resid_intersect[-2]))
        #print ref_wat_index_rd_one_rpa
        ref_slice_pro_oct_wat_list.append(np.concatenate((ref_pro_index,ref_oct_index,ref_wat_index_rd_one_rpa),axis=0))
   
    print ("get ref slicing list: ref_slice_pro_oct_wat_list, is done")
  
    # make a pool for slicing
    t1=time.time()
    pool_slice = multiprocessing.Pool( args.numProcessors )

    results_slice  = pool_slice.map(traj_slice, zip(ref_traj_list,ref_slice_pro_oct_wat_list))

    pool_slice.close()
    pool_slice.join()

    ref_traj_sliced_list = results_slice

    t2=time.time()
    print ("Slicing ref trajs is done", t2-t1)


    # free some memory
    del ref_oct_num_list, ref_wat_resid_list, results_slice
    gc.collect()
 
    #print ("del ref_oct_num_list, ref_wat_resid_list")

    ### join trajs 01
    t1=time.time()
    def traj_join_01(ndxs):
        global ref_traj_sliced_list

        traj_chunk = ref_traj_sliced_list[int(ndxs[0]):int(ndxs[1])]
        return md.join(traj_chunk, check_topology=False, discard_overlapping_frames=False)

    r1 = range(0,xxxx+1,10)
    # make a first pool for combining trajs; break down to 40 chunks
    pool_join_01 = multiprocessing.Pool( args.numProcessors )
    
    results_join_01 = pool_join_01.map(traj_join_01, zip([i for i in r1], r1[1:]))
   
    pool_join_01.close()
    pool_join_01.join()
    t2=time.time()
    print ("join ref trajs 01", t2-t1)
    gc.collect()

    ### join trajs 02
    t1=time.time()
    def traj_join_02(ndxs):
        global results_join_01 

        traj_chunk = results_join_01[int(ndxs[0]):int(ndxs[1])]
        return md.join(traj_chunk, check_topology=False, discard_overlapping_frames=False)

    #r2 = range(0,40+1,5)
    r2 = range(0,zzzz+1,2) # use 20 chunks for large dataset; 8 chunk is too big for parallel
    # make a second pool for combining trajs; break down to 8 chunks
    pool_join_02 = multiprocessing.Pool( args.numProcessors )

    results_join_02 = pool_join_02.map(traj_join_02, zip([i for i in r2], r2[1:]))

    pool_join_02.close()
    pool_join_02.join()
    t2=time.time()
    print ("join ref trajs 02", t2-t1)
 
    # combine the final 8 chunks
    t1=time.time()
    combined_ref_traj = md.join(results_join_02, check_topology=False, discard_overlapping_frames=False)
    t2=time.time()
    print ("combined_ref_traj", t2-t1)

    c_alpha=ref_traj_sliced_list[0].topology.select("protein and name CA")

    # free some memory
    del results_join_01, results_join_02, ref_traj_sliced_list, results, ref_wat_resid_intersect, ref_traj_list, ref_top_list, t2, t1
    gc.collect()

    featurizer = RMSDFeaturizer(reference_traj=combined_ref_traj, atom_indices=c_alpha) # Delete the first frame

    # free some memory
    del combined_ref_traj
    gc.collect()

    # loading target trajs 

    t1=time.time()
    pool_targetload = multiprocessing.Pool( args.numProcessors )

    results_targetload = pool_targetload.map(load_tar_xtc, zip(inputGro, inputXtc))
  
    pool_targetload.close()
    pool_targetload.join()

    target_traj_list = [t[0] for t in results_targetload]
    t2=time.time()
    print ("Loading target trajs", t2-t1)
    
    # free some memory
    del results_targetload
    gc.collect()

    # make a pool for slicing target trajs
    t1=time.time()
    pool_targetslice = multiprocessing.Pool( args.numProcessors )

    target_traj_sliced_list  = pool_targetslice.map(traj_slice, zip(target_traj_list,ref_slice_pro_oct_wat_list))

    pool_targetslice.close()
    pool_targetslice.join()

    t2=time.time()
    print ("Slicing target trajs", t2-t1)

    # free some memory
    del target_traj_list
    gc.collect()

    t1=time.time()
    rmsd = featurizer.transform(target_traj_sliced_list)
    #np.save("feature_rmsd_group_1_400.npy",rmsd)
    t2=time.time()
    print ("featurization time", t2-t1)

    # do clustering
    t1=time.time()
    n_clust = args.cluster_num
    cluster = MiniBatchKMedoids(n_clusters=n_clust,random_state=None)
    cluster.fit(rmsd)
    t2=time.time()
    print ("clustering time", t2-t1)

    # free some memory
    del rmsd
    gc.collect()

    for clust in range(n_clust):
       
        frames_within_clust_dict = defaultdict(list)
        clustpairs = np.where(np.array(cluster.labels_)==clust)
        for tnum, fnum in zip(clustpairs[0],clustpairs[1]):
            frames_within_clust_dict[tnum].append(fnum)

        def find_traj_and_frames(traj_ndx):
            global target_traj_sliced_list
            global frames_within_clust_dict
                     
            traj = target_traj_sliced_list[traj_ndx]
            return traj.slice(frames_within_clust_dict[traj_ndx])

        # open a pool for finding the frames
        t1=time.time()
        pool_clustering = multiprocessing.Pool( args.numProcessors )

        results_clustering = pool_clustering.map(find_traj_and_frames, frames_within_clust_dict.keys())

        pool_clustering.close()
        pool_clustering.join()
        t2=time.time()
        print ("result_clustering ", "round- ", clust, "time- ", t2-t1)
        gc.collect()

        t1=time.time()
        cluster_A = md.join(results_clustering, check_topology=False, discard_overlapping_frames=False)
        t2=time.time()
        print ("md join cluster_A ", "round- ", clust, "time- ", t2-t1)

        cluster_A.save('cluster_'+str(clust)+'.h5')
        
        # free some memory
        del cluster_A
        gc.collect()                

