import mdtraj as md
import numpy as np
from collections import defaultdict
from collections import Counter
import itertools
import math
import multiprocessing
import os, sys, errno
import re
import argparse
import time
import gc


def sse_nmr(files):
    gro, xtc = files
    traj= md.load(xtc,top=gro)
    top=traj.topology

    mon_1_h1=top.select("protein and resid 11 to 41")
    mon_1_h2=top.select("protein and resid 45 to 75")
    mon_1_h3=top.select("protein and resid 104 to 134")
    mon_1_h4=top.select("protein and resid 142 to 172")
    mon_1_h5=top.select("protein and resid 183 to 203")

    mon_2_h1=top.select("protein and resid 235 to 265")
    mon_2_h2=top.select("protein and resid 269 to 299")
    mon_2_h3=top.select("protein and resid 328 to 358")
    mon_2_h4=top.select("protein and resid 366 to 396")
    mon_2_h5=top.select("protein and resid 407 to 427")

    mon_all_list=[mon_1_h1,mon_1_h2,mon_1_h3,mon_1_h4,mon_1_h5,mon_2_h1,mon_2_h2,mon_2_h3,mon_2_h4,mon_2_h5]
    #mon_2_list=[mon_2_h1,mon_2_h2,mon_2_h3,mon_2_h4,mon_2_h6]

    mon_all_hlc_all=[]
    for hlx_index in mon_all_list:
        protein_index=hlx_index
        traj_sliced=traj.atom_slice(protein_index)

        sec_struc=md.compute_dssp(traj_sliced, simplified=True)

        frac_helix=[]
        for frame in sec_struc:
            frac_helix.append(list(frame).count('H')/float(len(sec_struc[0])))

        mon_all_hlc_all.append(frac_helix)

    #np.save('frac_hlx_oct_'+str(group)+'.npy',np.asarray(mon_all_hlc_all))
    return np.asarray(mon_all_hlc_all)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cal helicity for clusters')
    parser.add_argument('-o', '--outputDir', required=True,
                            help='The directory to which plot files should be saved')
    parser.add_argument('-np', '--numProcessors', required=False, type=int,
                                default=multiprocessing.cpu_count(),
                                help='Number of processors to use. ' + \
                                "Default for this machine is %d" % (multiprocessing.cpu_count(),) )

    args = parser.parse_args()

    prefix="/scratch/jjhuang/projects/project_rcf2/Rcf2_dimer_popc_at/files_for_analysis/"

    gro = prefix+ 'group_rcf2_%s/rcf2_popc_wat_100mmNaCl_%s_run_cont5time_rename.gro'
    xtc = prefix+ 'group_rcf2_%s/rcf2_popc_wat_100mmNaCl_%s_run_skip50_noPBC_TOTAL600ns.xtc'

    inputGro = [ gro % (g_pg,g_fg) for g_pg,g_fg in zip(range(1,21),range(1,21)) ]
    inputXtc = [ xtc % (x_pg,x_fg) for x_pg,x_fg in zip(range(1,21),range(1,21)) ]

    # make a pool for loading
    pool_load = multiprocessing.Pool( args.numProcessors )

    results = pool_load.map(sse_nmr, zip(inputGro, inputXtc))

    pool_load.close()
    pool_load.join()


    for ndx,rmsd in enumerate(results):
        np.save("frac_helicity_popc_group_"+str(ndx+1)+".npy", rmsd)
