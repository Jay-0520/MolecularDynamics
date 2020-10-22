import mdtraj as md
import numpy as np
import multiprocessing
import os, sys, errno
import re
import argparse
import time
import gc


def rmsd_nmr(files):
    gro, xtc = files
    traj= md.load(xtc,top=gro)
    top=traj.topology

    protein_index=top.select('protein')
    protein_ca=top.select('protein and name CA')
    #protein_ca=top.select('protein and name CA and (resid 11 to 41 or resid 45 to 75 or resid 104 to 134 or resid 142 to 172 or resid 183 to 203 or resid 235 to 265 or resid 269 to 299 or resid 328 to 358 or resid 366 to 396 or resid 407 to 427)')

    traj_sliced=traj.atom_slice(protein_index)
    #pnr="/scratch/jjhuang/projects/project_rcf2/OCT_dimers_100NaCl/results_analysis/native_structure/"
    ref_gro='/scratch/jjhuang/projects/project_pgaa/simulations_pgaa_monomer_neut/results_analysis/time_evolution_of_rmsd/reference/PgaA_pro_cap.gro'
    ref_xtc='/scratch/jjhuang/projects/project_pgaa/simulations_pgaa_monomer_neut/results_analysis/time_evolution_of_rmsd/reference/PgaA_pro_cap.gro'
    ref_traj=md.load(ref_xtc,top=ref_gro)

    #traj_all_models = []
    #for num in range(0,15):
    traj_rmsd=md.rmsd(traj_sliced, ref_traj, frame=0, atom_indices=protein_ca, parallel=True, precentered=False)
    #    traj_all_models.append(traj_rmsd)


    #np.save("rmsd_array_of_cluster_"+str(num)+".npy",np.amin(rmsd_all_models,axis=0))
    #return np.amin(rmsd_all_models,axis=0)
    #return np.mean(np.asarray(traj_all_models),axis=0)
    return traj_rmsd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cal RMSD for clusters')
    parser.add_argument('-o', '--outputDir', required=True,
                            help='The directory to which plot files should be saved')
    parser.add_argument('-np', '--numProcessors', required=False, type=int,
                                default=multiprocessing.cpu_count(),
                                help='Number of processors to use. ' + \
                                "Default for this machine is %d" % (multiprocessing.cpu_count(),) )

    args = parser.parse_args()

    prefix="/scratch/jjhuang/projects/project_pgaa/simulations_pgaa_dimer_mix/files_for_analysis/"

    gro = prefix+ 'group_%s/pgaadimwat_ions_25mm_%s_run_cont2time_rename.gro'
    xtc = prefix+ 'group_%s/pgaadimwat_ions_25mm_%s_run_skip50_noPBC_TOTAL300ns.xtc'

    inputGro = [ gro % (g_pg,g_fg) for g_pg,g_fg in zip(range(1,21),range(1,21)) ]
    inputXtc = [ xtc % (x_pg,x_fg) for x_pg,x_fg in zip(range(1,21),range(1,21)) ]

    # make a pool for loading
    pool_load = multiprocessing.Pool( args.numProcessors )

    results = pool_load.map(rmsd_nmr, zip(inputGro, inputXtc))

    pool_load.close()
    pool_load.join()

    for ndx,rmsd in enumerate(results):
        np.save("rmsd_dimer_mix_TM_group_"+str(ndx+1)+".npy", rmsd)


