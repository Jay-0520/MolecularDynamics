import mdtraj as md
import numpy as np
from collections import defaultdict
from collections import Counter
import itertools
from itertools import product

def best_hummer_interface_inter(traj,native,sb_ndx_mon_01,sb_ndx_mon_02):
    BETA_CONST = 50  # 1/nm2
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers
    sb_pairs = np.array(
        [(i,j) for (i,j) in itertools.product(sb_ndx_mon_01,sb_ndx_mon_02)
            if abs(native.topology.atom(i).residue.index - \
                   native.topology.atom(j).residue.index) > 2])

    sb_pairs_distances = md.compute_distances(native[0], sb_pairs)[0]
    native_sb_contacts = sb_pairs[sb_pairs_distances < NATIVE_CUTOFF]

    r = md.compute_distances(traj, native_sb_contacts)
    r0 = md.compute_distances(native[0], native_sb_contacts)

    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q

def best_hummer_interhlx_all(traj,native,sb_ndx_01,sb_ndx_02):
    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers
    sb_pairs = np.array(
        [(i,j) for (i,j) in product(sb_ndx_01, sb_ndx_02)
            if abs(native.topology.atom(i).residue.index - \
                   native.topology.atom(j).residue.index) > 2])

    sb_pairs_distances = md.compute_distances(native[0], sb_pairs)[0]
    native_sb_contacts = sb_pairs[sb_pairs_distances < NATIVE_CUTOFF]

    r = md.compute_distances(traj, native_sb_contacts)
    r0 = md.compute_distances(native[0], native_sb_contacts)

    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q

ref_gro='/scratch/jjhuang/projects/project_rcf2/Rcf2_dimer_popc_at/results_analysis/total_30_repeats/native_structure/rcf2_p.gro'
ref_xtc='/scratch/jjhuang/projects/project_rcf2/Rcf2_dimer_popc_at/results_analysis/total_30_repeats/native_structure/rcf2_p.xtc'
ref_traj=md.load(ref_xtc,top=ref_gro)

print(ref_traj)

prefix='/scratch/jjhuang/projects/project_rcf2/Rcf2_dimer_popc_at/files_for_analysis/'

for group in range(1,31):
    location='group_rcf2_'+str(group)
    xtcfile=prefix+location+'/rcf2_popc_wat_100mmNaCl_'+str(group)+'_run_skip50_noPBC_TOTAL600ns.xtc'
    grofile=prefix+location+'/rcf2_popc_wat_100mmNaCl_'+str(group)+'_run_cont5time_rename.gro'

    traj=md.load(xtcfile,top=grofile)
    top=traj.topology

    mon_1_h1=top.select("protein and resid 11 to 41 and mass > 2")
    mon_1_h2=top.select("protein and resid 45 to 75 and mass > 2")
    mon_1_h3=top.select("protein and resid 104 to 134 and mass > 2")
    mon_1_h4=top.select("protein and resid 142 to 172 and mass > 2")
    mon_1_h6=top.select("protein and resid 183 to 203 and mass > 2")

    mon_2_h1=top.select("protein and resid 235 to 265 and mass > 2")
    mon_2_h2=top.select("protein and resid 269 to 299 and mass > 2")
    mon_2_h3=top.select("protein and resid 328 to 358 and mass > 2")
    mon_2_h4=top.select("protein and resid 366 to 396 and mass > 2")
    mon_2_h6=top.select("protein and resid 407 to 427 and mass > 2")


    rcf1_index_dict = defaultdict(list)
    for residue_id in range(0,448):
        for idx in top.select('resid '+str(residue_id)):
            rcf1_index_dict[str(idx)].append(str(residue_id))

    mon_1_list=[mon_1_h1,mon_1_h2,mon_1_h3,mon_1_h4,mon_1_h6]
    mon_2_list=[mon_2_h1,mon_2_h2,mon_2_h3,mon_2_h4,mon_2_h6]

    h1_fract_all_list=[]
    for num in itertools.combinations(mon_1_list,2):
        h1_fraction_interhlx_list=[]
        for numhhh in range(15):
            h1_fraction_interhlx = best_hummer_interhlx_all(traj,ref_traj[numhhh],num[0],num[1])
            h1_fraction_interhlx_list.append(h1_fraction_interhlx)
        h1_fract_all_list.append(h1_fraction_interhlx_list)        

    h2_fract_all_list=[]
    for num in itertools.combinations(mon_2_list,2):
        h2_fraction_interhlx_list=[]
        for numhhh in range(15):
            h2_fraction_interhlx = best_hummer_interhlx_all(traj,ref_traj[numhhh],num[0],num[1])
            h2_fraction_interhlx_list.append(h2_fraction_interhlx)
        h2_fract_all_list.append(h2_fraction_interhlx_list)


    np.save("./data/fracinterhlx_conts_popc_"+str(group)+"_m1.npy", np.asarray(h1_fract_all_list))
    np.save("./data/fracinterhlx_conts_popc_"+str(group)+"_m2.npy", np.asarray(h2_fract_all_list))
   
    mon_1_in_list=[mon_1_h2, mon_1_h6]
    mon_2_in_list=[mon_2_h2, mon_2_h6] 

    fraction_interface_list=[]
    for num in itertools.product(mon_1_in_list,mon_2_in_list):
        temp_list=[]
        for numhhh in range(15):
            fraction_interface = best_hummer_interface_inter(traj,ref_traj[numhhh],num[0],num[1])
            temp_list.append(fraction_interface)
        fraction_interface_list.append(temp_list)

    np.save("./data/fracinterhlx_conts_popc_"+str(group)+".npy", np.asarray(fraction_interface_list))
