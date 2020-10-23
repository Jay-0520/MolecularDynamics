import mdtraj as md
import numpy as np
from collections import defaultdict
from collections import Counter
import itertools
from collections import Counter


prefix='/scratch/jjhuang/projects/project_rcf2/Rcf2_dimer_popc_at/files_for_analysis/'

for group in range(1,20):
    location='group_rcf2_'+str(group)
    xtcfile=prefix+location+'/rcf2_popc_wat_100mmNaCl_'+str(group)+'_run_skip50_noPBC_TOTAL600ns.xtc'
    grofile=prefix+location+'/rcf2_popc_wat_100mmNaCl_'+str(group)+'_run_cont5time_rename.gro'

    traj=md.load(xtcfile,top=grofile)
    top=traj.topology
# for TM 2 and TM 5 that defining the interface
#top.select('protein and sidechain and (name OD1 or name OE2 or name OD2 or name OE1 or name OG or name O1 or name O2 or name OH or name NE2 or name NE1 or name NE or name NH1 or name NH2 or name ND1 or name ND2 or name NZ or name SG or name SD)')
    monomer_01=top.select("protein and resid 45 to 75 or resid 183 to 203 and mass > 2")
    monomer_02=top.select("protein and resid 269 to 299 or resid 407 to 427 and mass > 2")

    rcf1_index_dict = defaultdict(list)
    for residue_id in range(0,448):
        for idx in top.select('resid '+str(residue_id)):
            rcf1_index_dict[str(idx)].append(str(residue_id))

    all_pairs_mons=np.array(list(itertools.product(monomer_01,monomer_02)))
    all_respairs_mons = [(rcf1_index_dict[str(i)][0], rcf1_index_dict[str(j)][0]) for i,j in all_pairs_mons]

    distances_all = md.compute_distances(traj, all_pairs_mons, periodic=False, opt=True)

    counts_less_than_cutoff = (distances_all < 0.45)


    counts_all_residues = {}
    for t, count_at_time_t in enumerate(counts_less_than_cutoff):
        temp_counts_per_respair = {}

        for residue_pair, count in zip(all_respairs_mons, count_at_time_t): ## in this 
            #all_res_two_pairs_without_self, same residue pair may exist multiple times because for one residue pair, 
            #it includes many atom pairs whose distance is within cut-off.
            if count and abs(int(residue_pair[0])-int(residue_pair[1])) > 2:
                temp_counts_per_respair[residue_pair] = 1 ### if it's true, then add residue pair to dict, only consider 
                # different residue pairs. If same residue pair exist multiple times, it's still counted once!!!
                # which means we only think about contacts between residues.
                #print temp_counts_per_residue
            #else:
                #temp_counts_per_respair[residue_pair] = 0 ###seems it's not necessary
        counts_all_residues[t] = Counter(temp_counts_per_respair)

    num_intermon_contact=[]
    for t, counts_at_t_dict in counts_all_residues.items():
        num_intermon_contact.append(sum(counts_at_t_dict.values()))

    np.save("num_interface_contact_popc_"+str(group)+".npy",np.asarray(num_intermon_contact))



