#
#
#
import mdtraj as md
import numpy as np
from collections import defaultdict
from collections import Counter
import itertools
from collections import Counter


group_id=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']


prefix='/scratch/jjhuang/projects/project_pgaa/simulations_pgaa_monomer_char/files_for_analysis/'

for num in ["XXX"]:
    group='group_'+str(num)
#pgaanagwat_ions_50mm_10_run_cont2time_L10ns_noPBC.xtc
#pgaanagwat_ions_50mm_10_run_cont2time_rename.gro    
    xtcfile=prefix+group+'/pgaanahwat_ions_50mm_'+str(num)+'_run_skip50_noPBC_L100ns.xtc'
    grofile=prefix+group+'/pgaanahwat_ions_50mm_'+str(num)+'_run_cont2time_rename.gro'
    
    traj=md.load(xtcfile,top=grofile)
    top=traj.topology

    protein=top.select("protein and resid 1 to 121 and mass > 2")
    sugar=top.select("not protein and not water and not name CL and not name NA and mass > 2")
    
    protein_sugar_index_dict = defaultdict(list)
    for sugar_id in range(123,144): ###
        for idx in top.select('resid '+str(sugar_id)):
            protein_sugar_index_dict[str(idx)].append(str(sugar_id))
        #if (sugar_id % 2 == 0):
        #    for idx in top.select('resid '+str(sugar_id)):
        #        protein_sugar_index_dict[str(idx)].append(str(sugar_id+1))
        #else:
        #    for idx in top.select('resid '+str(sugar_id)):
        #        protein_sugar_index_dict[str(idx)].append(str(sugar_id))
    for pro_id in range(1,122):
        for idx in top.select('resid '+str(pro_id)):
            protein_sugar_index_dict[str(idx)].append(str(pro_id))   
 
    
    all_pairs_prosug=np.array(list(itertools.product(protein,sugar)))
    #all_pairs_prosug=np.array(list(itertools.combinations(np.concatenate([protein,sugar],axis=0), 2)))
    all_respairs_prosug = [(protein_sugar_index_dict[str(i)][0],
                         protein_sugar_index_dict[str(j)][0]) for i,j in all_pairs_prosug]
    
    
    all_pairs_without_self = []
    for atom_pair, residue_pair in zip(all_pairs_prosug, all_respairs_prosug):
        if residue_pair[0] != residue_pair[1]:
            all_pairs_without_self.append(atom_pair)
            
    print (len(all_pairs_without_self))
    
    all_respairs_without_self = [(protein_sugar_index_dict[str(i)][0],
                                       protein_sugar_index_dict[str(j)][0]) for i,j in all_pairs_without_self]
    
    
    distances_all = md.compute_distances(traj, all_pairs_without_self,
                                         periodic=False, opt=True)
    
    counts_less_than_cutoff = (distances_all < 0.45)
    
    counts_all_residues = {}
    
    for t, count_at_time_t in enumerate(counts_less_than_cutoff):
        temp_counts_per_respair = {}
        
        for residue_pair, count in zip(all_respairs_without_self, count_at_time_t): ## in this 
            #all_res_two_pairs_without_self, same residue pair may exist multiple times because for one residue pair, 
            #it includes many atom pairs whose distance is within cut-off.
            if count: # check for TRP96 and TRP100

                # get xyz for TRP96 and TRP100 and sugar    

                temp_counts_per_respair[residue_pair] = 1 ### if it's true, then add residue pair to dict, only consider 
                # different residue pairs. If same residue pair exist multiple times, it's still counted once!!!
                # which means we only think about contacts between residues.
                #print temp_counts_per_residue
            #else:
                #temp_counts_per_respair[residue_pair] = 0 ###seems it's not necessary
            
        counts_all_residues[t] = Counter(temp_counts_per_respair)
    
    count_pro_contacts_sug=[]
    for (t, counts_at_t_dict),frame in zip(counts_all_residues.items(),traj):
        contacts_sug = Counter()
        for proid in range(1,122):
            #each_resid=[]
            for sugid in range(123,144):      
                contacts_sug[proid] += counts_at_t_dict[(str(sugid), str(proid))] ##Need Str because used manual index projection
                contacts_sug[proid] += counts_at_t_dict[(str(proid), str(sugid))]
        #print contacts_abd
    
        count_pro_contacts_sug.append(list(dict(contacts_sug).values()))


    np.save('count_number_nah_binding_rpa_'+str(num), count_pro_contacts_sug)




