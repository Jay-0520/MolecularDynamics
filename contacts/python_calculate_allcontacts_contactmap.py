import mdtraj as md
import numpy as np
from collections import defaultdict
from collections import Counter
import itertools
from collections import Counter


prefix='/scratch/jjhuang/projects/project_rcf2/Rcf2_dimer_popc_at/files_for_analysis/'

for group in range(1,21):
    #selfassembly_200mmdpc_dimer_10_norestrs 
    location='group_rcf2_'+str(group)
    xtcfile=prefix+location+'/rcf2_popc_wat_100mmNaCl_'+str(group)+'_run_skip50_noPBC_TOTAL600ns_L100ns.xtc'
    grofile=prefix+location+'/rcf2_popc_wat_100mmNaCl_'+str(group)+'_run_cont5time_rename.gro'

    traj=md.load(xtcfile,top=grofile)
    top=traj.topology
    protein=top.select('protein and mass > 2')

    rcf1_index_dict = defaultdict(list)
    for residue_id in range(0,448):
        for idx in top.select('resid '+str(residue_id)):
            rcf1_index_dict[str(idx)].append(str(residue_id))
    
    all_two_pairs=np.array(list(itertools.permutations(protein, 2)))
    distances_all = md.compute_distances(traj, all_two_pairs, periodic=False, opt=True)

    complete_pairs=[]
    for i in range(0,448):
        for j in range(0,448):
            complete_pairs.append([i,j])
    
    count_resid_dict=defaultdict(list)
    for respair in complete_pairs:
        count_resid_dict[str(respair[0])+' '+str(respair[1])]=0
    
    
    for f_distan in distances_all:
        check_respairs_dict=defaultdict(list)
        for cherespair in complete_pairs:
            check_respairs_dict[str(cherespair[0])+' '+str(cherespair[1])]=0
        for distan, respair in zip(f_distan,all_two_pairs):
            if distan < 0.6 :
                check_respairs_dict[str(rcf1_index_dict[str(respair[0])][0])+' '+str(rcf1_index_dict[str(respair[1])][0])]+=1
                if check_respairs_dict[str(rcf1_index_dict[str(respair[0])][0])+' '+str(rcf1_index_dict[str(respair[1])][0])] < 2:
                    count_resid_dict[str(rcf1_index_dict[str(respair[0])][0])+' '+str(rcf1_index_dict[str(respair[1])][0])]+=1

    order_count_std_list=[]
    order_count_mean_list=[]
    for c_pair in complete_pairs:
        num=count_resid_dict[str(c_pair[0])+' '+str(c_pair[1])]
        num_one=np.ones(num)
        num_zero=np.zeros(len(traj)-num)
        std_mean_array=np.concatenate([num_one,num_zero],axis=0)
        order_count_std_list.append(np.std(std_mean_array))
        order_count_mean_list.append(np.mean(std_mean_array))
    
    order_count_std_2D_list=[]
    order_count_mean_2D_list=[]
    count=0
    for i in range(0,448):
        order_count_std_2D_01=[]
        order_count_mean_2D_01=[]
        count=i*448
        for j in range(count,count+448):
            order_count_std_2D_01.append(np.asarray(order_count_std_list[j]))
            order_count_mean_2D_01.append(np.asarray(order_count_mean_list[j]))
        order_count_std_2D_list.append(np.asarray(order_count_std_2D_01))
        order_count_mean_2D_list.append(np.asarray(order_count_mean_2D_01))
    
    np.save('allcontacts_contactmap_std_'+str(group),np.asarray(order_count_std_2D_list))
    np.save('allcontacts_contactmap_mean_'+str(group),np.asarray(order_count_mean_2D_list))




