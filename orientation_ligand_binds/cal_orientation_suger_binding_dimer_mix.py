import mdtraj as md
import numpy as np
from collections import defaultdict
from collections import Counter
import itertools
from collections import Counter
import math
import pickle

def get_dist(a,b):
    return math.sqrt(np.array([i ** 2 for i in (np.array(a)-np.array(b))]).sum())

def vlen(v):
    ###Calculate the length of vector
    return np.sqrt(np.sum([ i*i for i in v ]))

def vangle(v1, v2):
    ###Calculat the angle beteween two vectors
    norm_v1 = v1 / vlen(v1)
    norm_v2 = v2 / vlen(v2)
    ang = np.degrees(np.arccos(round(np.vdot(norm_v1, norm_v2),5)))
    return ang

group_id=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']


prefix='/scratch/jjhuang/projects/project_pgaa/simulations_pgaa_monomer_neut/files_for_analysis/'

for num in ["XXX"]:
    group='group_'+str(num)
#pgaanagwat_ions_50mm_10_run_cont2time_L10ns_noPBC.xtc
#pgaanagwat_ions_50mm_10_run_cont2time_rename.gro    
    xtcfile=prefix+group+'/pgaanagwat_ions_50mm_'+str(num)+'_run_skip50_noPBC_L100ns.xtc'
    grofile=prefix+group+'/pgaanagwat_ions_50mm_'+str(num)+'_run_cont2time_rename.gro'

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
     
    dist_ang_list_all_95 = []
    dist_ang_list_all_99 = []
    for t, count_at_time_t in enumerate(counts_less_than_cutoff):
        dist_ang_list_95 = [] 
        dist_ang_list_99 = []   
        for residue_pair, count in zip(all_respairs_without_self, count_at_time_t): ## in this 
            #all_res_two_pairs_without_self, same residue pair may exist multiple times because for one residue pair, 
            #it includes many atom pairs whose distance is within cut-off.
            if count and residue_pair[0] in ["95"]: # 96 and 100 
                TRP_ID = int(residue_pair[0]) 
                SUG_ID = int(residue_pair[1])
                TRP_ndx = top.select("protein and resid "+str(TRP_ID)+" and (name CH2 or name CZ3 or name CE3 \
                or name CD2 or name CD1 or name CG or name NE1 or name CE2 or name CZ2)") 
                SUG_ndx01 = top.select("resid "+str(SUG_ID)+" and (name C1 or name C2 or name C3 or name C4 or \
                name C5 or name O5)")
                SUG_ndx02 = top.select("resid 129 and (name C or name O or name C7 or name C9 or name C10 or name C11)")

                TRP_masses = np.array([top.atom(a).element.mass for a in TRP_ndx])
                TRP_masses /= TRP_masses.sum()
                TRP_com=traj.xyz[t,TRP_ndx].astype('float64').T.dot(TRP_masses)
                
                SUG_masses01 = np.array([top.atom(a).element.mass for a in SUG_ndx01])
                SUG_masses01 /= SUG_masses01.sum()
                SUG_com01=traj.xyz[t,SUG_ndx01].astype('float64').T.dot(SUG_masses01)
                
                SUG_masses02 = np.array([top.atom(a).element.mass for a in SUG_ndx02])
                SUG_masses02 /= SUG_masses02.sum()
                SUG_com02=traj.xyz[t,SUG_ndx02].astype('float64').T.dot(SUG_masses02)
                
                dist_TRP_SUG01 = get_dist(TRP_com,SUG_com01)
                dist_TRP_SUG02 = get_dist(TRP_com,SUG_com02)
                
                if  dist_TRP_SUG01 >  dist_TRP_SUG02:
                    dist_TRP_SUG = dist_TRP_SUG02
                    SUG_ndx = SUG_ndx02
                else:
                    dist_TRP_SUG = dist_TRP_SUG01
                    SUG_ndx = SUG_ndx01
                
                # get plane normal for TRP
                #TRP96-CG TRP96-CE2 TRP96-CZ3
                t1, t2, t3 = traj.xyz[t,TRP_ndx[::3]]
                vt1 = t3 - t1
                vt2 = t2 - t1
                TRP_normal = np.cross(vt1, vt2)
                # get plane normal for SUG
                #DMP132-C1 DMP132-C5 DMP132-C3:
                s1, s2, s3 = traj.xyz[t,SUG_ndx[::2]] 
                vs1 = s3 - s1
                vs2 = s2 - s1
                SUG_normal = np.cross(vs2, vs1)           
               
                ang_TRP_SUG = vangle(TRP_normal,SUG_normal)
                
                TRP_CG  = top.select("resid "+str(TRP_ID)+" and (name CG)")[0] # 1549 
                TRP_CZ3 = top.select("resid "+str(TRP_ID)+" and (name CZ3)")[0] # 1618
                TRP_CG_xyz  = traj.xyz[t,TRP_CG,]
                TRP_CZ3_xyz = traj.xyz[t,TRP_CZ3,]  
                # C1, O5, C5, C4, C3 , C2 
                # C,  O,  C7, C9, C10, C11
                SUG_C2 =  SUG_ndx[5]
                SUG_C5 =  SUG_ndx[2]
                SUG_C2_xyz = traj.xyz[t,SUG_C2,]
                SUG_C5_xyz = traj.xyz[t,SUG_C5,]
                Vec_SUG =  SUG_C5_xyz - SUG_C2_xyz
                Vec_TRP =  TRP_CZ3_xyz - TRP_CG_xyz
                Vec_ang_TRP_SUG = vangle(Vec_SUG,Vec_TRP)      

                #dist_ang_list.append([t,SUG_ID, dist_TRP_SUG,ang_TRP_SUG])                
                dist_ang_list_95.append([t,SUG_ID,dist_TRP_SUG,ang_TRP_SUG,Vec_ang_TRP_SUG])
                #temp_counts_per_respair[residue_pair] = 1 ### if it's true, then add residue pair to dict, only consider 
                # different residue pairs. If same residue pair exist multiple times, it's still counted once!!!
                # which means we only think about contacts between residues.
                #print temp_counts_per_residue
            #else:
                #temp_counts_per_respair[residue_pair] = 0 ###seems it's not necessary
            if count and residue_pair[0] in ["99"]: # 96 and 100 
                TRP_ID = int(residue_pair[0])
                SUG_ID = int(residue_pair[1])
                TRP_ndx = top.select("protein and resid "+str(TRP_ID)+" and (name CH2 or name CZ3 or name CE3 \
                or name CD2 or name CD1 or name CG or name NE1 or name CE2 or name CZ2)")
                #SUG_ndx = top.select("resid "+str(SUG_ID)+" and (name C1 or name C2 or name C3 or name C4 or \
                #name C5 or name O5)")
                SUG_ndx01 = top.select("resid "+str(SUG_ID)+" and (name C1 or name C2 or name C3 or name C4 or \
                name C5 or name O5)")
                SUG_ndx02 = top.select("resid 129 and (name C or name O or name C7 or name C9 or name C10 or name C11)")

                TRP_masses = np.array([top.atom(a).element.mass for a in TRP_ndx])
                TRP_masses /= TRP_masses.sum()
                TRP_com=traj.xyz[t,TRP_ndx].astype('float64').T.dot(TRP_masses)
                
                SUG_masses01 = np.array([top.atom(a).element.mass for a in SUG_ndx01])
                SUG_masses01 /= SUG_masses01.sum()
                SUG_com01=traj.xyz[t,SUG_ndx01].astype('float64').T.dot(SUG_masses01)
                
                SUG_masses02 = np.array([top.atom(a).element.mass for a in SUG_ndx02])
                SUG_masses02 /= SUG_masses02.sum()
                SUG_com02=traj.xyz[t,SUG_ndx02].astype('float64').T.dot(SUG_masses02)
                
                dist_TRP_SUG01 = get_dist(TRP_com,SUG_com01)
                dist_TRP_SUG02 = get_dist(TRP_com,SUG_com02)
                
                if  dist_TRP_SUG01 >  dist_TRP_SUG02:
                    dist_TRP_SUG = dist_TRP_SUG02
                    SUG_ndx = SUG_ndx02
                else:
                    dist_TRP_SUG = dist_TRP_SUG01
                    SUG_ndx = SUG_ndx01
                
                # get plane normal for TRP
                t1, t2, t3 = traj.xyz[t,TRP_ndx[::3]]
                vt1 = t3 - t1
                vt2 = t2 - t1
                TRP_normal = np.cross(vt1, vt2)
                # get plane normal for SUG
                s1, s2, s3 = traj.xyz[t,SUG_ndx[::2]] 
                vs1 = s3 - s1
                vs2 = s2 - s1
                SUG_normal = np.cross(vs2, vs1)           
               
                ang_TRP_SUG = vangle(TRP_normal,SUG_normal)
                
                TRP_CG  = top.select("resid "+str(TRP_ID)+" and (name CG)")[0] # 1549 
                TRP_CZ3 = top.select("resid "+str(TRP_ID)+" and (name CZ3)")[0] # 1618
                TRP_CG_xyz  = traj.xyz[t,TRP_CG,]
                TRP_CZ3_xyz = traj.xyz[t,TRP_CZ3,]  
                # C1, O5, C5, C4, C3, C2 
                SUG_C2 =  SUG_ndx[5]
                SUG_C5 =  SUG_ndx[2]
                SUG_C2_xyz = traj.xyz[t,SUG_C2,]
                SUG_C5_xyz = traj.xyz[t,SUG_C5,]
                Vec_SUG =  SUG_C5_xyz - SUG_C2_xyz
                Vec_TRP =  TRP_CZ3_xyz - TRP_CG_xyz
                Vec_ang_TRP_SUG = vangle(Vec_SUG,Vec_TRP) 
 
                dist_ang_list_99.append([t,SUG_ID,dist_TRP_SUG,ang_TRP_SUG,Vec_ang_TRP_SUG])

               
        if not dist_ang_list_95:
            dist_ang_list_all_95.append(dist_ang_list_95)
        else:
            dist_ang_list_all_95.append(np.unique(dist_ang_list_95, axis=0).tolist())
    
        if not dist_ang_list_99:
            dist_ang_list_all_99.append(dist_ang_list_99)
        else:
            dist_ang_list_all_99.append(np.unique(dist_ang_list_99, axis=0).tolist())

    #print ()
    #np.save('orientation_sugar_binding_rpa_'+str(num), dist_ang_list_all, allow_pickle=False)
    with open('dist_ang_list_all_'+str(num)+'_T95_.pickle', 'wb') as handle:
        pickle.dump(dist_ang_list_all_95, handle, protocol=2)

    with open('dist_ang_list_all_'+str(num)+'_T99_.pickle', 'wb') as handle:
        pickle.dump(dist_ang_list_all_99, handle, protocol=2)

