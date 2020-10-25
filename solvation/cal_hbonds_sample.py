import numpy as np
import MDAnalysis
import MDAnalysis.analysis.hbonds



prefix='/scratch/jjhuang/projects/project_rcf2/Rcf2_dimer_popc_at/files_for_analysis/'

group_id=['XXX']

for group in group_id:
    location1 = 'group_rcf2_'+str(group)+'/'
    xtcfile = prefix+location1+'rcf2_popc_wat_100mmNaCl_'+str(group)+'_run_skip50_noPBC_TOTAL600ns.xtc'
#   grofile = prefix+location1+location2+location3+'/m2_virusa_npt_tiltrots_x'+str(x)+'_y0_z'+str(z)+'_rots'+str(rots)+'_03.gro'
    grofile = prefix+location1+'rcf2_popc_wat_100mmNaCl_'+str(group)+'_run_cont5time_rename.gro'

    #output_data_pickle='group_'+str(group)+'_data.pickle'
    output_data_dat='group_'+str(group)+'_data.dat'
    
    u = MDAnalysis.Universe(grofile, xtcfile, permissive=True)
    h = MDAnalysis.analysis.hbonds.HydrogenBondAnalysis(u, 'protein', 'resname SOL',distance=3.0,angle=120)
    #h = MDAnalysis.analysis.hbonds.HydrogenBondAnalysis(u, 'protein', 'resname SOL',distance=2.4,angle=0, distance2=3.2)
    results = h.run()
    h.generate_table()
    #h.save_table(output_data_pickle)
     
    np.savetxt(output_data_dat, h.table, delimiter=" ", fmt="%s")



