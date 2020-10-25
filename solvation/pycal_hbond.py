import mdtraj as md

### It is interesting that GRO doesn't have bonds information ###

def _get_bond_triplets_(topology, input_reswatndx, sidechain_only=False): #reswatid: list of strings
    """
    input_reswatidx(list of strings): must be all atom index 
    Resid doesn't work because PDB file maximum index
    
    #resname+resid format; eg, ALA120, HOH120, NA120
    
    """ 
    def can_participate(atom):
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        if str(atom.index) not in input_reswatndx:
            return False
        # Otherwise, accept it
        if str(atom.residue) == 'HOH17948': ## why ?
            print 'fuck'
        return True   

    def get_donors(e0, e1):
        # Find all matching bonds
        elems = set((e0, e1))
        atoms = [(one, two) for one, two in topology.bonds
            if set((one.element.symbol, two.element.symbol)) == elems]

        # Filter non-participating atoms
        atoms = [atom for atom in atoms
            if can_participate(atom[0]) and can_participate(atom[1])]

        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    nh_donors = get_donors('N', 'H')
    oh_donors = get_donors('O', 'H')
    
    xh_donors = np.array(nh_donors + oh_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(('O', 'N'))
    acceptors = [a.index for a in topology.atoms
        if a.element.symbol in acceptor_elements and can_participate(a)]

    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
    return bond_triplets[np.logical_not(self_bond_mask), :]

def _compute_bounded_geometry_(traj, triplets, distance_indices=[1, 2], 
                              angle_indices=[0, 1, 2], periodic=True):
    """
    Returns a tuple include (1) the mask for triplets that fulfill the distance
    criteria frequently enough, (2) the actual distances calculated, and (3) the
    angles between the triplets specified by angle_indices.
    """
    # First we calculate the requested distances
    distances = md.compute_distances(traj, triplets[:, distance_indices], periodic=periodic)

    # Calculate angles using the law of cosines
    abc_pairs = zip(angle_indices, angle_indices[1:] + angle_indices[:1])
    abc_distances = []

    # Calculate distances (if necessary)
    for abc_pair in abc_pairs:
        if set(abc_pair) == set(distance_indices):
            abc_distances.append(distances)
        else:
            abc_distances.append(md.compute_distances(traj, triplets[:, abc_pair],
                periodic=periodic))

    # Law of cosines calculation
    a, b, c = abc_distances
    cosines = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    np.clip(cosines, -1, 1, out=cosines) # avoid NaN error
    angles = np.arccos(cosines)

    triplets = np.repeat([triplets], traj.n_frames, axis=0)
    
    return distances, angles, triplets

def md_cal_hbonding(traj, input_reswatndx, distance_cutoff=0.35,angle_cutoff=0,
                    periodic=True, sidechain_only=False):
    """
    distance_cutoff = 0.35                    # nanometers
    angle_cutoff_radians = 2.0 * np.pi / 3.0  # radians
    angle_cutoff = 120                        # deg
    """
    angle_cutoff_radians=(angle_cutoff/float(180))*np.pi
    
    if traj.topology is None:
        raise ValueError('hbonding requires that traj contain topology '
                         'information')

    # Get the possible donor-hydrogen...acceptor triplets
    bond_triplets = _get_bond_triplets_(traj.topology,
        input_reswatndx=input_reswatndx, sidechain_only=sidechain_only)

    distances, angles, triplets = _compute_bounded_geometry_(traj, bond_triplets, periodic=periodic)

    # Find triplets that meet the criteria
    presence = np.logical_and(distances < distance_cutoff, angles > angle_cutoff_radians)
    #mask[mask] = np.mean(presence, axis=0) > freq
    
    triplets_hbond=[];distance_hbond=[];angles_hbond=[]
    for filt, pairs,dist,ang in zip(presence,triplets,distances,angles):
        triplets_hbond.append(pairs[np.array(filt)])
        distance_hbond.append(dist[np.array(filt)])
        angles_hbond.append(ang[np.array(filt)])
      
    return triplets_hbond,distance_hbond,angles_hbond

def gen_output_data(topology,wat_resid_list,
                    triplets_hbond,distance_hbond,angles_hbond,output_name):
    
    protein=topology.select('protein')
    wat_atom_idx_resid=defaultdict(list)
    for watid in wat_resid_list:
        ### watid - 1 python index format
        wat_ndx_list=[a.index for a in topology.residue(int(watid)-1).atoms]
        for watndx in wat_ndx_list:
            ### Python atom index - VMD resid
            wat_atom_idx_resid[str(watndx)].append('HOH'+str(watid)) 
        
    pro_atom_idx_resid=defaultdict(list)
    for index in protein:
        ### Python atom index - VMD resid
        pro_atom_idx_resid[str(index)].append(str(topology.atom(index).residue)) 
    
    wat_pro_all_dict=pro_atom_idx_resid.copy()
    wat_pro_all_dict.update(wat_atom_idx_resid)
    
    outputfile = open(output_name,"w")

    for frameID,(pairs_h,dist_h,ang_h) in enumerate(zip(triplets_hbond,distance_hbond,angles_hbond)):
    ## FrameID + donor index + acceptor index + donor resname + donor_resid + donor_atom
    ## + acceptor_resnm + acceptor_resid + acceptor_atom + distance + angle
        for p_h, d_h, a_h in zip(pairs_h,dist_h,ang_h):
            outputfile.write('frame_'+str(frameID)+" "+str(p_h[0])+" "+str(p_h[2])+" "+\
                         str(wat_pro_all_dict[str(p_h[0])][0])[:3]+" "+\
                         str(wat_pro_all_dict[str(p_h[0])][0])[3:]+" "+\
                         str(top.atom(p_h[1])).split('-')[-1]+" "+\
                         str(wat_pro_all_dict[str(p_h[2])][0])[:3]+" "+\
                         str(wat_pro_all_dict[str(p_h[2])][0])[3:]+" "+\
                         str(top.atom(p_h[2])).split('-')[-1]+" "+\
                         str(d_h)+" "+str(a_h)+"\n")
    
    outputfile.close()


tunnel_atoms=open('atoms.txt').readlines()[4].strip('\n').strip(' ').split()
wat_all_list=np.load('temp_wat_all_list.npy')

xtcfile='atpase_popc_oct_2_run_deloct_cont11time_150mM_NaCl_BREAK_MORE_L50ns_skip50_noPBC_onestep_centered.xtc'
grofile='atpase_popc_oct_2_run_deloct_cont11time_150mM_NaCl_BREAK_MORE_rename.pdb'

traj=md.load(xtcfile,top=grofile)
top=traj.topology

wat_atoms_input=[]
for watid in wat_all_list:
    ### watid - 1 python index format
    wat_ndx_list=[a.index for a in top.residue(int(watid)-1).atoms]
    wat_atoms_input.extend(wat_ndx_list)

all_input=np.concatenate([tunnel_atoms,wat_atoms_input])

triplets_hbond,distance_hbond,angles_hbond=md_cal_hbonding(traj, all_input, distance_cutoff=0.35,angle_cutoff=0,periodic=True, sidechain_only=False)

gen_output_data(top,wat_all_list,triplets_hbond,distance_hbond,angles_hbond,name?)

