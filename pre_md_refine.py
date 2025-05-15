#code sourced from https://github.com/oxpig/ImmuneBuilder/blob/main/ImmuneBuilder/refine.py
import pdbfixer
import os
import numpy as np
from openmm import app, LangevinIntegrator, CustomExternalForce, CustomTorsionForce, OpenMMException, Platform, unit, Vec3
from scipy import spatial
import json
import glob 

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms
spring_unit = ENERGY / (LENGTH ** 2)

CLASH_CUTOFF = 0.63

# Atomic radii for various atom types.
atom_radii = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}

# Sum of van-der-waals radii
radii_sums = dict(
    [(i + j, (atom_radii[i] + atom_radii[j])) for i in list(atom_radii.keys()) for j in list(atom_radii.keys())])
# Clash_cutoff-based radii values
cutoffs = dict(
    [(i + j, CLASH_CUTOFF * (radii_sums[i + j])) for i in list(atom_radii.keys()) for j in list(atom_radii.keys())])
# Using amber14 recommended protein force field
forcefield = app.ForceField("amber14/protein.ff14SB.xml")

def replace_hetatm_w_atom(pdb_path):
    with open(pdb_path, 'r') as f:
        pdb_lines = f.read()
    modified_pdb_lines = pdb_lines.replace('HETATM', 'ATOM  ') 
    with open(pdb_path, 'w') as f:
        f.writelines(modified_pdb_lines)

def get_residues_idx_from_dict_keys(dict_keys):
    if len(dict_keys) == 0:
        return [] 
    else:
        return [int(key.split('residue')[1]) for key in dict_keys]

def get_platform():
    print('Selecting simulation platform')
    try:
        platform = Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
        platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        print('Using GPU:CUDA')
    except: 
        try:
            platform = Platform.getPlatformByName("OpenCL")
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
            platform.setPropertyDefaultValue('Precision', 'mixed')
            print('Using GPU:OpenCL')
        except:
            platform = Platform.getPlatformByName("CPU")
            print("Switching to CPU, no GPU available.")
    return platform 

def save_structural_issues_info(structural_issues_info, num_refinement_attempts, output_file):

    disulfide_bond_ids = structural_issues_info['disulfide_bond_ids']
    peptide_bond_len_info = structural_issues_info['peptide_bond_len']
    cis_trans_bond_info = structural_issues_info['cis_trans_bond'] 
    chirality_info = structural_issues_info['chirality']
    clash_info = structural_issues_info['clash']
    strained_bond_info = structural_issues_info['strained_bond']

    bad_peptide_bond_residues, cis_residues, d_residues, clashing_atoms, strained_bonds = [], [], [], [], [] 

    bad_peptide_bond_residues = [key for key in peptide_bond_len_info if peptide_bond_len_info[key] != 'within_tolerance']
    cis_residues = [key for key in cis_trans_bond_info if cis_trans_bond_info[key] == 'cis']
    if chirality_info is not None:
        d_residues = [key for key in chirality_info if chirality_info[key] == 'D']
    if clash_info is not None:
        clashing_atoms = [key for key in clash_info if clash_info[key] == 'clash']
    if strained_bond_info is not None:
        strained_bonds = [key for key in strained_bond_info if strained_bond_info[key] == 'strained'] 

    out = {}
    out['num_refinement_attempts'] = num_refinement_attempts
    out['disulfide_bond_ids'] = disulfide_bond_ids
    out['bad_peptide_bond_residues'] = bad_peptide_bond_residues
    out['cis_residues'] = cis_residues
    out['D_residues'] = d_residues
    out['clashing_atoms'] = clashing_atoms
    out['strained_bonds'] = strained_bonds

    no_issues = (len(bad_peptide_bond_residues) == 0) and (len(cis_residues) == 0) and (len(d_residues) == 0) and (len(clashing_atoms) == 0) and len(strained_bonds) == 0
    out['no_issues'] = no_issues

    print('****************************************')
    print('STRUCTURAL ISSUES SUMMARY POST-REFINEMENT AFTER %d ATTEMPTS' % num_refinement_attempts)
    print(out)

    output_file = output_file.replace('.pdb', '_info.json')
    print('saving %s' % output_file)

    with open(output_file, 'w') as json_file:
        json.dump(out, json_file, indent=4)

    return out 

def get_structural_issues_info(input_file, output_file, check_for_strained_bonds=True, n=6, n_threads=-1):

    fixer = pdbfixer.PDBFixer(input_file)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()    

    topology, positions = fixer.topology, fixer.positions

    disulfide_bond_ids, disulfide_bond_residues = get_disulfide_bonds_info(topology, positions)
    peptide_bond_len_info, _ = get_peptide_bond_len_info(topology, positions)
    cis_trans_bond_info, _ = get_cis_trans_bond_info(topology, positions)
    chirality_info, _ = get_chirality_info(topology, positions)
    clash_info, _ = get_clash_info(toplogy, positions)
    _, strained_bond_info, _ = get_strained_sidechain_bonds_info(topology, positions, disulfide_bond_residues)

    bad_peptide_bond_residues = [key for key in peptide_bond_len_info if peptide_bond_len_info[key] != 'within_tolerance']
    cis_residues = [key for key in cis_trans_bond_info if cis_trans_bond_info[key] == 'cis']
    d_residues = [key for key in chirality_info if chirality_info[key] == 'D']
    clashing_atoms = [key for key in clash_info if clash_info[key] == 'clash']
    strained_bonds = [key for key in strained_bond_info if strained_bond_info[key] == 'strained'] 

    out = {}
    out['num_refinement_attempts'] = num_refinement_attempts
    out['disulfide_bond_ids'] = disulfide_bond_ids
    out['bad_peptide_bond_residues'] = bad_peptide_bond_residues
    out['cis_residues'] = cis_residues
    out['D_residues'] = d_residues
    out['clashing_atoms'] = clashing_atoms
    out['strained_bonds'] = strained_bonds
    
    return out 


def refine_multiple_rounds(input_file, output_file, add_disulfide_bonds=True, disulfide_bond_residues_idx=None, check_for_strained_bonds=True, num_attempts=20, num_iterations_per_round=6, custom_k1=None, custom_k2=None, n_threads=-1):

    structural_issues_info_best_round = {}
    prev_num_bad_peptide_bonds_and_cis_residues = 10000
    prev_num_d_residues = 10000 
    platform = get_platform()

    for i in range(num_attempts):
        print('****************************************')
        print("ON ATTEMPT %d" % (i+1))
        print('****************************************')
        pass_all_checks, num_bad_peptide_bonds_and_cis_residues, num_d_residues, structural_issues_info = refine_single_round(input_file, 
                                                                      output_file,
                                                                      platform,
                                                                      prev_num_bad_peptide_bonds_and_cis_residues,
                                                                      prev_num_d_residues,
                                                                      i+1,
                                                                      check_for_strained_bonds, 
                                                                      add_disulfide_bonds,
                                                                      disulfide_bond_residues_idx,
                                                                      num_iterations_per_round, 
                                                                      custom_k1,
                                                                      custom_k2,
                                                                      n_threads)

        if num_bad_peptide_bonds_and_cis_residues <= prev_num_bad_peptide_bonds_and_cis_residues and num_d_residues <= prev_num_d_residues:
            prev_num_bad_peptide_bonds_and_cis_residues = num_bad_peptide_bonds_and_cis_residues
            prev_num_d_residues = num_d_residues
            structural_issues_info_best_round = structural_issues_info

        if pass_all_checks:
            break

    return structural_issues_info_best_round

    


def refine_single_round(input_file, output_file, platform, prev_num_bad_peptide_bonds_and_cis_residues, prev_num_d_residues, round_num, check_for_strained_bonds, add_disulfide_bonds, disulfide_bond_residues_idx, num_iterations_per_round, custom_k1, custom_k2, n_threads):

    fixed_peptide_and_cis = False 
    checked_chirality = False
    pass_all_checks = False

    
    if custom_k1 is not None and custom_k2 is not None:
        k1s = [custom_k1]  
        k2s = [custom_k2]
    else:
        k1s = [2.5,1,0.5,0.25,0.1,0.0001] #last value is to simulate no restraints as final attempt
        k2s = [2.5,5,7.5,15,25,0.0001]

    fixer = pdbfixer.PDBFixer(input_file)

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    k1 = k1s[0]
    k2 = k2s[0]

    print('k1 = %.3f' % k1)
    print('k2 = %.3f' % k2)

    topology, positions = fixer.topology, fixer.positions

    for i in range(num_iterations_per_round):

        print("ON ITERATION %d" % i)

        try:
            if i == 0:
                topology, positions = disulfide_bonds_fixer(topology, positions, platform, add_disulfide_bonds, disulfide_bond_residues_idx)
            cis_trans_bond_info, _ = get_cis_trans_bond_info(topology, positions)
            cis_residues_keys = [key for key in cis_trans_bond_info if cis_trans_bond_info[key] == 'cis']
            cis_residues_idx = get_residues_idx_from_dict_keys(cis_residues_keys)
            peptide_bond_len_info, _ = get_peptide_bond_len_info(topology, positions)
            bad_peptide_bond_residues_keys = [key for key in peptide_bond_len_info if peptide_bond_len_info[key] != 'within_tolerance']
            bad_peptide_bond_residues_idx = get_residues_idx_from_dict_keys(bad_peptide_bond_residues_keys)
        
            print('fixing cis residues:')
            print(cis_residues_idx)
            print('fixing bad peptide bond residues:')
            print(bad_peptide_bond_residues_idx)
            simulation = cis_peptide_bond_fixer(topology, positions, platform, cis_residues_idx, bad_peptide_bond_residues_idx, k1=k1, k2=k2, n_threads=n_threads)
            topology, positions = simulation.topology, simulation.context.getState(getPositions=True).getPositions()
        except OpenMMException as e:
            if (i == num_iterations_per_round-1) and ("positions" not in locals()):
                print("OpenMM failed to refine {}".format(input_file), flush=True)

        peptide_bond_len_info, _ = get_peptide_bond_len_info(topology, positions)
        cis_trans_bond_info, _ = get_cis_trans_bond_info(topology, positions)

        bad_peptide_bond_residues = [key for key in peptide_bond_len_info if peptide_bond_len_info[key] != 'within_tolerance']
        cis_residues = [key for key in cis_trans_bond_info if cis_trans_bond_info[key] == 'cis']

        print('bad peptide bond residues:')
        print(bad_peptide_bond_residues)
        print('cis residues:')
        print(cis_residues)

        #if peptide bonds are the wrong length, decrease the strength of the positional restraint
        if len(bad_peptide_bond_residues) > 0:
            k1 = k1s[min(i+1, len(k1s)-1)]
            print('k1 = %.3f' % k1)

        #if there are still cis isomers in the model, increase the force to fix these
        if len(cis_residues) > 0:
            k2 = k2s[min(i+1, len(k2s)-1)]
            print('k2 = %.3f' % k2)

        disulfide_bond_ids, disulfide_bond_residues = get_disulfide_bonds_info(topology, positions)
        chirality_info, _ = get_chirality_info(topology, positions)
        d_residues = [key for key in chirality_info if chirality_info[key] == 'D']
        
        if len(bad_peptide_bond_residues) == 0 and len(cis_residues) == 0:

            fixed_peptide_and_cis = True 
            #if peptide bond lengths and torsions are okay, check and fix the chirality.
            try:
                print('fixing chirality issues')
                chirality_info, _ = get_chirality_info(topology, positions)
                d_residues_keys = [key for key in chirality_info if chirality_info[key] == 'D']
                print('d residues:')
                print(d_residues_keys)
                d_residues_idx = get_residues_idx_from_dict_keys(d_residues_keys)
                checked_chirality = True 
                simulation = chirality_fixer(simulation, d_residues_idx)
                topology, positions = simulation.topology, simulation.context.getState(getPositions=True).getPositions()
            except OpenMMException as e:
                topology, positions = fixer.topology, fixer.positions
                continue

            if check_for_strained_bonds:
                #if all other checks pass, check and fix strained sidechain bonds:
                try:
                    strained_residues, strained_bond_info, pass_strained_sidechain_check = get_strained_sidechain_bonds_info(topology, positions, disulfide_bond_residues)
                    if len(strained_residues) > 0:
                        print('fixing strained bonds')
                        recheck_strained_residues = True
                        topology, positions = strained_sidechain_bonds_fixer(topology, positions, platform, strained_residues, n_threads=n_threads)
                    else:
                        recheck_strained_residues = False
                except OpenMMException as e:
                    topology, positions = fixer.topology, fixer.positions
                    continue
            else:
                recheck_strained_residues = False

            disulfide_bond_ids, disulfide_bond_residues = get_disulfide_bonds_info(topology, positions)
            peptide_bond_len_info, pass_peptide_bond_len_check = get_peptide_bond_len_info(topology, positions)
            cis_trans_bond_info, pass_cis_check = get_cis_trans_bond_info(topology, positions)
            chirality_info, pass_chirality_check = get_chirality_info(topology, positions)
            clash_info, pass_clash_check = get_clash_info(topology, positions)
            print('clashing atoms:')
            clashing_atoms = [key for key in clash_info if clash_info[key] == 'clash']
            print(clashing_atoms)
            pass_all_checks = pass_peptide_bond_len_check and pass_cis_check and pass_chirality_check and pass_clash_check
            if recheck_strained_residues:
                _, strained_bond_info, pass_strained_sidechain_check = get_strained_sidechain_bonds_info(topology, positions, disulfide_bond_residues)
                pass_all_checks = pass_all_checks and pass_strained_sidechain_check

            bad_peptide_bond_residues = [key for key in peptide_bond_len_info if peptide_bond_len_info[key] != 'within_tolerance']
            cis_residues = [key for key in cis_trans_bond_info if cis_trans_bond_info[key] == 'cis']
            d_residues = [key for key in chirality_info if chirality_info[key] == 'D']
            
            if pass_all_checks:
                break 

    num_bad_peptide_bonds_and_cis_residues = len(bad_peptide_bond_residues) + len(cis_residues) 
    if not(checked_chirality):
        chirality_info, _ = get_chirality_info(topology, positions)
        d_residues = [key for key in chirality_info if chirality_info[key] == 'D']
    num_d_residues = len(d_residues)
    structural_issues_info = {}

    structural_issues_info['disulfide_bond_ids'] = disulfide_bond_ids
    structural_issues_info['peptide_bond_len'] = peptide_bond_len_info
    structural_issues_info['cis_trans_bond'] = cis_trans_bond_info
    if fixed_peptide_and_cis:
        structural_issues_info['chirality'] = chirality_info
        structural_issues_info['clash'] = clash_info
        structural_issues_info['strained_bond'] = strained_bond_info
    else:
        structural_issues_info['chirality'] = None
        structural_issues_info['clash'] = None
        structural_issues_info['strained_bond'] = None 

    if num_bad_peptide_bonds_and_cis_residues <= prev_num_bad_peptide_bonds_and_cis_residues and num_d_residues <= prev_num_d_residues:
        print('SAVING %s' % output_file)
        modeller = app.Modeller(topology, positions)
        with open(output_file, "w") as out_handle:
            app.PDBFile.writeFile(topology, positions, out_handle, keepIds=True)
        structural_issues_info = save_structural_issues_info(structural_issues_info, round_num, output_file) 
    else:
        print('NOT SAVING STRUCTURE BECAUSE NO IMPROVEMENT')
        
    return pass_all_checks, num_bad_peptide_bonds_and_cis_residues, num_d_residues, structural_issues_info


def disulfide_bonds_fixer(topology, positions, platform, add_disulfide_bonds, disulfide_bond_residues_idx, disulfide_bond_cutoff=5.0, n_threads=-1):

    #note: disulfide_bond_residues_idx is 0-indexed 
    if add_disulfide_bonds and disulfide_bond_residues_idx is None:
        print("***Automatically detecting disulfides")
    elif add_disulfide_bonds and disulfide_bond_residues_idx is not None:
        print("***Adding disulfides at specified indices:")
        print(disulfide_bond_residues_idx)
    else:
        print("***Not adding any disulfide bonds")

    modeller = app.Modeller(topology, positions)

    disulfide_bonds = []
    for bond in modeller.topology.bonds():
        if bond.atom1.name == 'SG' and bond.atom2.name == 'SG':
            disulfide_bonds.append(bond)
            print('deleting disulfide bond between atom %d and atom %d' % (bond.atom1.index, bond.atom2.index))
    modeller.delete(disulfide_bonds)

    pos = np.array(modeller.positions.value_in_unit(LENGTH))

    if add_disulfide_bonds and disulfide_bond_residues_idx is None: #autodetect disulfides 
        disulfide_bonds_added = [] 
        for chain in modeller.topology.chains():
            for residue1 in chain.residues():
                for residue2 in chain.residues():
                    if residue1.name == 'CYS' and residue2.name == 'CYS' and residue1.index != residue2.index:
                        atom1 = next(atom for atom in residue1.atoms() if atom.name == 'SG')
                        atom2 = next(atom for atom in residue2.atoms() if atom.name == 'SG')
                        disulfide_dist = round(np.linalg.norm(pos[atom1.index] -  pos[atom2.index]),2)
                        disulfide_bond_id = '%d_%d' % (min(residue1.index,residue2.index),max(residue1.index,residue2.index))
                        if disulfide_dist <= disulfide_bond_cutoff:  
                            if abs(residue2.index-residue1.index) > 2:
                                if disulfide_bond_id not in disulfide_bonds_added:
                                    print('adding disulfide bond between residue %d and residue %d (current distance = %.2f)' % (residue1.index,residue2.index,disulfide_dist))
                                    modeller.topology.addBond(atom1, atom2)
                                    disulfide_bonds_added.append(disulfide_bond_id)

        print('disulfide bonds added')
        print(disulfide_bonds_added)
    elif add_disulfide_bonds and disulfide_bond_residues_idx is not None: #add disulfides at prespecified indices 
        disulfide_bonds_added = [] 
        for chain in modeller.topology.chains():
            for residue1 in chain.residues():
                for residue2 in chain.residues():
                    if residue1.name == 'CYS' and residue2.name == 'CYS' and residue1.index != residue2.index:
                        id1 = '%d_%d' % (residue1.index,residue2.index)
                        id2 = '%d_%d' % (residue2.index,residue1.index)
                        if id1 in disulfide_bond_residues_idx or id2 in disulfide_bond_residues_idx:
                            atom1 = next(atom for atom in residue1.atoms() if atom.name == 'SG')
                            atom2 = next(atom for atom in residue2.atoms() if atom.name == 'SG')
                            disulfide_dist = round(np.linalg.norm(pos[atom1.index] -  pos[atom2.index]),2)
                            disulfide_bond_id = '%d_%d' % (min(residue1.index,residue2.index),max(residue1.index,residue2.index))
                            if disulfide_dist <= 10.0:  
                                if disulfide_bond_id not in disulfide_bonds_added:
                                    print('adding disulfide bond between residue %d and residue %d (current distance = %.2f)' % (residue1.index,residue2.index,disulfide_dist))
                                    modeller.topology.addBond(atom1, atom2)
                                    disulfide_bonds_added.append(disulfide_bond_id)

                                    
                                  
        print('disulfide bonds added')
        print(disulfide_bonds_added)

    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(modeller.topology)
    integrator = LangevinIntegrator(0, 0.01, 0.0)
    if n_threads > 0:
        simulation = app.Simulation(modeller.topology, system, integrator, platform, {'Threads', str(n_threads)})
    else:
        simulation = app.Simulation(modeller.topology, system, integrator, platform)    
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    
    return simulation.topology, simulation.context.getState(getPositions=True).getPositions()



def cis_peptide_bond_fixer(topology, positions, platform, cis_residues_idx, bad_peptide_bond_residues_idx, k1=2.5, k2=2.5, n_threads=-1):

    modeller = app.Modeller(topology, positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(modeller.topology)

    force = CustomExternalForce("k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", 100 * spring_unit)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for residue in modeller.topology.residues():
        if residue.index not in bad_peptide_bond_residues_idx and residue.index not in cis_residues_idx: 
            for atom in residue.atoms():
                if atom.name in ["CA", "CB", "N", "C"]:
                    force.addParticle(atom.index, modeller.positions[atom.index])
    
    system.addForce(force)


    if len(bad_peptide_bond_residues_idx) > 0:
        peptide_bond_force = CustomExternalForce("k1 * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        peptide_bond_force.addGlobalParameter("k1", k1 * spring_unit)
        for p in ["x0", "y0", "z0"]:
            peptide_bond_force.addPerParticleParameter(p)

        for residue in modeller.topology.residues():
            if residue.index in bad_peptide_bond_residues_idx:
                for atom in residue.atoms():
                    if atom.name in ["N", "C"]:
                        peptide_bond_force.addParticle(atom.index, modeller.positions[atom.index])
        
        system.addForce(peptide_bond_force)


    if len(cis_residues_idx) > 0:
        cis_force = CustomTorsionForce("10*k2*(1+cos(theta))^2")
        cis_force.addGlobalParameter("k2", k2 * ENERGY)

        for chain in modeller.topology.chains():
            residues = [res for res in chain.residues()]
            relevant_atoms = [{atom.name:atom.index for atom in res.atoms() if atom.name in ["N", "CA", "C"]} for res in residues]
            for i in cis_residues_idx:
                resi = relevant_atoms[i-1]
                n_resi = relevant_atoms[i]
                cis_force.addTorsion(resi["CA"], resi["C"], n_resi["N"], n_resi["CA"])
        
        system.addForce(cis_force)

    integrator = LangevinIntegrator(0, 0.01, 0.0)
    if n_threads > 0:
        simulation = app.Simulation(modeller.topology, system, integrator, platform, {'Threads': str(n_threads)})
    else:
        simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    return simulation
    

def chirality_fixer(simulation, d_residues_idx):

    topology = simulation.topology
    positions = simulation.context.getState(getPositions=True).getPositions()
    
    d_stereoisomers = []
    for residue in topology.residues():
        if residue.index in d_residues_idx:
            #if it is a D-stereoisomer then flip its H atom
            indices = {x.name:x.index for x in residue.atoms() if x.name in ["HA", "CA"]}
            positions[indices["HA"]] = 2*positions[indices["CA"]] - positions[indices["HA"]]
            
            #fix the H atom in place
            particle_mass = simulation.system.getParticleMass(indices["HA"])
            simulation.system.setParticleMass(indices["HA"], 0.0)
            d_stereoisomers.append((indices["HA"], particle_mass))
            
    if len(d_stereoisomers) > 0:
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()
        for atom in d_stereoisomers:
            simulation.system.setParticleMass(*atom)
        simulation.minimizeEnergy()
    
    return simulation


def strained_sidechain_bonds_fixer(topology, positions, platform, strained_residues, n_threads=-1):

    #delete all atoms except the main chain for badly refined residues.
    bb_atoms = ["N","CA","C"]
    bad_side_chains = sum([[atom for atom in residue.atoms() if atom.name not in bb_atoms] for residue in strained_residues],[])
    modeller = app.Modeller(topology, positions)
    modeller.delete(bad_side_chains)
    
    #save model with deleted side chains to temporary file.
    random_number = str(int(np.random.rand()*10**8))
    tmp_file = f"side_chain_fix_tmp_{random_number}.pdb"
    with open(tmp_file,"w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
        
    #load model into pdbfixer
    fixer = pdbfixer.PDBFixer(tmp_file)
    os.remove(tmp_file)
    
    #repair deleted side chains 
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(modeller.topology)
    integrator = LangevinIntegrator(0, 0.01, 0.0)
    if n_threads > 0:
        simulation = app.Simulation(modeller.topology, system, integrator, platform, {'Threads', str(n_threads)})
    else:
        simulation = app.Simulation(modeller.topology, system, integrator, platform)    
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    
    return simulation.topology, simulation.context.getState(getPositions=True).getPositions()


def get_disulfide_bonds_info(topology, positions, disulfide_bond_cutoff=5.0, n_threads=-1):

    modeller = app.Modeller(topology, positions)
    pos = np.array(modeller.positions.value_in_unit(LENGTH))

    disulfide_bond_dist = {}
    disulfide_bond_residues = []  
    disulfide_bond_ids = [] 
 
    for chain in modeller.topology.chains():
        for residue1 in chain.residues():
            for residue2 in chain.residues():
                if residue1.name in ['CYS','CYX'] and residue2.name in ['CYS','CYX'] and residue1.index != residue2.index:

                    atom1 = next(atom for atom in residue1.atoms() if atom.name == 'SG')
                    atom2 = next(atom for atom in residue2.atoms() if atom.name == 'SG')
                    disulfide_dist = round(np.linalg.norm(pos[atom1.index] -  pos[atom2.index]),2)
                    disulfide_bond_id = '%d_%d' % (min(residue1.index,residue2.index),max(residue1.index,residue2.index))

                    if disulfide_dist >= 1.7 and disulfide_dist <= 2.3: 
                        if residue1 not in disulfide_bond_residues:
                            disulfide_bond_residues.append(residue1)
                        if residue2 not in disulfide_bond_residues:
                            disulfide_bond_residues.append(residue2)
                        if disulfide_bond_id not in disulfide_bond_ids:
                            disulfide_bond_ids.append(disulfide_bond_id)

                    disulfide_bond_dist[disulfide_bond_id] = disulfide_dist

    print('disulfide distances')
    print(disulfide_bond_dist)
                        
    return disulfide_bond_ids, disulfide_bond_residues

def get_peptide_bond_len_info(topology, positions):

    peptide_bond_len_info = {}
    pass_check = True
    for chain in topology.chains():
        residues = [{atom.name:atom.index for atom in res.atoms() if atom.name in ["N", "C"]} for res in chain.residues()]
        for i in range(len(residues)-1):
            key = 'residue%d' % i
            #for simplicity we only check the peptide bond length as the rest should be correct as they are hard coded 
            v = np.linalg.norm(positions[residues[i]["C"]] -  positions[residues[i+1]["N"]])
            if abs(v - 1.329*LENGTH) > 0.1*LENGTH:
                peptide_bond_len_info[key] = 'outside_tolerance'
                pass_check = False
            else:
                peptide_bond_len_info[key] = 'within_tolerance'
    return peptide_bond_len_info, pass_check


def cis_bond(p0,p1,p2,p3):

    ab = p1-p0
    cd = p2-p1
    db = p3-p2
    
    u = np.cross(-ab, cd)
    v = np.cross(db, cd)
    return np.dot(u,v) > 0
            

def get_cis_trans_bond_info(topology, positions):

    cis_trans_bond_info = {}
    pass_check = True
    pos = np.array(positions.value_in_unit(LENGTH))
    for chain in topology.chains():
        residues = [res for res in chain.residues()]
        relevant_atoms = [{atom.name:atom.index for atom in res.atoms() if atom.name in ["N", "CA", "C"]} for res in residues]
        for i in range(1,len(residues)):
            key = 'residue%d' % i
            if residues[i].name == "PRO":
                continue

            resi = relevant_atoms[i-1]
            n_resi = relevant_atoms[i]
            p0,p1,p2,p3 = pos[resi["CA"]],pos[resi["C"]],pos[n_resi["N"]],pos[n_resi["CA"]]
            if cis_bond(p0,p1,p2,p3):
                cis_trans_bond_info[key] = 'cis'
                pass_check = False
            else:
                cis_trans_bond_info[key] = 'trans'
    return cis_trans_bond_info, pass_check


def get_chirality_info(topology, positions):

    chirality_info = {}
    pass_check = True
    pos = np.array(positions.value_in_unit(LENGTH))
    for i,residue in enumerate(topology.residues()):
        key = 'residue%d' % i
        if residue.name == "GLY":
            continue

        atom_indices = {atom.name:atom.index for atom in residue.atoms() if atom.name in ["N", "CA", "C", "CB"]}
        vectors = [positions[atom_indices[i]] - positions[atom_indices["CA"]] for i in ["N", "C", "CB"]]

        val = (np.dot(np.cross(vectors[0], vectors[1]), vectors[2]))._value
        if val < 0:
            chirality_info[key] = 'D'
            pass_check = False
        else:
            chirality_info[key] = 'L' 
    return chirality_info, pass_check 


def get_clash_info(topology, positions):

    clash_info = {} 
    pass_check = True
    heavies = [x for x in topology.atoms() if x.element.symbol != "H"]
    pos = np.array(positions.value_in_unit(LENGTH))[[x.index for x in heavies]]

    tree = spatial.KDTree(pos)
    pairs = tree.query_pairs(r=max(cutoffs.values()))

    for pair in pairs:
        atom_i, atom_j = heavies[pair[0]], heavies[pair[1]]
        key = 'atom%d_atom%d' % (atom_i.residue.index, atom_j.residue.index)

        if atom_i.residue.index == atom_j.residue.index:
            continue
        elif (atom_i.name == "C" and atom_j.name == "N") or (atom_i.name == "N" and atom_j.name == "C"):
            continue

        atom_distance = np.linalg.norm(pos[pair[0]] - pos[pair[1]])
            
        if (atom_i.name == "SG" and atom_j.name == "SG") and atom_distance > 1.88:
            continue

        elif atom_distance < (cutoffs[atom_i.element.symbol + atom_j.element.symbol]):
            clash_info[key] = 'clash'
            pass_check = False
        else:
            clash_info[key] = 'no_clash'
    return clash_info, pass_check 


def get_strained_sidechain_bonds_info(topology, positions, disulfide_bond_residues):

    strained_bond_info = {} 
    pass_check = True 
    atoms = list(topology.atoms())
    pos = np.array(positions.value_in_unit(LENGTH))
    
    system = forcefield.createSystem(topology)
    bonds = [x for x in system.getForces() if type(x).__name__ == "HarmonicBondForce"][0]
    
    #initialise arrays for bond details
    n_bonds = bonds.getNumBonds()
    i = np.empty(n_bonds, dtype=int)
    j = np.empty(n_bonds, dtype=int)
    k = np.empty(n_bonds)
    x0 = np.empty(n_bonds)
    
    #extract bond details to arrays
    for n in range(n_bonds):
        i[n],j[n],_x0,_k = bonds.getBondParameters(n)
        k[n] = _k.value_in_unit(spring_unit)
        x0[n] = _x0.value_in_unit(LENGTH)
        
    #check if there are any abnormally strained bond
    distance = np.linalg.norm(pos[i] - pos[j], axis=-1)
    check = k*(distance - x0)**2 > 100
    
    #return residues with strained bonds if any
    strained_residues = [atoms[x].residue for x in i[check]]
    strained_residues = [residue for residue in strained_residues if residue not in disulfide_bond_residues] 
    for idx in range(0,n_bonds):
        key = 'atom%d_atom%d' % (i[idx],j[idx])
        if check[idx]:
            strained_bond_info[key] = 'strained'
            pass_check = False 
        else:
            strained_bond_info[key] = 'not_strained'
             
    return strained_residues, strained_bond_info, pass_check

