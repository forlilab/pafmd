import argparse
import re 
import sys
import json

import parmed as pmd
from openmm import *
from openmm.app import *
from openmm.unit import *

import pdbfixer

from mdtraj.reporters import NetCDFReporter


def get_system_charge(system):
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            charges = [force.getParticleParameters(i)[0].value_in_unit(elementary_charge) 
                      for i in range(force.getNumParticles())]
            return sum(charges)
    return None

def fixpdb(pdbfile: str, keep_heterogens: bool=False):
    """Fixes common problems in PDB such as:
            - missing atoms
            - missing residues
            - missing hydrogens
            - remove nonstandard residues

    :param pdbfile: pdb string old format
    :type pdbfile: str
    :param pdbxfile: pdb string new format
    :type pdbxfile: str
    :param keep_heterogens: if False all heterogen atoms but waters are deleted, defaults to False
    :type keep_heterogens: bool, optional
    :return: new topology and positions
    :rtype: tuple[Topology, list]
    """
    fixer = pdbfixer.PDBFixer(filename=pdbfile)
    fixer.findMissingResidues()
    
    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=True)

    fixer.findMissingAtoms() 
    fixer.addMissingAtoms()
    return fixer.topology, fixer.positions

def run_sim(input_receptor_path, fix_pdb, water_model, output_dir, heating_steps, equil_steps, production_steps, solvation_padding_constant=1.2, Tstart=0, Tend=300, Tstep=30, save_checkpoint=False):

    os.makedirs(output_dir, exist_ok=True)

    print('Selecting simulation platform')
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('CudaPrecision', 'mixed')
        platform.setPropertyDefaultValue('CudaDeviceIndex', '0')
        print('Using GPU:CUDA')
    except: 
        try:
            platform = openmm.Platform.getPlatformByName("OpenCL")
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
            platform.setPropertyDefaultValue('Precision', 'mixed')
            print('Using GPU:OpenCL')
        except:
            platform = openmm.Platform.getPlatformByName("CPU")
            print("Switching to CPU, no GPU available.")

    if water_model == 'tip3p':
        water_ff = 'amber14/tip3p.xml'
    elif water_model == 'opc':
        water_ff = './opc.xml' 

    if water_model == 'tip3p':
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    elif water_model == 'opc':
        forcefield = ForceField('amber14-all.xml', './opc.xml')

    if not(fix_pdb):
        # Load the PDB file
        print('loading pdb %s' % input_receptor_path)
        pdb = PDBFile(input_receptor_path)
        modeller = Modeller(pdb.topology, pdb.positions)
    else:
        print('fixing pdb')
        pdb_topology, pdb_positions = fixpdb(input_receptor_path)
        modeller = Modeller(pdb_topology, pdb_positions)

    #we are assuming hydrogens have already been added via pre_md_refine script 
        
    prepared_system_fname = '%s/prepared_system_no_solvent.pdb' % output_dir 
    print('writing %s' % prepared_system_fname)
    with open(prepared_system_fname, 'w') as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

    residue_templates = {}

    for chain in modeller.topology.chains():
        for residue1 in chain.residues():
            for residue2 in chain.residues():
                if (residue1.name == 'CYS' and residue2.name == 'CYS') and (residue1.index != residue2.index):
                    atom1 = next(atom for atom in residue1.atoms() if atom.name == 'SG')
                    atom2 = next(atom for atom in residue2.atoms() if atom.name == 'SG')
                    for bond in modeller.topology.bonds():
                        if (bond.atom1 == atom1 and bond.atom2 == atom2) or (bond.atom1 == atom2 and bond.atom2 == atom1):
                            print('disulfide detected between:')
                            print(residue1)
                            print(residue2)
                            residue_templates[residue1] = 'CYX'         
                            residue_templates[residue2] = 'CYX' 

    print(residue_templates) 
    
    print('Adding solvent')
    modeller.addSolvent(forcefield, model=water_model, padding=solvation_padding_constant*nanometer, neutralize=True, ionicStrength=0.15*molar, residueTemplates=residue_templates) 
    print('System has %d atoms' % modeller.topology.getNumAtoms())
    solvated_system_fname = '%s/solvated_system.pdb' % output_dir 
    print('writing %s' % solvated_system_fname)
    with open(solvated_system_fname, 'w') as outfile:
        PDBFile.writeFile(modeller.topology, modeller.positions, outfile)

    residue_templates = {}

    for chain in modeller.topology.chains():
        for residue1 in chain.residues():
            if residue1.name == 'CYS':
                for residue2 in chain.residues():
                    if residue2.name == 'CYS' and residue1.index != residue2.index:
                        atom1 = next(atom for atom in residue1.atoms() if atom.name == 'SG')
                        atom2 = next(atom for atom in residue2.atoms() if atom.name == 'SG')
                        for bond in modeller.topology.bonds():
                            if (bond.atom1 == atom1 and bond.atom2 == atom2) or (bond.atom1 == atom2 and bond.atom2 == atom1):
                                residue_templates[residue1] = 'CYX'         
                                residue_templates[residue2] = 'CYX' 


    print('Creating the system') 
    # Create the system
    system = forcefield.createSystem(modeller.topology, 
                                     nonbondedMethod=PME, 
                                     nonbondedCutoff=10*angstroms, 
                                     constraints=HBonds, #SHAKE
                                     rigidWater=True, #SETTLE
                                     ignoreExternalBonds=True,
                                     residueTemplates=residue_templates
                                     )

    system_wo_constraints = forcefield.createSystem(modeller.topology, 
                                     nonbondedMethod=PME, 
                                     nonbondedCutoff=10*angstroms, 
                                     rigidWater=False,
                                     ignoreExternalBonds=True,
                                     residueTemplates=residue_templates
                                     )

    unique_residues = set()
    for residue in modeller.topology.residues():
        unique_residues.add(residue.name)

    print("UNIQUE RESIDUES IN SYSTEM:")
    print(unique_residues)

    net_charge = get_system_charge(system)
    print(f"NET CHARGE OF SYSTEM: {net_charge}")
    if net_charge > 1e-3 or net_charge < -1e-3:
        print("ERROR: system is not neutral")
        sys.exit()

    # Create the integrator
    integrator = LangevinMiddleIntegrator(Tstart*kelvin, 1/picosecond, 0.002*picoseconds)

    # Create the simulation object
    simulation = Simulation(modeller.topology, system, integrator, platform)

    # Set the positions
    simulation.context.setPositions(modeller.positions)

    pdb_positions = simulation.context.getState(getPositions=True).getPositions()

    #roughly following protocol of: https://pubs.acs.org/doi/epdf/10.1021/acs.jctc.2c01189?ref=article_openPDF

    AA=['ALA','ASP','CYS','GLU','PHE','GLY','HIS','HID','HIE','HIP','ILE','LYS','LEU','MET','ARG','PRO','GLN','ASN','SER','THR','VAL','TRP','TYR','CYX','LYN','ASH','GLH'] 
    restraint_force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint_force.addGlobalParameter("k",10.*kilocalories_per_mole/angstroms**2)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')

    #apply restrained potential of 10kcal/mol to all protein atoms 
    for i, res in enumerate(modeller.topology.residues()):
        if res.name in AA:                              
            for atom in res.atoms():
                restraint_force.addParticle(atom.index,pdb_positions[atom.index].value_in_unit(nanometers))

    restraint_force_index = system.addForce(restraint_force)
    simulation.context.reinitialize(preserveState=True)

    print('RUNNING RESTRAINED ENERGY MINIMIZATION ROUND 1')
    simulation.minimizeEnergy(maxIterations=200)

    print('RUNNING UNRESTRAINED ENERGY MINIMIZATION ROUND 2')
    system.removeForce(restraint_force_index)
    simulation.context.reinitialize(preserveState=True)
    simulation.minimizeEnergy(maxIterations=500)

    pdb_positions = simulation.context.getState(getPositions=True).getPositions()
    output_fname = '%s/minimization_round2.pdb' % output_dir 
    print('SAVING %s' % output_fname)
    PDBFile.writeFile(simulation.topology, pdb_positions, open(output_fname, 'w'))

    structure = pmd.openmm.load_topology(modeller.topology, system_wo_constraints, pdb_positions)
    output_fname = '%s/minimization_round2.prmtop' % output_dir 
    print('SAVING %s' % output_fname)    
    structure.save(output_fname, overwrite=True)
    output_fname = '%s/minimization_round2.inpcrd' % output_dir 
    structure.save(output_fname, overwrite=True)

    print("RUNNING NVT EQUILIBRIATION")

    restraint_force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint_force.addGlobalParameter("k",10.*kilocalories_per_mole/angstroms**2)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')

    for i, res in enumerate(modeller.topology.residues()):
        if res.name in AA:                              
            for atom in res.atoms():
                if not re.search(r'H',atom.name): #all heavy atoms - no hydrogens
                    restraint_force.addParticle(atom.index,pdb_positions[atom.index].value_in_unit(nanometers))

    restraint_force_index = system.addForce(restraint_force)
    simulation.context.reinitialize(preserveState=True)

    nT = int((Tend - Tstart) / Tstep)+1
    for i in range(nT):
        temperature = Tstart + i * Tstep
        integrator.setTemperature(temperature)
        print(f"Temperature set to {temperature} K.")
        simulation.step(heating_steps) 

    system.removeForce(restraint_force_index)
    simulation.context.reinitialize(preserveState=True)

    print("RUNNING NPT EQUILIBRIATION")   
     
    # Add barostat for NPT equilibration
    barostat = MonteCarloBarostat(1*atmosphere, 300*kelvin)
    barostat_index = system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)

    # Equilibration step of 1 ns in NPT
    simulation.reporters.append(StateDataReporter(sys.stdout, 5000, step=True, potentialEnergy=True, temperature=True))
    simulation.step(equil_steps)   

    total_steps = heating_steps*nT + equil_steps + production_steps
    print('Adding reporters to the simulation')
    #every 0.1ns
    output_fname = '%s/statistics.csv' % output_dir
    simulation.reporters.append(StateDataReporter(output_fname, 50000, step=True, time=True,
                                                totalEnergy=True, potentialEnergy=True, kineticEnergy=True, 
                                                temperature=True, volume=True, density=True,
                                                progress=True, remainingTime=True, speed=True, totalSteps=total_steps))
    #every 0.1ns
    simulation.reporters.append(StateDataReporter(sys.stdout, 50000, step=True, time=True,
                                                totalEnergy=True, potentialEnergy=True, kineticEnergy=True, 
                                                temperature=True, volume=True, density=True,
                                                progress=True, remainingTime=True, speed=True, totalSteps=total_steps, separator='\t'))

    target_frames = 500 
    save_freq = int(production_steps/target_frames)
    print('saving frames every %d steps' % save_freq)

    save_freq_fname = '%s/save_freq.txt' % output_dir
    with open(save_freq_fname, 'w') as f:
        f.write(str(save_freq))

    output_fname = '%s/trajectory.nc' % output_dir
    simulation.reporters.append(NetCDFReporter(output_fname,
                                            reportInterval=save_freq)) 

    if save_checkpoint:
        output_fname = '%s/simulation.chk' % output_dir
        simulation.reporters.append(CheckpointReporter(output_fname, save_freq)) 


    simulation.step(production_steps)  



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( 
        "--input_receptor_path", type=str, default=None
    )
    parser.add_argument(
        "--water_model", type=str, default='tip3p'
    )
    parser.add_argument(
        "--fix_pdb", action='store_true', default=False
    )
    parser.add_argument(
        "--output_dir", type=str, default=None 
    )
    parser.add_argument(
        "--production_steps", type=int, default=125000000, help='default corresponds to 125000000 (250 ns)'
    )

    args = parser.parse_args()

    if args.water_model not in ['tip3p', 'opc']:
        raise ValueError("water model must be one of tip3p or opc")

    heating_steps = 50000 #50 ps per temp
    equil_steps = 500000 #1 ns
 
    run_sim(args.input_receptor_path, args.fix_pdb, args.water_model, args.output_dir, heating_steps, equil_steps, args.production_steps)

