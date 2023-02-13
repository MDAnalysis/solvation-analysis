import MDAnalysis as mda
from MDAnalysis import transformations
import pathlib
import numpy as np
from solvation_analysis.tests.datafiles import eax_data

boxes = {
    'ea': [45.760393, 45.760393, 45.760393, 90, 90, 90],
    'eaf': [47.844380, 47.844380, 47.844380, 90, 90, 90],
    'fea': [48.358954, 48.358954, 48.358954, 90, 90, 90],
    'feaf': [50.023129, 50.023129, 50.023129, 90, 90, 90],
}
us = {}
for solvent_dir in pathlib.Path(eax_data).iterdir():
    u_solv = mda.Universe(
        str(solvent_dir / 'topology.pdb'),
        str(solvent_dir / 'trajectory_equil.dcd')
    )
    # our dcd lacks dimensions so we must manually set them
    box = boxes[solvent_dir.stem]
    set_dim = transformations.boxdimensions.set_dimensions(box)
    u_solv.trajectory.add_transformations(set_dim)
    us[solvent_dir.stem] = u_solv

atom_groups_dict = {}
for name, u in us.items():
    atom_groups = {}
    atom_groups['li'] = u.atoms.select_atoms("element Li")
    atom_groups['pf6'] = u.atoms.select_atoms("byres element P")
    residue_lengths = np.array([len(elements) for elements in u.residues.elements])
    eax_fec_cutoff = np.unique(residue_lengths, return_index=True)[1][2]
    atom_groups[name] = u.atoms.select_atoms(f"resid 1:{eax_fec_cutoff}")
    atom_groups['fec'] = u.atoms.select_atoms(f"resid {eax_fec_cutoff + 1}:600")
    atom_groups_dict[name] = atom_groups

solutes = {}
for name, atom_groups in u_eax_atom_groups.items():
    solute = Solute.from_atoms(
        atom_groups['li'],
        {'pf6': atom_groups['pf6'], name: atom_groups[name], 'fec': atom_groups['fec']},
    )
    solute.run()
    solutes[name] = solute