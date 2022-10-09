import numpy as np
from collections import defaultdict


def verify_solute_atoms(solute_atoms):
    # we presume that the solute_atoms has the same number of atoms on each residue
    # and that they all have the same indices on those residues
    # and that the residues are all the same length
    # then this should work
    all_res_len = np.array([res.atoms.n_atoms for res in solute_atoms.residues])
    assert np.all(all_res_len[0] == all_res_len), (
        "All residues must be the same length."
    )
    res_atom_local_ix = defaultdict(list)
    res_atom_ix = defaultdict(list)

    for atom in solute_atoms.atoms:
        res_atom_local_ix[atom.resindex].append(atom.ix - atom.residue.atoms[0].ix)
        res_atom_ix[atom.resindex].append(atom.index)
    res_occupancy = np.array([len(ix) for ix in res_atom_local_ix.values()])
    assert np.all(res_occupancy[0] == res_occupancy), (
        "All residues must have the same number of solute_atoms atoms on them."
    )

    res_atom_array = np.array(list(res_atom_local_ix.values()))
    assert np.all(res_atom_array[0] == res_atom_array), (
        "All residues must have the same solute_atoms atoms on them."
    )

    res_atom_ix_array = np.array(list(res_atom_ix.values()))
    return res_atom_ix_array
