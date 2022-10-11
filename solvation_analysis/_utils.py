import numpy as np
from collections import defaultdict
from functools import reduce


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


def verify_solute_atoms_dict(solute_atoms_dict):
    # first we verify the input format
    atom_group_lengths = []
    for solute_name, solute_atom_group in solute_atoms_dict.items():
        assert isinstance(solute_name, str), (
            "The keys of solutes_dict must be strings."
        )
        assert isinstance(solute_atom_group, mda.AtomGroup), (
            f"The values of solutes_dict must be MDAnalysis.AtomGroups. But the value"
            f"for {solute_name} is a {type(solute_atom_group)}."
        )
        assert len(solute_atom_group) == len(solute_atom_group.residues), (
            "The solute_atom_group must have a single atom on each residue."
        )
        atom_group_lengths.append(len(solute_atom_group))
    assert np.all(np.array(atom_group_lengths) == atom_group_lengths[0]), (
        "AtomGroups in solutes_dict must have the same length because there should be"
        "one atom per solute molecule."
    )

    # verify that the solute_atom_groups have no overlap
    solute_atom_group = reduce(lambda x, y: x | y, [atoms for atoms in solute_atoms_dict.values()])
    assert solute_atom_group.n_atoms == sum([atoms.n_atoms for atoms in solute_atoms_dict.values()])

    return solute_atom_group
