# Data Generation Process

This README describes the data-generating process for the test data used in this package.

## Trajectory Files

### bn_fec_data

Molecular dynamics runs were performed on a simulated Li-ion battery electrolyte composed 
of 363 Butryro-Nitride (BN), 49 Ethylene Carbonate (EC), and 49 Lithium Hexafluorophosphate (LiPF6<sub>6</sub>).
The energy was minimized in PACKMOL and the trajectory was generated with LAMMPS. There is a 5 ns equilibration
period followed by a 5 ns production run.

`bn_fec.data` was generated with the Pymatgen Python package. OPLS parameters for BN were downloaded
from LigParGen and parameters for FEC were provided by Tingzheng Hou, see https://doi.org/10.1016/j.nanoen.2019.103881.
`bn_fec_short_wrap.dcd` is the wrapped trajectory file from LAMMPS
`bn_fec_short_unwrap.dcd` is the unwrapped trajectory file from LAMMPS
`bn_fec_elements.csv` is a csv file with the element name of every atom name in the 
trajectory file. It is used to add names to the Universe.
`bn_solv_df_large.csv` is a csv file of the solvation_data over a longer simulation, 500 frames

### ea_fec_data

Molecular dynamics runs were performed on a simulated Li-ion battery electrolyte composed 
of Ethyl Acetate (EA), Fluorinated Ethylene Carbonate (FEC), and Lithium Hexafluorophosphate (LiPF6<sub>6</sub>).
The energy was minimized in PACKMOL and the trajectory was generated with OpenMM. There is a 5 ns equilibration
period followed by a 5 ns production run.

`ea_fec.dcd` is an abbreviated dcd file of the trajectory, 10 frames long
`ea_fec.pdb` contains the topology of the trajectory

### eax_data

Molecular dynamics runs were performed on a simulated Li-ion battery electrolyte composed 
of Fluorinated Ethylene Carbonate (FEC), Lithium Hexafluorophosphate (LiPF6<sub>6</sub>), and one of four 
fluorinated Ethyl Acetate species (EA, EAf, fEA, and fEAf, abbreviated in general as EAx).
The energy was minimized in PACKMOL and the trajectory was generated with OpenMM. There is a 5 ns equilibration
period followed by a 5 ns production run.

Each simulation has a `dcd` file for the trajectory and a `pdb` file for the topology.


## Radial Distribution Functions

The radial distribution functions were generated with MDAnalysis.rdf.interRDF(). 

The `rdf_vs_li_easy` and `rdf_vs_li_hard` directories contain RDFs of various molecular
species against the Li ions, most of these are fairly well-defined RDFs. The `easy` RDFs are
generated from 500 frames of the trajectory while the `hard` RDFs are generated from
only 50 frames, so they are noisier.

The `rdf_non_solvated` directory contains RDFs of non-Li molecular species against eachother.
These RDFs have no clear structure and should not register a solvation shell. These are added
to provide a negative test of the solvation shell identification kernel.
