# Data Generation Process

This README describes the data-generating process for the test data used in this package.

Molecular dynamics runs were performed on a simulated Li-ion battery electrolyte composed 
of 363 Butryro-Nitride (BN), 49 Ethylene Carbonate (EC), and 49 Lithium Hexafluorophosphate (LiPF6<sub>6</sub>).
The energy was minimized in PACKMOL and the trajectory was generated with LAMMPS. There is a 5 ns equilibration
period followed by a 5 ns production run.

## Trajectory Files

`bn_fec.data` was generated with the Pymatgen Python package using OPLS parameters downloaded
from LigParGen.
`bn_fec_short_wrap.dcd` is the wrapped trajectory file from LAMMPS
`bn_fec_short_unwrap.dcd` is the unwrapped trajectory file from LAMMPS
`bn_fec_elements.csv` is a csv file with the element name of every atom name in the 
trajectory file. It is used to add names to the Universe.

## Radial Distribution Functions

The radial distribution functions were generated with MDAnalysis.rdf.interRDF(). 

The `rdf_vs_li_easy` and `rdf_vs_li_hard` directories contain RDFs of various molecular
species against the Li ions, most of these are fairly well-defined RDFs. The `easy` RDFs are
generated from 500 frames of the trajectory while the `hard` RDFs are generated from
only 50 frames, so they are noisier.

The `rdf_non_solvated` directory contains RDFs of non-Li molecular species against eachother.
These RDFs have no clear structure and should not register a solvation shell. These are added
to provide a negative test of the solvation shell identification kernel.
