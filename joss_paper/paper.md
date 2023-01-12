---
title: 'SolvationAnalysis: A Python toolkit for understanding liquid solvation structure in classical molecular dynamics simulations'
tags:
 - Python
 - chemistry
 - molecular dynamics
 - solvation structure
authors:
 - name: Orion Archer Cohen
   corresponding: true
   orcid: 0000-0003-3940-2456
   affiliation: 1
 - name: Hugo Macdermott-Opeskin
   orcid: 0000-0002-7393-7457
   affiliation: 2
 - name: Lauren Lee
   orcid:
   affiliation: 1
 - name: Tingzheng Hou
   orcid: 0000-0002-7163-2561
   affiliation: 3
 - name: Kara Fong
   orcid: 0000-0002-0711-097X
   affiliation: 4
 - name: Ryan Kingsbury
   orcid: 0000-0002-7168-3967
   affiliation: 1
 - name: Jingyang Wang
   orcid:
   affiliation: 1
 - name: Kristin A. Persson
   orcid: 0000-0003-2495-5509
   affiliation: "1,6"
affiliations:
 - name: Materials Science Division, Lawrence Berkeley National Laboratory, USA
   index: 1
 - name: Australian National University, Australia
   index: 2
 - name: Tsinghua Shenzhen International Graduate School, China
   index: 3
 - name: Trinity College, University of Cambridge, UK
   index: 4
 - name: Department of Materials Science, University of California, USA
   index: 5
 - name: Molecular Foundry, Lawrence Berkeley National Laboratory, USA
   index: 6
date: 12 January 2023
bibliography: paper.bib
---

# Summary

The macroscopic behavior of matter is determined by the microscopic
arrangement of atoms, but this structure is often
difficult or impossible to observe experimentally. Instead, researchers use
simulation techniques like molecular dynamics to probe the microscopic
structure and dynamics of everything from proteins to battery electrolytes.
SolvationAnalysis extracts solvation information from completed
molecular dynamics simulations, this lets research access key solvation
structure statistics with minimal effort, accelerating scientific research.

# Statement of need

Molecular dynamics studies of liquid solvation structures often replicate
established analyses on novel systems. In electrolyte systems, it is common
to calculate coordination numbers, radial distribution functions, solute
dissociation, cluster speciation, etc.[Hou:2019] In principle, these analyses are highly
similar across a diversity of systems. In practice, many specialized bespoke
tools have sprung up to address the same underlying problem. Enter `SolvationAnalysis`, 
an easy-to-use Python package with an interactive interface for
computing a wide variety of solvation properties. Building on `MDAnalysis` and
`pandas` [@Michaud-Agrawal:2014][@Gowers:2016][@pandas:2020], it efficiently
processes output from a wide variety of Molecular Dynamics applications.

`SolvationAnalysis` was designed to free researchers from laboriously
implementing common calculations. In addition to routine properties like
coordination numbers, solute-solvent pairing, and solute speciation,
SolvationAnalysis uses tools from the SciPy ecosystem [@numpy:2020][@scipy:2020]
to implement analyses of network formation [@Wang:2022] and residence
times [@Self:2019]. Since researchers will inevitably plot these
properties the package includes a robust set of visualization tools built
on top of `Matplotlib` and `Plotly` [@matplotlib:2007][@plotly:2015].
A full set of tutorials based on state-of-the-art electrolyte systems
[@Hou:2019][@Dong-Joo:2022] are also included to familiarize new researchers
with analyzing solvation structures. Together, these features allow for
rapid interactive or programmatic calculation of solvation properties.

# Figures

![A visual summary of SolvationAnalysis capabilities. \label{fig:summary}](summary_figure.jpg)

# Acknowledgements

Thank you to Oliver Beckstein, Richard Gowers, Irfan Alibay, and Lily Wang for
technical advice about MDAnalysis and Python development. Thank you to Google 
Summer of Code, the NSF GRFP Fellowship, and the US Department of Energy for 
funding.

# References