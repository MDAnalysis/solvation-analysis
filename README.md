SolvationAnalysis
==============================
[//]: # (Badges)

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://www.numfocus.org/)
[![Powered by MDAnalysis](https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA)](https://www.mdanalysis.org)
[![GitHub Actions Status](https://github.com/MDAnalysis/solvation-analysis/workflows/CI/badge.svg)](https://github.com/MDAnalysis/solvation-analysis/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/MDAnalysis/solvation-analysis//branch/main/graph/badge.svg)](https://codecov.io/gh/MDAnalysis/solvation-analysis//branch/main)
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://solvation-analysis.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/371804402.svg)](https://zenodo.org/badge/latestdoi/371804402)


---

The macroscopic behavior of a liquid is determined by its microscopic structure. 
For ionic systems, like batteries and many enzymes, the solvation environment 
surrounding ions is especially important. By studying the solvation of interesting 
materials, scientists can better understand, engineer, and design new technologies. 

Solvation analysis implements a 
[robust, cohesive, and fast set of methods] 
for analyzing the solvation structure of a liquid. It seamlessly integrates with
[MDAnalysis], making use of the core AtomGroup
and Universe data structures to parse solvation information. If you are interested
in understanding the solvation structure of a liquid, this package is for you!

Main development by @orioncohen, with mentorship from @richardjgowers, @IAlibay, and 
@hmacdope.

Find the documentation on [readthedocs].

---

#### Acknowledgements

[Google Summer of Code] and the [MDAnalysis] team provided funding and support for this project.

Tingzheng Hou (@htz1992213) contributed invaluable scientific guidance and mentorship.

Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.



[readthedocs]: (https://solvation-analysis.readthedocs.io/en/latest/)
[robust, cohesive, and fast set of methods]:(https://summerofcode.withgoogle.com/projects/#6227159028334592)
[Google Summer of Code]: https://summerofcode.withgoogle.com/
[MDAnalysis]: https://www.mdanalysis.org/
