# from abc import ABC


from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rdf import InterRDF
import numpy as np


class Solute:
    def __init__(self, solute, solvents, **kwargs):
        super(Solute, self).__init__(solute.universe.trajectory, **kwargs)
        self.solute = solute
        self.solvents = solvents

    def run(self):
        # run all RDFs and store their results
        rdfs = []
        for solvent in self.solvents:
            rdf = InterRDF(self.solute, solvent)
            rdf.run()
            rdfs.append(rdf)
        self.rdfs = rdfs
        self.ion_speciation = IonSpeciation(self.solute, self.solvents)
        self.ion_pairing = IonPairing(self.solute, self.solvents)
        self.coordination_numbers = CoordinationNumber(self.solute, self.solvents)


class IonSpeciation(AnalysisBase):
    def __init__(self, solute, solvent, **kwargs):
        super(IonSpeciation, self).__init__(solute.universe.trajectory, **kwargs)
        self._parameter = solvent
        self._ag = solute

    def _prepare(self):
        # OPTIONAL
        # Called before iteration on the trajectory has begun.
        # Data structures can be set up at this time
        self.result = []

    def _single_frame(self):
        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store result of `some_function` for a single frame
        self.result.append(some_function(self._ag, self._parameter))

    def _conclude(self):
        # OPTIONAL
        # Called once iteration on the trajectory is finished.
        # Apply normalisation and averaging to results here.
        self.result = np.asarray(self.result) / np.sum(self.result)


class IonPairing(AnalysisBase):
    def __init__(self, solute, solvent, **kwargs):
        super(IonPairing, self).__init__(solute.universe.trajectory,
                                         **kwargs)
        self._parameter = solvent
        self._ag = solute

    def _prepare(self):
        # OPTIONAL
        # Called before iteration on the trajectory has begun.
        # Data structures can be set up at this time
        self.result = []

    def _single_frame(self):
        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store result of `some_function` for a single frame
        self.result.append(some_function(self._ag, self._parameter))

    def _conclude(self):
        # OPTIONAL
        # Called once iteration on the trajectory is finished.
        # Apply normalisation and averaging to results here.
        self.result = np.asarray(self.result) / np.sum(self.result)



class CoordinationNumber(AnalysisBase):
    def __init__(self, solute, solvent, **kwargs):
        super(CoordinationNumber, self).__init__(solute.universe.trajectory,
                                                 **kwargs)
        self._parameter = solvent
        self._ag = solute

    def _prepare(self):
        # OPTIONAL
        # Called before iteration on the trajectory has begun.
        # Data structures can be set up at this time
        self.result = []

    def _single_frame(self):
        # REQUIRED
        # Called after the trajectory is moved onto each new frame.
        # store result of `some_function` for a single frame
        self.result.append(some_function(self._ag, self._parameter))

    def _conclude(self):
        # OPTIONAL
        # Called once iteration on the trajectory is finished.
        # Apply normalisation and averaging to results here.
        self.result = np.asarray(self.result) / np.sum(self.result)
