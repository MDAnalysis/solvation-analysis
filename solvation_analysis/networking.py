"""
================
Networking
================
:Author: Orion Cohen, Tingzheng Hou, Kara Fong
:Year: 2021
:Copyright: GNU Public License v3

Study the topology and structure of solute-solvent networks.

Networking yields a complete description of coordinated solute-solvent networks
in the solution, regardless of identify. This could include cation-anion networks
or hydrogen bond networks.

While ``networking`` can be used in isolation, it is meant to be used
as an attribute of the Solution class. This makes instantiating it and calculating the
solvation data a non-issue.
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from solvation_analysis.residence import Residence
from solvation_analysis._column_names import *


class Networking:
    """
    Calculate the number and size of solute-solvent networks.

    A network is defined as a bipartite graph of solutes and solvents, where edges
    are defined by coordination in the solvation_data DataFrame. A single solvent
    or multiple solvents can be selected, but coordination between solvents will
    not be included, only coordination between solutes and solvents.

    Networking uses the solvation_data to construct an adjacency matrix and then
    extracts the connected subgraphs within it. These connected subgraphs are stored
    in a DataFrame in Networking.network_df.

    Several other representations of the networking data are included in the attributes.

    Parameters
    ----------
    solvents : str or list[str]
        the solvents to include in the solute-solvent network.
    solvation_data : pandas.DataFrame
        a dataframe of solvation data with columns FRAME, "solvated_atom", "atom_ix",
        "dist", "res_name", and "res_ix".
    solute_res_ix : np.ndarray
        the residue indices of the solutes in solvation_data
    res_name_map : pd.Series
        a mapping between residue indices and the solute & solvent names in a Solution.

    Attributes
    ----------
    network_df : pd.DataFrame
        the dataframe containing all networking data. the indices are the frame and
        network index, respectively. the columns are the res_name and res_ix.
    network_sizes : pd.DataFrame
        a dataframe of network sizes. the index is the frame. the column headers
        are network sizes, or the number of solutes + solvents in the network, so
        the columns might be [2, 3, 4, ...]. the values in each column are the
        number of networks with that size in each frame.
    solute_status : dict of {str: float}
        a dictionary where the keys are the "status" of the solute and the values
        are the fraction of solute with that status, averaged over all frames.
        "alone" means that the solute not coordinated with any of the networking
        solvents, network size is 1.
        "paired" means the solute and is coordinated with a single networking
        solvent and that solvent is not coordinated to any other solutes, network
        size is 2.
        "in_network" means that the solute is coordinated to more than one solvent
        or its solvent is coordinated to more than one solute, network size >= 3.
    solute_status_by_frame : pd.DataFrame
        as described above, except organized into a dataframe where each
        row is a unique frame and the columns are "alone", "paired", and "in_network".

    Examples
    --------
     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> networking = Networking.from_solution(solution, 'PF6')
    """

    def __init__(self, solvents, solvation_data, solute_res_ix, res_name_map):
        self.solvents = solvents
        self.solvation_data = solvation_data
        solvent_present = np.isin(self.solvents, self.solvation_data['res_name'].unique())
        if not solvent_present.all():
            raise Exception(f"Solvent(s) {np.array(self.solvents)[~solvent_present]} not found in solvation data.")
        self.solute_res_ix = solute_res_ix
        self.res_name_map = res_name_map
        self.n_solute = len(solute_res_ix)
        self.network_df = self._generate_networks()
        self.network_sizes = self._calculate_network_sizes()
        self.solute_status, self.solute_status_by_frame = self._calculate_solute_status()
        self.solute_status = self.solute_status.to_dict()

    @staticmethod
    def from_solution(solution, solvents):
        """
        Generate a Networking object from a solution and solvent names.

        Parameters
        ----------
        solution : Solution
        solvents : str or list of str
            the strings should be the name of solvents in the Solution. The
            strings must match exactly for Networking to work properly. The
            selected solvents will be used to construct the networking graph
            that is described in documentation for the Networking class.

        Returns
        -------
        Networking
        """
        return Networking(
            solvents,
            solution.solvation_data,
            solution.solute_res_ix,
            solution.res_name_map,
        )

    @staticmethod
    def _unwrap_adjacency_dataframe(df):
        # this class will transform the biadjacency matrix into a proper adjacency matrix
        connections = df.reset_index(level=0).drop(columns=FRAME)
        idx = connections.columns.append(connections.index)
        directed = connections.reindex(index=idx, columns=idx, fill_value=0)
        undirected = directed.values + directed.values.T
        adjacency_matrix = csr_matrix(undirected)
        return adjacency_matrix

    def _generate_networks(self):
        """
        This function generates a dataframe containing all the solute-solvent networks
        in every frame of the simulation. The rough approach is as follows:

        1. transform the solvation_data DataFrame into an adjacency matrix
        2. determine the connected subgraphs in the adjacency matrix
        3. tabulate the res_ix involved in each network and store in a DataFrame
        """
        solvents = [self.solvents] if isinstance(self.solvents, str) else self.solvents
        solvation_subset = self.solvation_data[np.isin(self.solvation_data.res_name, solvents)]
        # reindex solvated_atom to residue indexes
        reindexed_subset = solvation_subset.reset_index(level=1)
        reindexed_subset.solvated_atom = self.solute_res_ix[reindexed_subset.solvated_atom].values
        dropped_reindexed = reindexed_subset.set_index(['solvated_atom'], append=True)
        reindexed_subset = dropped_reindexed.reorder_levels([FRAME, 'solvated_atom', 'atom_ix'])
        # create adjacency matrix from reindexed df
        graph = Residence.calculate_adjacency_dataframe(reindexed_subset)
        network_arrays = []
        # loop through each time step / frame
        for frame, df in graph.groupby(FRAME):
            # drop empty columns
            df = df.loc[:, (df != 0).any(axis=0)]
            # save map from local index to residue index
            solute_map = df.index.get_level_values(1).values
            solvent_map = df.columns.values
            ix_to_res_ix = np.concatenate([solvent_map, solute_map])
            adjacency_df = Networking._unwrap_adjacency_dataframe(df)
            _, network = connected_components(
                csgraph=adjacency_df,
                directed=False,
                return_labels=True
            )
            network_array = np.vstack([
                np.full(len(network), frame),  # frame
                network,  # network
                self.res_name_map[ix_to_res_ix],  # res_names
                ix_to_res_ix,  # res index
            ]).T
            network_arrays.append(network_array)
        # create and return network dataframe
        all_clusters = np.concatenate(network_arrays)
        cluster_df = (
            pd.DataFrame(all_clusters, columns=[FRAME, 'network', 'res_name', 'res_ix'])
                .set_index([FRAME, 'network'])
                .sort_values([FRAME, 'network'])
        )
        return cluster_df

    def _calculate_network_sizes(self):
        # This utility calculates the network sizes and returns a convenient dataframe.
        cluster_df = self.network_df
        cluster_sizes = cluster_df.groupby([FRAME, 'network']).count()
        size_counts = cluster_sizes.groupby([FRAME, 'res_name']).count().unstack(fill_value=0)
        size_counts.columns = size_counts.columns.droplevel()
        return size_counts

    def _calculate_solute_status(self):
        """
        This utility calculates the percentage of each solute with a given "status".
        Namely, whether the solvent is "alone", "paired" (with a single solvent), or
        "in_network" of > 2 species.
        """
        status = self.network_sizes.rename(columns={2: 'paired'})
        status['in_network'] = status.iloc[:, 1:].sum(axis=1).astype(int)
        status['alone'] = self.n_solute - status.loc[:, ['paired', 'in_network']].sum(axis=1)
        status = status.loc[:, ['alone', 'paired', 'in_network']]
        solute_status_by_frame = status / self.n_solute
        solute_status = solute_status_by_frame.mean()
        return solute_status, solute_status_by_frame

    def get_network_res_ix(self, network_index, frame):
        """
        Return the indexes of all residues in a selected network.

        The network_index and frame must be provided to fully specify the network.
        Once the indexes are returned, they can be used to select an AtomGroup with
        the species of interest, see Examples.

        Parameters
        ----------
        network_index : int
            The index of the network of interest
        frame : int
            the frame in the trajectory to perform selection at. Defaults to the
            current trajectory frame.
        Returns
        -------
        res_ix : np.ndarray

        Examples
        --------
         .. code-block:: python

            # first define Li, BN, and FEC AtomGroups
            >>> solution = Solution(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
            >>> networking = Networking.from_solution(solution, 'PF6')
            >>> res_ix = networking.get_network_res_ix(1, 5)
            >>> solution.u.residues[res_ix].atoms
            <AtomGroup with 126 Atoms>

        """
        res_ix = self.network_df.loc[pd.IndexSlice[frame, network_index], 'res_ix'].values
        return res_ix.astype(int)
