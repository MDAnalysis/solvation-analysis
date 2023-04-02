"""
================
Networking
================
:Author: Orion Cohen, Tingzheng Hou, Kara Fong
:Year: 2021
:Copyright: GNU Public License v3

Study the topology and structure of solute-solvent networks.

Networking yields a complete description of coordinated solute-solvent networks
in the solute, regardless of identify. This could include cation-anion networks
or hydrogen bond networks.

While ``networking`` can be used in isolation, it is meant to be used
as an attribute of the Solute class. This makes instantiating it and calculating the
solvation data a non-issue.
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from solvation_analysis._utils import calculate_adjacency_dataframe
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
        a dataframe of solvation data with columns "frame", "solute_atom", "solvent_atom",
        "distance", "solvent_name", and "solvent".
    solute_res_ix : np.ndarray
        the residue indices of the solutes in solvation_data
    res_name_map : pd.Series
        a mapping between residue indices and the solute & solvent names in a Solute.

    Examples
    --------
     .. code-block:: python

        # first define Li, BN, and FEC AtomGroups
        >>> solute = Solute.from_atoms(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
        >>> networking = Networking.from_solute(solute, 'PF6')
    """

    def __init__(self, solvents, solvation_data, solute_res_ix, res_name_map):
        self.solvents = solvents
        self.solvation_data = solvation_data
        solvent_present = np.isin(self.solvents, self.solvation_data[SOLVENT].unique())
        # TODO: we need all analysis classes to run when there is no solvation_data
        # if not solvent_present.all():
        #     raise Exception(f"Solvent(s) {np.array(self.solvents)[~solvent_present]} not found in solvation data.")
        self.solute_res_ix = solute_res_ix
        self.res_name_map = res_name_map
        self.n_solute = len(solute_res_ix)
        self._network_df = self._generate_networks()
        self._network_sizes = self._calculate_network_sizes()
        self._solute_status, self._solute_status_by_frame = self._calculate_solute_status()
        self._solute_status = self._solute_status.to_dict()

    @staticmethod
    def from_solute(solute, solvents):
        """
        Generate a Networking object from a solute and solvent names.

        Parameters
        ----------
        solute : Solute
        solvents : str or list of str
            the strings should be the name of solvents in the Solute. The
            strings must match exactly for Networking to work properly. The
            selected solvents will be used to construct the networking graph
            that is described in documentation for the Networking class.

        Returns
        -------
        Networking
        """
        return Networking(
            solvents,
            solute.solvation_data,
            solute.solute_res_ix,
            solute.res_name_map,
        )

    @staticmethod
    def _unwrap_adjacency_dataframe(df):
        # this class will transform the biadjacency matrix into a proper adjacency matrix
        connections = df.reset_index(FRAME).drop(columns=FRAME)
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
        3. tabulate the solvent involved in each network and store in a DataFrame
        """
        solvents = [self.solvents] if isinstance(self.solvents, str) else self.solvents
        solvation_subset = self.solvation_data[np.isin(self.solvation_data[SOLVENT], solvents)]
        # create adjacency matrix from solvation_subset
        graph = calculate_adjacency_dataframe(solvation_subset)
        network_arrays = []
        # loop through each time step / frame
        for frame, df in graph.groupby(FRAME):
            # drop empty columns
            df = df.loc[:, (df != 0).any(axis=0)]
            # save map from local index to residue index
            solute_map = df.index.get_level_values(SOLUTE_IX).values
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
        if len(network_arrays) == 0:
            all_clusters = []
        else:
            all_clusters = np.concatenate(network_arrays)
        cluster_df = (
            pd.DataFrame(all_clusters, columns=[FRAME, NETWORK, SOLVENT, SOLVENT_IX])
                .set_index([FRAME, NETWORK])
                .sort_values([FRAME, NETWORK])
        )
        return cluster_df

    def _calculate_network_sizes(self):
        # This utility calculates the network sizes and returns a convenient dataframe.
        cluster_df = self.network_df
        cluster_sizes = cluster_df.groupby([FRAME, NETWORK]).count()
        size_counts = cluster_sizes.groupby([FRAME, SOLVENT]).count().unstack(fill_value=0)
        size_counts.columns = size_counts.columns.droplevel(None)  # the column value is None
        return size_counts

    def _calculate_solute_status(self):
        """
        This utility calculates the fraction of each solute with a given "status".
        Namely, whether the solvent is "isolated", "paired" (with a single solvent), or
        "networked" of > 2 species.
        """
        # an empty df with the right index
        status = self.network_sizes.iloc[:, 0:0]
        status[PAIRED] = self.network_sizes.iloc[:, 0:1].sum(axis=1).astype(int)
        status[NETWORKED] = self.network_sizes.iloc[:, 1:].sum(axis=1).astype(int)
        status[ISOLATED] = self.n_solute - status.loc[:, [PAIRED, NETWORKED]].sum(axis=1)
        status = status.loc[:, [ISOLATED, PAIRED, NETWORKED]]
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
            >>> solute = Solute(Li, {'BN': BN, 'FEC': FEC, 'PF6': PF6})
            >>> networking = Networking.from_solute(solute, 'PF6')
            >>> res_ix = networking.get_network_res_ix(1, 5)
            >>> solute.u.residues[res_ix].atoms
            <AtomGroup with 126 Atoms>

        """
        res_ix = self.network_df.loc[pd.IndexSlice[frame, network_index], SOLVENT_IX].values
        return res_ix.astype(int)

    @property
    def network_df(self):
        """
        The dataframe containing all networking data. the indices are the frame and
        network index, respectively. the columns are the solvent_name and res_ix.
        """
        return self._network_df

    @property
    def network_sizes(self):
        """
        A dataframe of network sizes. the index is the frame. the column headers
        are network sizes, or the number of solutes + solvents in the network, so
        the columns might be [2, 3, 4, ...]. the values in each column are the
        number of networks with that size in each frame.
        """
        return self._network_sizes

    @property
    def solute_status(self):
        """
        A dictionary where the keys are the "status" of the solute and the values
        are the fraction of solute with that status, averaged over all frames.
        "isolated" means that the solute not coordinated with any of the networking
        solvents, network size is 1.
        "paired" means the solute and is coordinated with a single networking
        solvent and that solvent is not coordinated to any other solutes, network
        size is 2.
        "networked" means that the solute is coordinated to more than one solvent
        or its solvent is coordinated to more than one solute, network size >= 3.
        """
        return self._solute_status

    @property
    def solute_status_by_frame(self):
        """
        As described above, except organized into a dataframe where each
        row is a unique frame and the columns are "isolated", "paired", and "networked".
        """
        return self._solute_status_by_frame
