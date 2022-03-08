from collections import defaultdict
from enum import Enum

from matplotlib.pyplot import axis
from utility import get_box_centers

import numpy as np

class Timeframe(Enum):
    t0 = 0
    t1 = 1
    t2 = 2

class Graph(object):
    """ Graph data structure, undirected by default.
    Taken from https://stackoverflow.com/questions/19472530/representing-graphs-data-structure-in-python
    After consideration of different Data Structures: https://en.wikipedia.org/wiki/Graph_(abstract_data_type)#Representations
    """

    def __init__(self, connections, directed=False):
        self._graph = defaultdict(set)
        self._directed = directed
        self.add_connections(connections)

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """

        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].add(node2)
        if not self._directed:
            self._graph[node2].add(node1)

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self._graph.items():  # python3: items(); python2: iteritems()
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """

        return node1 in self._graph and node2 in self._graph[node1]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))

class SpatioTemporalGraph(Graph):
    """ Special Graph representation for spatio-temporal graphs
    Graph data structure, undirected by default.
    Taken from https://stackoverflow.com/questions/19472530/representing-graphs-data-structure-in-python
    After consideration of different Data Structures: https://en.wikipedia.org/wiki/Graph_(abstract_data_type)#Representations
    """
    def __init__(self,box_center_list , connections, directed=False, ):
        super().__init__(connections, directed)
        
        self._center_points = []
        self._center_points_stacked = np.empty((0,3))

        for boxes in box_center_list:
            box_center_points = get_box_centers(boxes)
            self._center_points.append([box_center_points])
            np.append(self._center_points_stacked,box_center_points, axis = 0)

    def __init__(self, connections, directed=False, ):
        super().__init__(connections, directed)
        
        self._center_points = []
        self._center_points_stacked = np.empty((0,3))

    def get_spatial_pointpairs(self, timeframe: Timeframe):
        spatial_pointpairs = []
        for reference_node in self._graph:
            if(reference_node[0]== timeframe):
                for neighbor_node in self._graph[reference_node]:
                    # print(neighbor_index[0])
                    timestep, idx = neighbor_node[0],neighbor_node[1]
                    if timestep == timeframe:
                        spatial_pointpairs.append([reference_node[1],idx])
        return spatial_pointpairs
        
    def get_temporal_pointpairs(self):
        temporal_pairs_indices = []
        for reference_node in self._graph:
            reference_timeframe = reference_node[0]
            # Find corresponding indices in global centers list
            point_a = self.get_points(reference_node)
            reference_idx_global = np.argwhere(self._center_points_stacked == point_a)[0,0]

            for neighbor_node in self._graph[reference_node]:
                # print(neighbor_index[0])
                neighbor_timeframe, neighbor_idx = neighbor_node[0],neighbor_node[1]
                if neighbor_timeframe != reference_timeframe:
                    # Find corresponding indices in global centers list
                    point_b = self.get_points(neighbor_node)
                    neighbor_idx_global = np.argwhere(self._center_points_stacked == point_b)[0,0]
                    #Append global indices into list
                    temporal_pairs_indices.append([reference_idx_global,neighbor_idx_global])
        return temporal_pairs_indices
        
    def get_points(self,reference_node):
        if(reference_node[0]== Timeframe.t0):
            return self._center_points[0][reference_node[1]]
        elif (reference_node[0]== Timeframe.t1):
            return self._center_points[1][reference_node[1]]
        elif (reference_node[0]== Timeframe.t2):
            return self._center_points[2][reference_node[1]]
        else:
            return AttributeError