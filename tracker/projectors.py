'''
Taken from https://github.com/dvl-tum/mot_neural_solver
Check out the corresponding Paper https://arxiv.org/abs/1912.07515
This is serves as inspiration for our own code
'''
import numpy as np
import torch
from datasets.mot_graph import Graph
from datasets.nuscenes_mot_graph import NuscenesMotGraph
from torch_scatter import scatter_add, scatter_mean

# from utils.evaluation import compute_constr_satisfaction_rate

def compute_constr_satisfaction_rate(graph_obj: Graph, edges_out, undirected_edges = True, return_flow_vals = False):
    """
    Taken from https://github.com/dvl-tum/mot_neural_solver
    Check out the corresponding Paper https://arxiv.org/abs/1912.07515
    This is serves as inspiration for our own code

    Determines the proportion of Flow Conservation inequalities that are satisfied.
    For each node, the sum of incoming (resp. outgoing) edge values must be less or equal than 1.

    Args:
        graph_obj: 'Graph' object
        edges_out: BINARIZED output values for edges (1 if active, 0 if not active)
        undirected_edges: determines whether each edge in graph_obj.edge_index appears in both directions (i.e. (i, j)
        and (j, i) are both present (undirected_edges =True), or only (i, j), with  i<j (undirected_edges=False)
        return_flow_vals: determines whether the sum of incoming /outglong flow for each node must be returned

    Returns:
        constr_sat_rate: float between 0 and 1 indicating the proprtion of inequalities that are satisfied

    """
    # Get tensors indicataing which nodes have incoming and outgoing flows (e.g. nodes in first frame have no in. flow)
    edge_ixs = graph_obj.temporal_directed_edge_indices # Adapted and isolated to temporal edges and edge predictions

    if undirected_edges:
        sorted, _ = edge_ixs.t().sort(dim = 1)
        sorted = sorted.t()
        div_factor = 2. # Each edge is predicted twice, hence, we divide by 2
    else:
        sorted = edge_ixs # Edges (i.e. node pairs) are already sorted
        div_factor = 1.  # Each edge is predicted once, hence, hence we divide by 1.

    # Compute incoming and outgoing flows for each node
    flow_out = scatter_add(edges_out, sorted[0],dim_size=graph_obj.num_nodes) / div_factor
    flow_in = scatter_add(edges_out, sorted[1], dim_size=graph_obj.num_nodes) / div_factor


    # Determine how many inequalitites are violated
    violated_flow_out = (flow_out > 1).sum()
    violated_flow_in = (flow_in > 1).sum()

    # Compute the final constraint satisfaction rate
    violated_inequalities = (violated_flow_in + violated_flow_out).float()
    flow_out_constr, flow_in_constr= sorted[0].unique(), sorted[1].unique()
    num_constraints = len(flow_out_constr) + len(flow_in_constr)
    constr_sat_rate = 1 - violated_inequalities / num_constraints
    if not return_flow_vals:
        return constr_sat_rate.item()

    else:
        return constr_sat_rate.item(), flow_in, flow_out

class GreedyProjector:
    """
    Applies the greedy rounding scheme described in https://arxiv.org/pdf/1912.07515.pdf, Appending B.1
    """
    def __init__(self, full_graph:NuscenesMotGraph):
        self.final_graph:Graph = full_graph.graph_obj
        self.num_nodes = full_graph.graph_obj.num_nodes
        
    def project(self, threshold:float = 0.5):
        round_preds = (self.final_graph.temporal_directed_edge_preds > threshold).float()
        temporal_edge_index = self.final_graph.temporal_directed_edge_indices

        self.constr_satisf_rate, flow_in, flow_out = compute_constr_satisfaction_rate(graph_obj = self.final_graph,
                                                                                     edges_out = round_preds,
                                                                                     undirected_edges = False,
                                                                                     return_flow_vals = True)
        # Determine the set of constraints that are violated
        nodes_names = torch.arange(self.num_nodes).to(flow_in.device)
        in_type = torch.zeros(self.num_nodes).to(flow_in.device)
        out_type = torch.ones(self.num_nodes).to(flow_in.device)

        flow_in_info = torch.stack((nodes_names.float(), in_type.float())).t()
        flow_out_info = torch.stack((nodes_names.float(), out_type.float())).t()
        all_violated_constr = torch.cat((flow_in_info, flow_out_info))
        mask = torch.cat((flow_in > 1, flow_out > 1))

        # Sort violated constraints by the value of thei maximum pred value among incoming / outgoing edges
        all_violated_constr = all_violated_constr[mask]
        vals, sorted_ix = torch.sort(all_violated_constr[:, 1], descending=True)
        all_violated_constr = all_violated_constr[sorted_ix]

        # Iterate over violated constraints.
        for viol_constr in all_violated_constr:
            node_name, viol_type = viol_constr

            # Determine the set of incoming / outgoing edges
            mask = torch.zeros(self.num_nodes).bool().to(flow_in.device)
            mask[node_name.int()] = True
            if viol_type == 0:  # Flow in violation
                mask = mask[temporal_edge_index[1]]

            else:  # Flow Out violation
                mask = mask[temporal_edge_index[0]]
            flow_edges_ix = torch.where(mask)[0]

            # If the constraint is still violated, set to 1 the edge with highest score, and set the rest to 0
            if round_preds[flow_edges_ix].sum() > 1:
                max_pred_ix = max(flow_edges_ix, key=lambda ix: self.final_graph.temporal_directed_edge_preds[ix]*round_preds[ix]) # Multiply for round_preds so that if the edge has been set to 0
                                                                                                                 # it can not be set back to 1
                round_preds[mask] = 0
                round_preds[max_pred_ix] = 1

        # Assert that there are no constraint violations
        assert scatter_add(round_preds, temporal_edge_index[1], dim_size=self.num_nodes).max() <= 1
        assert scatter_add(round_preds, temporal_edge_index[0], dim_size=self.num_nodes).max() <= 1

        # return round_preds, constr_satisf_rate
        self.final_graph.temporal_directed_edge_preds = round_preds