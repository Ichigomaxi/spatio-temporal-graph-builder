import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

class Message_Passing_spatio_temporal(MessagePassing):
    def __init__(self,  in_channels,
                        out_channels):
        super(Message_Passing_spatio_temporal, self).__init__(aggr='max') #  "Max" aggregation.

        self.lin = torch.nn.Linear(4, 10,bias= True )
        self.act = torch.nn.ReLU()

        # Node to Edge Update
        self.update_lin = torch.nn.Linear(in_channels * 2 , 10, bias=False)
        self.update_act = torch.nn.ReLU()
        # Edge to node Update

        

    def forward(self, graph_object_mini_batch):
        # Keys: 'x', 'temporal_edges_mask', 'edge_attr', 'edge_labels', 'edge_index'
        x = graph_object_mini_batch.x 
        edge_index = graph_object_mini_batch.edge_index
        edge_feature = graph_object_mini_batch.edge_attr
        # edge_label = graph_object_mini_batch.edge_labels
        temporal_edges_mask = graph_object_mini_batch.temporal_edges_mask

        self.edge_updater(edge_index, edge_feature, temporal_edges_mask)

        nodes_embeddings = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        
        new_edge_embedding = self.edge_update()
        temporal_edge_embeddings = new_edge_embedding[temporal_edges_mask]
        return 

    def message(self, x_i, x_j, edge_feature):
        # compute Message out of edge embeddings and 
        message_i_j = torch.nn.Linear( torch.cat([x_i, edge_feature])  )
        concat_x = torch.cat([x_i,x_j])
        concat_x = self.lin(concat_x)
        concat_x = self.act(concat_x)
        
        return concat_x

    def update(self, aggr_out, x):
        new_embedding = torch.cat([aggr_out, x], dim=1)
        
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        
        return new_embedding
    
    def edge_update(self, edge_index, edge_feature, temporal_edges_mask) -> torch.Tensor:
        r"""Computes or updates features for each edge in the graph.
        This function can take any argument as input which was initially passed
        to :meth:`edge_updater`.
        Furthermore, tensors passed to :meth:`edge_updater` can be mapped to
        the respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        print("edge_index", edge_index)
        # print("edge_index",  )
        print("edge_feature", edge_feature)
        print("temporal_edges_mask", temporal_edges_mask)
