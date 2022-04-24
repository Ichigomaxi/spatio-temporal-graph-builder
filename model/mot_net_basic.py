from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import torch
import torch.nn.functional as F

from model.message_passing_module import Message_Passing_spatio_temporal

class MOTNet(torch.nn.Module):
    def __init__(self, input_feature_size:int):
        super(MOTNet, self).__init__()
        model_params = {}
        message_passing_param_dict = {'node_feature_input_size': 4, 
                                    "edge_feature_inpute_size": 2,
                                    "node_feature_output_size":10,
                                    "edge_feature_output_size":10,
                                    }

        
        model_params['message_passing_param_dict'] = message_passing_param_dict

        self.MessPassNet = Message_Passing_spatio_temporal(input_feature_size, message_passing_param_dict)
        self.EdgeClassifier = torch.nn.Sequential(*[
                    torch.nn.Linear(message_passing_param_dict["edge_feature_output_size"],3),
                    torch.nn.LogSigmoid()
                    ])
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.squeeze(1)        

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x

    