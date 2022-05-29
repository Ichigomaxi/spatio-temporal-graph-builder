import torch
import numpy as np

# Load the summary/ submission 
if __name__ == "__main__":
    
    print("torch.cat")
    dummy_index_a = torch.empty(0,dtype=torch.long)
    dummy_index_b = torch.empty(0,dtype=torch.long)
    dummy_index_c = torch.tensor([2,0],dtype=torch.long)
    list_dummy = [dummy_index_a, dummy_index_b,dummy_index_c]
    print(list_dummy)
    edge_indices = torch.cat(list_dummy)
    print(edge_indices)

    # This can help
    print("torch.cat with empty lists")
    dummy_index_a = torch.tensor([],dtype=torch.long)
    dummy_index_b = torch.tensor([],dtype=torch.long)
    dummy_index_c = torch.tensor([2,0],dtype=torch.long)
    list_dummy = [dummy_index_a, dummy_index_b,dummy_index_c]
    print(list_dummy)
    edge_indices = torch.cat(list_dummy)
    print(edge_indices)

    # this is unusable
    print("torch.stack")
    dummy_index_a = torch.empty(2)
    dummy_index_b = torch.empty(2)
    dummy_index_c = torch.tensor([2,0])
    list_dummy = [dummy_index_a, dummy_index_b,dummy_index_c]
    print(list_dummy)
    edge_indices = torch.stack(list_dummy)
    print(edge_indices)