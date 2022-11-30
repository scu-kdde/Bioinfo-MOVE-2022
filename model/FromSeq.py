import sys
from torch import nn
import torch
from torch.nn import functional as F
from config import Config


class FromSeq(nn.Module):
    def __init__(self) -> None:
        super(FromSeq, self).__init__()
        
        config = Config()
        self.smi_emb = nn.Embedding(config.charsmiset_size+1, config.embedding_size)
        self.smi_conv1 = nn.Conv1d(1,config.num_filters,(config.smi_window_lengths,config.embedding_size),stride=1,padding=0)
        self.smi_conv2 = nn.Conv1d(config.num_filters,config.num_filters*2,(config.smi_window_lengths,1),stride=1,padding=0)
        self.smi_conv3 = nn.Conv1d(config.num_filters*2,config.hidden_dim,(config.smi_window_lengths,1),stride=1,padding=0)
        self.smi_maxpool = nn.MaxPool2d(kernel_size=(1,141))
        
        self.fas_emb = nn.Embedding(config.charseqset_size+1, config.embedding_size)
        self.fas_conv1 = nn.Conv1d(1,config.num_filters,(config.fas_window_lengths,config.embedding_size),stride=1,padding=0)
        self.fas_conv2 = nn.Conv1d(config.num_filters,config.num_filters*2,(config.fas_window_lengths,1),stride=1,padding=0)
        self.fas_conv3 = nn.Conv1d(config.num_filters*2,config.hidden_dim,(config.fas_window_lengths,1),stride=1,padding=0)
        self.fas_maxpool = nn.MaxPool2d(kernel_size=(1,1479))
        
    
    
    def forward(self,smiles,fasta):
        smiles_vector = self.smi_emb(smiles)   
        smiles_vector = torch.unsqueeze(smiles_vector,1) 
        smiles_vector = self.smi_conv1(smiles_vector)       
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = self.smi_conv2(smiles_vector)  
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = self.smi_conv3(smiles_vector)  
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = smiles_vector.squeeze()         
        smiles_vector = self.smi_maxpool(smiles_vector)  
        smile_seq = smiles_vector.squeeze()             
        
        fasta_vector = self.fas_emb(fasta)          
        fasta_vector = torch.unsqueeze(fasta_vector,1)   
        fasta_vector = self.fas_conv1(fasta_vector)  
        fasta_vector = torch.relu(fasta_vector)     
        fasta_vector = self.fas_conv2(fasta_vector)  
        fasta_vector = torch.relu(fasta_vector)      
        fasta_vector = self.fas_conv3(fasta_vector)  
        fasta_vector = fasta_vector.squeeze()         
        fasta_vector = self.fas_maxpool(fasta_vector)   
        fasta_seq = fasta_vector.squeeze()      
        return smile_seq,fasta_seq
    

        
        