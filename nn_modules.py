#!/usr/bin/env python
from __future__ import print_function
# from itertools import count
# import torch
# import torch.autograd
# import torch.nn.functional as F
# from torch.autograd import Variable
from torch import nn
import torch
import torch.nn.functional as F





class Mlp(nn.Module):

    def __init__(self, n_in, n_out, n_hidden=None, n_hidden_factor=2,
            nl='relu', final_nl='relu', dropout=0.5, resiudal_if_possible=False):
        super(Mlp, self).__init__()

        # settings
        if n_hidden is None:
            n_hidden = int(round(n_hidden_factor * n_in))
        self.nl = nl
        self.final_nl = final_nl
        self.residual = resiudal_if_possible and n_in == n_out




        self.bn_in = torch.nn.BatchNorm1d(n_hidden)
        self.bn_hidden = torch.nn.BatchNorm1d(n_hidden)
        # layers
        self.linear_in = nn.Linear(n_in, n_hidden)

        self.linear_hidden = nn.Linear(n_hidden, n_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_out = nn.Linear(n_hidden, n_out)

        


    def non_linearity(self,x, nl_name):
        if nl_name == 'sigmoid':
            return F.sigmoid(x)
        elif nl_name == 'relu':
            return F.relu(x)
        else:
            raise NameError('%s is an unknown nonlinearity'%nl_name)


    def forward(self,  x_in):

        # in 
        x = self.linear_in(x_in)
        x = self.non_linearity(x, nl_name=self.nl)
        x = self.bn_in(x)

        # hidden
        x = self.linear_hidden(x)
        x = self.non_linearity(x, nl_name=self.nl)
        x = self.bn_hidden(x)
        x = self.dropout(x)

        # out
        x = self.linear_out(x)
        x = self.non_linearity(x, nl_name=self.final_nl)

        if self.residual:
            return (x_in + x)/2.0
        else:
            return x 






class NodeToEdgeFeatModule(nn.Module):

    def __init__(self, n_in, n_out):
        super(NodeToEdgeFeatModule, self).__init__()
        
        
        self.mlp = Mlp(n_in=n_in*2, n_out=n_out, 
                           resiudal_if_possible=True)

    def forward(self, xa, xb):

        # on concat
        xab = torch.cat([xa, xb], 1)
        xba = torch.cat([xb, xa], 1)

        xab = self.mlp(xab)
        xba = self.mlp(xba)

      
        return (xab + xba)/2




class SizeRegularizer(nn.Module):
    def __init__(self, n_edge_feat):
        super(SizeRegularizer, self).__init__()
        self.mlp_gamma = Mlp(n_in=n_edge_feat, n_out=1,
                          final_nl='sigmoid')

    def forward(self, u_size, v_size, edge_features):

        # gamma in the range 0-1
        gamma = self.mlp_gamma(edge_features)

        tmp = (1.0/u_size)**gamma + (1.0/v_size)**gamma 
        return 1.0/tmp



class EdgePriorityModule(nn.Module):

        def __init__(self, n_edge_feat, node_to_edge_feat_module, n_node_feat):
            super(EdgePriorityModule, self).__init__()
        

            self.node_to_edge_feat_module = node_to_edge_feat_module
            self.mlp_raw_prio = Mlp(n_in=n_edge_feat + n_node_feat, n_out=1,
                          final_nl='sigmoid')

            self.mlp_prio = Mlp(n_in=3,n_hidden=9, n_out=1,
                          final_nl='sigmoid')


            self.size_regularizer_mlp = SizeRegularizer(n_edge_feat=n_edge_feat + n_node_feat)
            

        def forward(self, edge_features, fu, fv, e_size, u_size, v_size):

            from_nodes = self.node_to_edge_feat_module(fu, fv)
            all_feat = torch.cat([edge_features, from_nodes], 1)
            raw_prio = self.mlp_raw_prio(all_feat)
            size_reg = self.size_regularizer_mlp(u_size=u_size,v_size=v_size,edge_features=all_feat)
            size_reg = torch.unsqueeze(size_reg,0)
            size_weighted_prio = raw_prio*size_reg/100.0

            final_feat = torch.cat([raw_prio, size_reg, size_weighted_prio],1)
            beta = 0.85 
            return beta*self.mlp_prio(final_feat) + (1.0 - beta )*raw_prio




            

            
       




class InitModule(nn.Module):

        def __init__(self, n_feat_in, n_feat_out):
            super(InitModule, self).__init__()


            self.mlp = Mlp(n_in=n_feat_in, n_out=n_feat_out, 
                           resiudal_if_possible=True)

        def forward(self, x):
            return  self.mlp(x)




class MergeModule(nn.Module):

    def __init__(self, n_feat):
        super(MergeModule, self).__init__()
        self.n_feat = n_feat


        # acts on inputs
        self.linear_input = nn.Linear(n_feat, n_feat*1)

        # acts on residual
        self.linear_residual = nn.Linear(n_feat*1, n_feat*2)

        # acts on concat
        self.linear_concat = nn.Linear(n_feat*2,n_feat*2)

        self.linear_2 = nn.Linear(n_feat*2,n_feat)

        self.dropout = nn.Dropout(p=0.3)


    def forward(self, x_a, x_b):

        x_a = self.dropout(x_a)
        x_b = self.dropout(x_b)

        ##################################
        # layers on inputs themselfs
        ##################################
        r_a = ( F.relu(self.linear_input(x_a)) + x_a  )/2.0
        r_b = ( F.relu(self.linear_input(x_b)) + x_b  )/2.0

        ##################################
        # make input symetric
        ##################################
        
        ab = torch.cat([r_a, r_b], 1)
        ba = torch.cat([r_b, r_a], 1)


        # linear on concats and then residual
        res_ab = F.relu(self.linear_concat(ab))
        res_ba = F.relu(self.linear_concat(ba))
        res_concat = (res_ab + res_ba)/2.0

        # linear on residual
        res_residual  = F.relu(self.linear_residual((r_a + r_b)/2))

        res = (res_concat + res_residual)/2.0

        res = self.dropout(res)

        res = F.relu(self.linear_2(res)) 

        return res



