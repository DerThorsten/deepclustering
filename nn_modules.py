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


class EdgePriorityModule(nn.Module):

        def __init__(self, n_edge_feat):
            super(EdgePriorityModule, self).__init__()
            self.n_edge_feat = n_edge_feat
            fac = 2
            self.linear_1 = nn.Linear(n_edge_feat,n_edge_feat*fac)
            self.linear_2 = nn.Linear(n_edge_feat*fac,n_edge_feat*fac)
            self.linear_3 = nn.Linear(n_edge_feat*fac,n_edge_feat*fac)
            #self.linear_4 = nn.Linear(n_edge_feat*fac,n_edge_feat*fac)
            #self.linear_5 = nn.Linear(n_edge_feat*fac,n_edge_feat*fac)
            self.linear_6 = nn.Linear(n_edge_feat*fac,1)


        def forward(self, edge_features):
            res_1 =  (F.relu(self.linear_1(edge_features)))
            res_2 =  (F.relu(self.linear_2(res_1)) + res_1)/2.0
            res_3 =  (F.relu(self.linear_3(res_2)) + res_1 + res_2)/3.0
            #res_4 =  (F.relu(self.linear_4(res_3)) + res_1 + res_2 + res_3)/4.0
            #res_5 =  (F.relu(self.linear_5(res_4)))
            res =  F.sigmoid(self.linear_6(res_3))

            return res




class EdgeInitModule(nn.Module):

        def __init__(self, n_edge_feat_in, n_edge_feat_out):
            super(EdgeInitModule, self).__init__()
            self.n_edge_feat_in = n_edge_feat_in
            self.n_edge_feat_out = n_edge_feat_out

            self.linear_1 = nn.Linear(n_edge_feat_in,n_edge_feat_out)
            # self.linear_2 = nn.Linear(n_edge_feat_out,n_edge_feat_out)
            # self.linear_3 = nn.Linear(n_edge_feat_out,n_edge_feat_out)
            # self.linear_4 = nn.Linear(n_edge_feat_out,n_edge_feat_out)
            # self.linear_5 = nn.Linear(n_edge_feat_out,n_edge_feat_out)

        def forward(self, edge_features):
            res_1 =  (F.relu(self.linear_1(edge_features))+edge_features)/2.0
            #res_2 =  (F.relu(self.linear_2(res_1)) + res_1)/2.0
            #res_3 =  (F.relu(self.linear_3(res_2)) + res_1 + res_2)/3.0
            #res_4 =  (F.relu(self.linear_4(res_3)) + res_1 + res_2 + res_3)/4.0
            #res_5 =  (F.relu(self.linear_5(res_4)))


            return res_1





class EdgeMergeModule(nn.Module):

    def __init__(self, n_feat):
        super(EdgeMergeModule, self).__init__()
        self.n_feat = n_feat


        # acts on inputs
        self.linear_input = nn.Linear(n_feat, n_feat*1)

        # acts on residual
        self.linear_residual = nn.Linear(n_feat*1, n_feat*2)

        # acts on concat
        self.linear_concat = nn.Linear(n_feat*2,n_feat*2)

        self.linear_2 = nn.Linear(n_feat*2,n_feat)


    def forward(self, x_a, x_b):
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

        # res is n_feat * 12
        res = (res_concat + res_residual)/2.0

        res = F.relu(self.linear_2(res)) 

        return res
