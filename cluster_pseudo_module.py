from __future__ import print_function

# multi purpose
import numpy
import scipy

# needed parts of nifty
import nifty
import nifty.segmentation
import nifty.filters
import nifty.graph.rag

import torch
from torch.autograd import Variable

import nn_modules
import os.path


# let the craziness begin
class ClusterCallback(nifty.graph.EdgeContractionGraphCallback):


    def __init__(self,cluster_net, rag, edge_feat, node_feat, edge_gt = None):

        super(ClusterCallback, self).__init__()


        # get the modules from parent
        self.edge_init_module     = cluster_net.edge_init_module  
        self.edge_merge_module    = cluster_net.edge_merge_module   
        self.edge_priority_module = cluster_net.edge_priority_module


        self.cluster_net = cluster_net
        # lists of torch tensors
        tl_edge_gt        = [None]*rag.numberOfEdges
        tl_edge_features  = [None]*rag.numberOfEdges
        tl_node_features  = [None]*rag.numberOfNodes
        tl_edge_sizes  = [None]*rag.numberOfEdges
        #tl_node_sizes  = [None]*rag.numberOfNodes
        has_gt = False
        for e in rag.edges():
            ef = edge_feat[e,:]
            ef  = torch.from_numpy(ef[None,:])
            ef  = Variable(ef,requires_grad=True)
            tl_edge_features[e] = self.edge_init_module(ef)

            if edge_gt is not None:
                has_gt = True
                egt = edge_gt[e]
                egt = torch.FloatTensor([float(egt)])
                egt = Variable(egt,requires_grad=True)
                tl_edge_gt[e] = egt

        for n in rag.nodes():
            nf = node_feat[n,:]
            nf  = torch.from_numpy(nf[None,:])
            nf  = Variable(nf,requires_grad=True)
            tl_node_features[n] = nf





        self.tl_edge_features = tl_edge_features
        self.tl_node_features = tl_node_features
        self.tl_edge_gt = tl_edge_gt
        #self.n_edge_feat = edge_feat.shape[1]
        #self.n_node_feat = node_feat.shape[1]



        # the prio-queue
        self.pq =nifty.tools.ChangeablePriorityQueue(rag.numberOfEdges)

        for e in rag.edges():
            self.pq.push(e, self.get_float_priority(e))

    def contractEdge(self, edge_to_contract):
        #print("     contract edge",edge_to_contract)
        self.pq.deleteItem(edge_to_contract)

    def mergeEdges(self, alive_edge, dead_edge):
        #print("     merge edges",alive_edge, dead_edge)

        # get tensors
        ta = self.tl_edge_features[alive_edge]
        td = self.tl_edge_features[dead_edge]

        # merge tensors with a neural network (kind-of)
        self.tl_edge_features[alive_edge] = self.edge_merge_module(ta, td)

        # update prio
        self.pq.push(alive_edge, self.get_float_priority(alive_edge))

        # delete dead edge from prio
        self.pq.deleteItem(dead_edge)


        if self.tl_edge_gt[alive_edge] is not None:
            self.tl_edge_gt[alive_edge] = (self.tl_edge_gt[alive_edge] + self.tl_edge_gt[alive_edge])/2

    def mergeNodes(self, aliveNode, deadNode):
        #("     merge nodes", aliveNode, deadNode)
        pass

    def contractEdgeDone(self, contractedEdge):
        #print("     contract edge done", contractedEdge)
        pass

    # these calls are NOT called by cpp any more
    def get_priority(self, edge_index):
        te = self.tl_edge_features[edge_index]
        p = self.edge_priority_module(te)
        return p

    def get_float_priority(self, edge_index):
        return float(self.get_priority(edge_index).data.numpy()[0])

    def get_gt(self, edge_index):
        return self.tl_edge_gt[edge_index]




class ClusterPseudoModule(object):
    def __init__(self, n_edge_feat_in, n_node_feat_in):

        # input shapes
        self.n_edge_feat_in = n_edge_feat_in
        self.n_node_feat_in = n_node_feat_in

        # real pytorch modules
        self.edge_init_module     = nn_modules.EdgeInitModule(n_edge_feat_in=n_edge_feat_in, n_edge_feat_out=n_edge_feat_in)
        self.edge_merge_module    = nn_modules.EdgeMergeModule(n_feat= n_edge_feat_in)
        self.edge_priority_module = nn_modules.EdgePriorityModule(n_edge_feat= n_edge_feat_in)


        self.nn_modules = [
            self.edge_init_module,    
            self.edge_merge_module,   
            self.edge_priority_module
        ]


    def cluster_callback_factory(self,rag, edge_feat, node_feat, edge_gt = None):
        return ClusterCallback(cluster_net=self, rag=rag, edge_feat=edge_feat, 
            node_feat=node_feat, edge_gt=edge_gt)



    def parameters(self):
        return list(self.edge_merge_module.parameters()) + list(self.edge_priority_module.parameters())



    def train(self):
        for nn_m in self.nn_modules:
            nn_m.train()

    def eval(self):
        for nn_m in self.nn_modules:
            nn_m.eval()



    def save_parameters(self, base_path):

        for i,m in enumerate(self.nn_modules):
            p  = base_path + "_layer_%d"%i
            torch.save(m.state_dict(), p)



    def load_parameters(self, base_path):


        for i,m in enumerate(self.nn_modules):
            p  = base_path + "_layer_%d"%i
            if os.path.isfile(p):
                m.load_state_dict(torch.load(p))
            else:
                raise FileNotFoundError    
    