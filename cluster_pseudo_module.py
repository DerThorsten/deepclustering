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





def numpy_to_float_var(x, requires_grad=True):
    x  = torch.from_numpy(x)
    x  = Variable(x,requires_grad=requires_grad)
    return x

def scalar_to_float_var(x, requires_grad=True):
    x = torch.FloatTensor([float(x)])
    x = Variable(x,requires_grad=requires_grad)
    return x

# let the craziness begin
class ClusterCallback(nifty.graph.EdgeContractionGraphCallback):


    def __init__(self, shape, cluster_net, rag, edge_feat, node_feat, edge_gt = None):

        super(ClusterCallback, self).__init__()


        self.rag = rag
        # accumulate the mean edge value
        # along the superpixel boundaries
        # length of each boundary and node sizes\
        dummy =  numpy.ones(shape).astype('float32')
        __edge_features, __node_features = nifty.graph.rag.accumulateMeanAndLength(
            rag,  dummy, [100,100],1)

        edge_sizes = __edge_features[:,1]
        node_sizes = __node_features[:,1]





        # get the modules from parent
        self.edge_init_module     = cluster_net.edge_init_module  
        self.node_init_module     = cluster_net.node_init_module  
        self.edge_merge_module    = cluster_net.edge_merge_module   
        self.node_merge_module    = cluster_net.node_merge_module   
        self.edge_priority_module = cluster_net.edge_priority_module


        self.cluster_net = cluster_net

        # lists of torch tensors
        self.tl_edge_gt        = [None]*rag.numberOfEdges
        self.tl_edge_features  = [None]*rag.numberOfEdges
        self.tl_node_features  = [None]*rag.numberOfNodes
        self.tl_edge_sizes  = [None]*rag.numberOfEdges
        self.tl_node_sizes  = [None]*rag.numberOfNodes


        for e in rag.edges():
            ef = edge_feat[e,:]
            ef = numpy_to_float_var(ef[None,:])
            self.tl_edge_features[e] = self.edge_init_module(ef)
            self.tl_edge_sizes[e] = scalar_to_float_var(edge_sizes[e])


            if edge_gt is not None:
                self.tl_edge_gt[e] = scalar_to_float_var(edge_gt[e])

        for n in rag.nodes():
            nf = node_feat[n,:]
            nf  = numpy_to_float_var(nf[None,:])
            self.tl_node_features[n] = self.node_init_module(nf)
            self.tl_node_sizes[n] = scalar_to_float_var(node_sizes[n])

        


     

        # the prio-queue
        self.pq =nifty.tools.ChangeablePriorityQueue(rag.numberOfEdges)

        #for e in rag.edges():
        #    self.pq.push(e, self.get_float_priority(e))


    def init_priorities(self, cg):
        self.cg = cg
        for e in self.rag.edges():
            self.pq.push(e, self.get_float_priority(e))


    def contractEdge(self, edge_to_contract):
        #print("     contract edge",edge_to_contract)
        self.pq.deleteItem(edge_to_contract)

    def mergeEdges(self, alive_edge, dead_edge):

        

        # get tensors
        ta = self.tl_edge_features[alive_edge]
        td = self.tl_edge_features[dead_edge]

        # merge tensors with a neural network (kind-of)
        self.tl_edge_features[alive_edge] = self.edge_merge_module(ta, td)

        # update prio
        self.pq.push(alive_edge, self.get_float_priority(alive_edge))

        # delete dead edge from prio
        self.pq.deleteItem(dead_edge)

        size_a = self.tl_edge_sizes[alive_edge]
        size_d = self.tl_edge_sizes[dead_edge]
        new_size = size_a + size_d 
        if self.tl_edge_gt[alive_edge] is not None:
            self.tl_edge_gt[alive_edge] = \
                (size_a*self.tl_edge_gt[alive_edge] + size_d*self.tl_edge_gt[dead_edge])/(new_size)

        sa = self.tl_edge_sizes[alive_edge]
        sd = self.tl_edge_sizes[dead_edge]


        # update sizes
        self.tl_edge_sizes[alive_edge] = new_size

    def mergeNodes(self, alive_node, dead_node):

        # update sizes
        self.tl_node_sizes[alive_node] =  self.tl_node_sizes[alive_node] + self.tl_node_sizes[dead_node]

        # get tensors
        ta = self.tl_node_features[alive_node]
        td = self.tl_node_features[dead_node]

        # merge tensors with a neural network (kind-of)
        self.tl_node_features[alive_node] = self.node_merge_module(ta, td)

        pass

    def contractEdgeDone(self, contracted_edge):
        #print("     contract edge done", contracted_edge)
        new_node = self.cg.nodeOfDeadEdge(contracted_edge)
        for node,edge in self.cg.nodeAdjacency(new_node):

            # update prio
            self.pq.push(edge, self.get_float_priority(edge))

        

    # these calls are NOT called by cpp any more
    def get_priority(self, edge_index):
        te = self.tl_edge_features[edge_index]
        u,v = self.cg.uv(edge_index)
        tu = self.tl_node_features[u]
        tv = self.tl_node_features[v]
        e_size = self.tl_edge_sizes[edge_index]
        u_size = self.tl_node_sizes[u]
        v_size = self.tl_node_sizes[v]
        p = self.edge_priority_module(edge_features=te, fu=tu, fv=tv,
            e_size=e_size,u_size=u_size, v_size=v_size)
        return p

    def get_float_priority(self,  edge_index):
        return float(self.get_priority(edge_index).data.numpy()[0])


    def get_edge_size_tensor(self, edge_index):
        return self.tl_edge_sizes[edge_index]

    def get_node_size_tensor(self, node_index):
        return self.tl_node_sizes[node_index]


    def get_gt(self, edge_index):
        return self.tl_edge_gt[edge_index]




class ClusterPseudoModule(object):
    def __init__(self, n_edge_feat_in, n_node_feat_in):

        # input shapes
        self.n_edge_feat_in = n_edge_feat_in
        self.n_node_feat_in = n_node_feat_in

        # real pytorch modules
        self.edge_init_module     = nn_modules.InitModule(n_feat_in=n_edge_feat_in, n_feat_out=n_edge_feat_in)
        self.node_init_module     = nn_modules.InitModule(n_feat_in=n_node_feat_in, n_feat_out=n_node_feat_in)
        self.edge_merge_module    = nn_modules.MergeModule(n_feat= n_edge_feat_in)
        self.node_merge_module    = nn_modules.MergeModule(n_feat= n_node_feat_in)

        self.node_to_edge_feat_module = nn_modules.NodeToEdgeFeatModule(n_in=n_node_feat_in, n_out=n_node_feat_in)
        self.edge_priority_module = nn_modules.EdgePriorityModule(n_edge_feat= n_edge_feat_in, 
            node_to_edge_feat_module=self.node_to_edge_feat_module,
            n_node_feat=n_node_feat_in)


        self.nn_modules = [
            self.edge_init_module,    
            self.node_init_module,
            self.edge_merge_module,   
            self.node_merge_module,
            self.node_to_edge_feat_module,
            self.edge_priority_module
        ]


    def cluster_callback_factory(self,shape, rag, edge_feat, node_feat, edge_gt = None):
        return ClusterCallback(shape=shape,cluster_net=self, rag=rag, 
            edge_feat=edge_feat, node_feat=node_feat, 
            edge_gt=edge_gt)



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
    