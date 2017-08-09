#!/usr/bin/env python
from __future__ import print_function
from itertools import count
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

import numpy
import nifty
import nifty.graph.rag
import nifty.segmentation
import skimage.data    as sdata
import skimage.filters as sfilt



# multi purpose
import numpy
import scipy

# plotting
import pylab

# to download data and unzip it
import os
import urllib.request
import zipfile

# to read the tiff files
import skimage.io
import skimage.filters
import skimage.morphology

# classifier
from sklearn.ensemble import RandomForestClassifier

# needed parts of nifty
import nifty
import nifty.segmentation
import nifty.filters
import nifty.graph.rag
import nifty.ground_truth
from cluster_pseudo_module import ClusterPseudoModule



import sklearn.preprocessing
import utilities

#############################################################
# Download  ISBI 2012:
# =====================
# Download the  ISBI 2012 dataset 
# and precomputed results form :cite:`beier_17_multicut`
# and extract it in-place.
fname = "data.zip"
url = "http://files.ilastik.org/multicut/NaturePaperDataUpl.zip"
if not os.path.isfile(fname):
    urllib.request.urlretrieve(url, fname)
    zip = zipfile.ZipFile(fname)
    zip.extractall()


#############################################################
# Setup Datasets:
# =================
# load ISBI 2012 raw and probabilities
# for train and test set
# and the ground-truth for the train set
raw_dsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_train.tif'),
    'test' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_test.tif'),
}
# read pmaps and convert to 01 pmaps
pmap_dsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/probabilities_train.tif'),
    'test' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/probabilities_test.tif'),
}
pmap_dsets = {
    'train' : pmap_dsets['train'].astype('float32')/255.0,
    'test' : pmap_dsets['test'].astype('float32')/255.0
}
gt_dsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/groundtruth.tif'),
    'test'  : None
}




out_dir = "out"






#############################################################
# stats of precomputed features:
# ===================
# Here we train the net
n_train_iter = 10
n_training_instances = raw_dsets['train'].shape[0]
raw_dset  = raw_dsets['train']
pmap_dset = pmap_dsets['train']
gt_dset   = gt_dsets['train']
all_precomputed_edge_feat = []
all_precomputed_node_feat = []

for train_iter in range(n_train_iter):

    # get a random slice 
    slice_index = numpy.random.randint(n_training_instances)
    #print("training iter%d using slice %d"%(train_iter,slice_index))
 

    
    # get raw data, pmap and gt
    raw             = raw_dset[slice_index, :, :]
    pmap            = pmap_dset[slice_index, :, :]
    binary_gt_image = gt_dset[slice_index, :, :]

    # oversementation rag and edge gt
    overseg, rag, edge_gt = utilities.get_overseg_rag_and_gt(raw=raw, pmap=pmap, binary_gt_image=binary_gt_image)



    # precomputed features
    precomputed_feat = utilities.get_precomputed_feat(raw=raw, pmap=pmap, rag=rag)
    precomputed_edge_feat, precomputed_node_feat  = precomputed_feat


    all_precomputed_edge_feat.append(precomputed_edge_feat)
    all_precomputed_node_feat.append(precomputed_node_feat)



precomputed_edge_feat = numpy.concatenate(all_precomputed_edge_feat)
precomputed_node_feat = numpy.concatenate(all_precomputed_node_feat)
node_feat_scaler = sklearn.preprocessing.StandardScaler()
edge_feat_scaler = sklearn.preprocessing.StandardScaler()
node_feat_scaler.fit(precomputed_edge_feat)
edge_feat_scaler.fit(precomputed_node_feat)
n_precomputed_edge_feat = precomputed_edge_feat.shape[1]
n_precomputed_node_feat = precomputed_node_feat.shape[1]





#############################################################
#
# the pseudo net:
# =================
#
#############################################################

pseudo_net = ClusterPseudoModule(n_edge_feat_in=n_precomputed_edge_feat,
    n_node_feat_in=n_precomputed_node_feat)
parameters = pseudo_net.parameters()


# load potential parameters
from_sratch = True
try:
    from_sratch = False
    pseudo_net.load_parameters(os.path.join(out_dir,"params.pkl"))
except FileNotFoundError:
    pass

if from_sratch:
    print("NEW    parameters")
else:
    print("loaded parameters")



if True:

    #############################################################
    # Training Loop:
    # ===================
    # Here we train the net



    # the optimizer
    learning_rate = 0.00075
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)



    n_train_iter = 200
    n_training_instances = raw_dsets['train'].shape[0]

    raw_dset  = raw_dsets['train']
    pmap_dset = pmap_dsets['train']
    gt_dset   = gt_dsets['train']

    for train_iter in range(n_train_iter):






        # get a random slice 
        slice_index = numpy.random.randint(n_training_instances)
        print("training iter%d using slice %d"%(train_iter,slice_index))

        
        # get raw data, pmap and gt
        raw             = raw_dset[slice_index, :, :]
        pmap            = pmap_dset[slice_index, :, :]
        binary_gt_image = gt_dset[slice_index, :, :]

        # oversementation rag and edge gt
        overseg, rag, edge_gt = utilities.get_overseg_rag_and_gt(raw=raw, pmap=pmap, binary_gt_image=binary_gt_image)
        edge_gt = edge_gt.round()

        # precomputed features
        precomputed_feat = utilities.get_precomputed_feat(raw=raw, pmap=pmap, rag=rag)
        precomputed_edge_feat, precomputed_node_feat  = precomputed_feat
        

        # normalize
        precomputed_edge_feat = node_feat_scaler.transform  (precomputed_edge_feat)  
        precomputed_node_feat = edge_feat_scaler.transform(precomputed_node_feat) 


        # generate the cluster callback
        cluster_callback = pseudo_net.cluster_callback_factory(rag=rag,
            edge_feat=precomputed_edge_feat,node_feat=precomputed_node_feat,
            edge_gt=edge_gt)


        # here we construct the edge contraction graph
        cg = nifty.graph.edgeContractionGraph(rag, cluster_callback)

        # the loss
        loss = Variable(torch.FloatTensor([1]))
        loss[0] = 0

        # here we do a single round of clustering
        # which will also give us the loss
        pq = cluster_callback.pq
        counter = 0 
        while(cg.numberOfEdges>=1):
            
            #print(cg.numberOfEdges)

            # the lowest edge
            min_edge = pq.top()
            min_p = pq.topPriority()

            if min_p >= 2.0:
                #print("early exit v2.0")
                break

            #get the gt of this very edge
            gt = cluster_callback.get_gt(min_edge)
            float_gt = gt.data.numpy()[0]

            #print("min_edge",min_edge,"min_p",min_p,"gt",float_gt)

            prio_tensor = cluster_callback.get_priority(min_edge)

            # if gt wants to merge
            if(float_gt < 0.5):
                # easy case:
                # we merge
                cg.contractEdge(min_edge)

                # it would be better if  min_p would approach zero
                loss += prio_tensor


            # gt want to keep them separated
            else:
                loss += 1.0 - prio_tensor
                
                # yes this no bug
                pq.push(min_edge, 99.0)

            counter += 1


        average_loss = loss.data.numpy()[0]/counter
        print("average loss", average_loss)



           




        # aaaand a gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # and save the parameters
        pseudo_net.save_parameters(os.path.join(out_dir,"params.pkl"))



else :

    raw_dset  = raw_dsets['test']
    pmap_dset = pmap_dsets['test']





    # get a random slice 
    slice_index = 5
    print("slice %d"%(slice_index))

    
    # get raw data, pmap and gt
    raw             = raw_dset[slice_index, :, :]
    pmap            = pmap_dset[slice_index, :, :]
    #binary_gt_image = gt_dset[slice_index, :, :]

    # oversementation rag and edge gt
    overseg, rag = utilities.get_overseg_rag(raw=raw, pmap=pmap)


    # precomputed features
    precomputed_feat = utilities.get_precomputed_feat(raw=raw, pmap=pmap, rag=rag)
    precomputed_edge_feat, precomputed_node_feat  = precomputed_feat
    

    # normalize
    precomputed_edge_feat = node_feat_scaler.transform  (precomputed_edge_feat)  
    precomputed_node_feat = edge_feat_scaler.transform(precomputed_node_feat) 


    # generate the cluster callback
    cluster_callback = pseudo_net.cluster_callback_factory(rag=rag,
        edge_feat=precomputed_edge_feat,node_feat=precomputed_node_feat)


    # here we construct the edge contraction graph
    cg = nifty.graph.edgeContractionGraph(rag, cluster_callback)

    # the loss
    loss = Variable(torch.FloatTensor([1]))
    loss[0] = 0

    # here we do a single round of clustering
    # which will also give us the loss
    pq = cluster_callback.pq
    counter = 0 
    while(cg.numberOfEdges>=1):
        
        #print(cg.numberOfEdges)

        # the lowest edge
        min_edge = pq.top()
        min_p = pq.topPriority()
        print("min p",min_p)
        if min_p >= 0.5:
            print("DOOOONE")
            break



        
        cg.contractEdge(min_edge)


    res = cg.nodeUfd.find(numpy.arange(rag.numberOfNodes))

    res = nifty.graph.rag.projectScalarNodeDataToPixels(rag, res)

    import pylab
    pylab.imshow(nifty.segmentation.segmentOverlay(raw, res, 0.2, thin=False))
    pylab.show()
    print(res)
