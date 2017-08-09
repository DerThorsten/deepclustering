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




#############################################################
# Helper Functions:
# ===================
# Function to compute features for a RAG
# (used later)
def get_precomputed_feat(raw, pmap, rag):
    uv = rag.uvIds()
    nrag = nifty.graph.rag

    # list of all edge features we fill 
    feats = []

    # helper function to convert 
    # node features to edge features
    def nodeToEdgeFeat(nodeFeatures):
        uF = nodeFeatures[uv[:,0], :]
        vF = nodeFeatures[uv[:,1], :]
        feats = [ numpy.abs(uF-vF), uF + vF, uF *  vF,
                 numpy.minimum(uF,vF), numpy.maximum(uF,vF)]
        return numpy.concatenate(feats, axis=1)


    # accumulate features from raw data
    fRawEdge, fRawNode = nrag.accumulateStandartFeatures(rag=rag, data=raw,
        minVal=0.0, maxVal=255.0, numberOfThreads=1)
    feats.append(fRawEdge)
    feats.append(nodeToEdgeFeat(fRawNode))

    # accumulate data from pmap
    fPmapEdge, fPmapNode = nrag.accumulateStandartFeatures(rag=rag, data=pmap, 
        minVal=0.0, maxVal=1.0, numberOfThreads=1)
    feats.append(fPmapEdge)
    feats.append(nodeToEdgeFeat(fPmapNode))

    # accumulate node and edge features from
    # superpixels geometry 
    fGeoEdge = nrag.accumulateGeometricEdgeFeatures(rag=rag, numberOfThreads=1)
    feats.append(fGeoEdge)

    fGeoNode = nrag.accumulateGeometricNodeFeatures(rag=rag, numberOfThreads=1)
    feats.append(nodeToEdgeFeat(fGeoNode))

    return numpy.concatenate(feats, axis=1),numpy.concatenate([fRawNode, fPmapNode],axis=1)

def get_edge_gt(rag, overseg, binary_gt_image):
    # local maxima seeds
    seeds = nifty.segmentation.localMaximaSeeds(binary_gt_image)

    # growing map
    growMap = nifty.filters.gaussianSmoothing(1.0-binary_gt_image, 1.0)
    growMap += 0.1*nifty.filters.gaussianSmoothing(1.0-binary_gt_image, 6.0)
    gt = nifty.segmentation.seededWatersheds(growMap, seeds=seeds)


    # map the gt to the edges
    overlap = nifty.ground_truth.overlap(segmentation=overseg, 
                               groundTruth=gt)

    return overlap.differentOverlaps(rag.uvIds())


def get_overseg_rag_and_gt(raw, pmap, binary_gt_image):
    # oversementation
    overseg = nifty.segmentation.distanceTransformWatersheds(pmap, threshold=0.3)
    overseg -= 1

    # region adjacency graph
    rag = nifty.graph.rag.gridRag(overseg)

    # edge gt
    edge_gt = get_edge_gt(rag=rag, overseg=overseg,
        binary_gt_image=binary_gt_image)

    return overseg, rag, edge_gt



def get_overseg_rag(raw, pmap):
    # oversementation
    overseg = nifty.segmentation.distanceTransformWatersheds(pmap, threshold=0.3)
    overseg -= 1

    # region adjacency graph
    rag = nifty.graph.rag.gridRag(overseg)


    return overseg, rag
