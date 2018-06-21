#################################################################
#                                                               #
#   KNOT UTILITIES                                              #
#   ==============                                              #
#   ( knut.py )                                                 #
#   first instance: 20180621                                    #
#   written by Stefanos Kourtis ( kourtis@bu.edu )              #
#                                                               #
#   Auxiliary routines for reading and converting between       #
#   representations of knots and evaluation of basic knot       #
#   properties.                                                 #
#                                                               #
#   DEPENDENCIES: numpy                                         #
#                                                               #
#################################################################

import sys
sys.dont_write_bytecode = True  # Do not write compiled bytecode
from numpy import *

def writhe(x,return_signs=False):
    """ Return writhe of planar code x for len(x)>1. """
    # Since planar code contains only "under" crossings,
    # this should yield handedness sign for each one:
    sg = array([c[1]-c[3] for c in x])
    # Make sure to treat the last arc properly:
    sg[abs(sg)>1] = -sign(sg[abs(sg)>1])
    if ( return_signs ): return sum(sg),sg
    return sum(sg)

def pd2tait(x):
    """ Return the Tait graph of planar code x.
        Disregards handedness information. """
    gc = pd2gc(x)
    tg = gc2tait(gc)
    return tg

def pd2gc(X):
    """ Convert planar data code to Gauss code.
        Disregards handedness information. """
    x  = array(X)               # If type(X) != ndarray
    x  = x[argsort(x[:,0])]     # If X unordered
    na = x.max()                # Number of arcs
    ai = arange(1,na+1)         # Arc indices
    # Get crossing indexing from arc indexing.
    # Arc indexing is redundant: make the first na/2
    # "incoming" arcs representatives of the rest,
    # find "representees" and put them in cp, then
    # replace representees with representatives in ai
    cp = (x[:,1::2]%na).min(1)  # Modulo for last arc
    cp[cp==0] = na              # Undo modulo
    ai[cp-1] = x[:,0]           # Replace
    # Contiguously relabel representatives as in ix
    _,ix = unique(ai,return_inverse=True) 
    # Arcs not appearing as first indices in any X
    # symbol stand for "over" crossings: give them
    # opposite Gauss code sign, merge, and order:
    mc = array(list(set(range(1,na+1))-set(x[:,0])))
    aa = concatenate((x[:,0],-mc))
    sg = sign(aa[argsort(abs(aa))])
    gc = sg*(ix+1)
    return gc

def gc2tait(gc):
    """ Return Tait graph from Gauss code. """
    nc = abs(array(gc)).max() # Number of crossings
    # Tait signs of edges are determined by "skips"
    # w.r.t. fully alternating over/under sequence:
    sg = (-1)**(abs(sign(gc)-array([1,-1]*nc))/2)
    # Get crossing indices in order of appearance:
    _,od = unique(abs(array(gc)),return_index=True)
    ix = list(abs(array(gc))[sort(od)])
    sg = sg[sort(od)]         # Re-sort Tait signs
    oc = zeros(nc,int)        # Edge traversal flag
    el = []                   # Edge list
    nv = 0                    # Vertex counter
    bf = 0                    # Buffer counter
    # Gauss code of knot effectively describes an
    # Eulerian cycle over the Tait graph with each
    # edge doubled. Appearence of a number in the
    # Gauss code describes traversal of an edge,
    # sign defines which of the two edges between
    # two vertices is traversed. We use this to
    # reconstruct the Tait graph:
    for i,c in enumerate(gc[:-1]):
        c0 = abs(c)-1         # Current edge index
        c1 = abs(gc[i+1])-1   # Next edge index
        # If current edge has been visited before
        # then edge has been recorded and we are
        # now traversing "backwards", so put origin
        # vertex in buffer:
        if ( oc[c0] == 1 ):
            bf = el[ix.index(c0+1)][0]
            continue
        # If next edge has not been visited, record
        # "forward" traversal of current edge and
        # put destination vertex in buffer:
        if ( oc[c1] == 0 ):
            el = el + [[bf,nv+1]]
            nv = nv+1
            bf = nv
        # If next edge has been visited, record
        # "forward" traversal of current edge and
        # put origin vertex of next edge in buffer:
        else:
            el = el + [[bf,el[ix.index(c1+1)][1]]]
        oc[c0] = 1  # Record current edge traversal
    ea = array(el)+1          # 1-based edgelist
    return (ea.T*sg).T        # Signed edgelist
