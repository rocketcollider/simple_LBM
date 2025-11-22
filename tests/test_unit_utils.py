from ..src.utils import *
import numpy as np

#replicate constants to ensure tests are independent!
weights = 1/np.array([36, 9, 36, 9, 9, 9, 36, 9, 36])
weights[4] *= 4

e_i = np.array([
    [-1,-1],[0,-1],[1,-1],
    [-1, 0],[0, 0],[1, 0],
    [-1, 1],[0, 1],[1, 1]
])
e_i_abs=np.sqrt([2,1,2,1,7,1,2,1,2])

def indx_of_npa(lst, npa):
    for i, elm in enumerate(lst):
        if np.array_equal(elm.shape, npa.shape) and  np.allclose(elm, npa):
            return i
    return -1

def test_single_periodic_wall():
    """
    Define bottom (maximum y) as periodic with ceiling (minimum y).
    Expect the edges to be lost! This is due to the nature of the periodic boundary.
    Connecticals of target and source should be the same! (A connectical nr. 3 streams into a connectical nr. 3)
    """
    test_grid = np.arange(1,5)[:,None,None]*np.arange(1,4)[None,:,None]*np.arange(0.1,1,0.1)[None,None,:]
    selector = gen_periodic_walls([[0,1]]).transferes
    #this one is technically implementation-specific. automirroring could change this!
    assert len(selector) == 3, "# of selectors wrong. This is most likely an implementation-change, update test accordingly!"
    #this one is definitional!
    assert len(selector[0]) == 2, "selector should consist of ([target cells], [source cells]) ==2"
    assert len(selector[0][0]) == 3, "need to contain selection for all 3 dimension of flow (3rd is connectionals)"

    #upper rim (target) cons are 0.7,0.8,0.9
    #expected target slices are :-1, :, 1: ; 0
    #expected source slices are 1:, :, :-1 ; -1
    #lower rim (source) cons are 0.7,0.8,0.9
    #total selections are therefore:
    targets = [0.7*np.arange(1,4), 0.8*np.arange(1,5), 0.9*np.arange(2,5)]
    sources = [0.7*np.arange(1,4)*4, 0.8*np.arange(1,5)*4, 0.9*np.arange(2,5)*4]

    idx1 = indx_of_npa(targets, test_grid[*selector[0][0]])
    assert idx1 > -1, "first target selector returned unexpected range"
    idx2 = indx_of_npa(targets, test_grid[*selector[1][0]])
    assert idx2 > -1, "second target selector returned unexpected range"
    idx3 = indx_of_npa(targets, test_grid[*selector[2][0]])
    assert idx3 > -1, "third target selector returned unexpected range"

    assert idx1 != idx2, "selectors 1 and 2 are the same"
    assert idx2 != idx3, "selectors 2 and 3 are the same"
    assert idx3 != idx1, "selectors 1 and 3 are the same"

    sdx1 = indx_of_npa(sources, test_grid[*selector[0][1]])
    assert idx1 > -1, "first target selector returned unexpected range"
    sdx2 = indx_of_npa(sources, test_grid[*selector[1][1]])
    assert idx2 > -1, "second target selector returned unexpected range"
    sdx3 = indx_of_npa(sources, test_grid[*selector[2][1]])
    assert idx3 > -1, "third target selector returned unexpected range"

    assert idx1 != sdx1, "target and source mismatch"
    assert idx2 != sdx2, "target and source mismatch"
    assert idx3 != sdx2, "target and source mismatch"

def test_periodic_corner():
    """
    Define lower right corner as periodic corner with upper left.
    Expect only a single connectical in normal operation! All other connectical are already set in
    periodic wall boundaries.
    """
    test_grid = np.arange(1,5)[:,None,None]*np.arange(1,4)[None,:,None]*np.arange(0.1,1,0.1)[None,None,:]
    selector = gen_periodic_walls([[1,1]]).transferes
    #this one is technically implementation-specific. automirroring could change this!
    assert len(selector) == 1, "# of selectors wrong. This is most likely an implementation-change, update test accordingly!"
    #this one is definitional!
    assert len(selector[0]) == 2, "selector should consist of ([target cells], [source cells]) ==2"
    assert len(selector[0][0]) == 3, "need to contain selection for all 3 dimension of flow (3rd is connectionals)"

    #upper rim (target) cons are 0.7,0.8,0.9
    #expected target slices are :-1, :, 1: ; 0
    #expected source slices are 1:, :, :-1 ; -1
    #lower rim (source) cons are 0.7,0.8,0.9
    #total selections are therefore:
    x = np.ones(1)[0]

    assert test_grid[*selector[0][0]] == 0.9*x, "target selector returned unexpected range"
    assert test_grid[*selector[0][1]] == 0.9*x*12, "source selector returned unexpected range"

def test_noslip_wall():
    """
    No-slip boundary condition simply reassings connecticals.
    Important is to change the direction and therefore connecticals to match! Similarly the correct set
    of connecticals needs to be used for rearrangement. However, the particular order is actually unimportant.
    """
    test_grid = np.arange(1,5)[:,None,None]*np.arange(1,4)[None,:,None]*np.arange(0.1,1,0.1)[None,None,:]

    selector = gen_no_slip_walls([[0,1]]).transferes

    assert len(selector) == 1, "expected single wall == single selector"
    #definitional!
    assert len(selector[0]) == 2, "selector should consist of ([target cells], [source cells]) ==2"
    assert len(selector[0][0]) == 3, "need to contain selection for all 3 dimension of flow (3rd is connectionals)"

    assert set(selector[0][0][2]) == set((0,1,2),)
    assert set(selector[0][1][2]) == set((8,7,6),)
    trg = np.array(selector[0][0][2])
    src = np.array(selector[0][1][2])
    assert np.all((trg+src) == 8), "Only works for specifically aranged D2Q9"
    target = test_grid[:,-1:,selector[0][0][2]] #order is allowed to change
    source = test_grid[:,-1:,selector[0][1][2]] #but MUST me in reverse to source!

    assert (test_grid[selector[0][0]] == target).all(), f"TARGET selector returend unexpected range:{selector[0][0][2]}"
    assert (test_grid[selector[0][1]] == source).all(), f"SOURCE selector returend unexpected range:{selector[0][0][2]}"

def test_const_v_wall_down():
    #                                    wall   velocity
    to_be_tested = gen_velocity_boundary([0,1], [5,0.5])
    test_grid = np.arange(1,6)[:,None,None]*np.arange(1,5)[None,:,None]*np.arange(0.1,1,0.1)[None,None,:]
    ret = basic_stream(test_grid)
    zeros = np.zeros_like(test_grid)
    test = to_be_tested(ret, ret)
    assert test.shape == (5,4,9), "This test should never fail!!!"

    test_vels = test[:,-1,:,None]*e_i[None,:,:]
    test_rho = np.sum(test[:,-1,:],-1)

    reduced = np.sum(test_vels,1)/test_rho[:,None]
    assert np.allclose(reduced, (5,0.5))

def test_const_v_wall_up():
    """
    Technically nonsensically, as a vel-boundary can't force constant outflow of material.
    However, a good test to see if orientation / sign of velocity is respected.
    """
    #                                     wall   velocity
    to_be_tested = gen_velocity_boundary([0,-1], [5,0.5])
    test_grid = np.arange(1,6)[:,None,None]*np.arange(1,5)[None,:,None]*np.arange(0.1,1,0.1)[None,None,:]
    ret = basic_stream(test_grid)
    zeros = np.zeros_like(test_grid)
    test = to_be_tested(ret, ret)
    assert test.shape == (5,4,9), "This test should never fail!!!"

    test_vels = np.sum(test[:,0,:,None]*e_i[None,:,:], 1) # 
    test_rho = np.sum(test[:,0,:],-1)

    assert np.allclose(test_vels/test_rho[:,None], (5,0.5))

def test_const_pressure_down():
    to_be_tested = gen_density_boundary([0,1], 0.5)

    test_grid = np.arange(1,6)[:,None,None]*np.arange(1,5)[None,:,None]*np.arange(0.1,1,0.1)[None,None,:]
    ret = basic_stream(test_grid)
    zeros = np.zeros_like(test_grid)
    test = to_be_tested(ret, ret)
    assert test.shape == (5,4,9), "This test should never fail!!!"

    test_rho = np.sum(test[:,-1,:],-1)
    assert np.allclose(test_rho, 0.5)