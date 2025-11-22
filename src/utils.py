import numpy as np
import scipy as sp

# PATTERN
#  0 1 2
#  3 4 5
#  6 7 8


weights = 1/np.array([36, 9, 36, 9, 9, 9, 36, 9, 36])
weights[4] *= 4

e_i = np.array([
    [-1,-1],[0,-1],[1,-1],
    [-1, 0],[0, 0],[1, 0],
    [-1, 1],[0, 1],[1, 1]
])
e_i_abs=np.sqrt([2,1,2,1,7,1,2,1,2])

higher_sobel_x = np.transpose([
        [ -5, -4, 0,  4,  5],
        [ -8,-10, 0, 10,  8],
        [-10,-20, 0, 20, 10],
        [ -8,-10, 0, 10,  8],
        [ -5, -4, 0,  4,  5]
    ])
higher_sobel_y = np.transpose([
        [ -5, -8,-10, -8, -5],
        [ -4,-10,-20,-10, -4],
        [  0,  0,  0,  0,  0],
        [  4, 10, 20, 10,  4],
        [  5,  8, 10,  8,  5]
    ])

higher_scharr_x = np.transpose([
        [-1,-1, 0, 1, 1],
        [-2,-4, 0, 4, 2],
        [-3,-6, 0, 6, 3],
        [-2,-4, 0, 4, 2],
        [-1,-1, 0, 1, 1]
    ])
higher_scharr_y = np.transpose([
        [-1,-2,-3,-2,-1],
        [-1,-4,-6,-4,-1],
        [ 0, 0, 0, 0, 0],
        [ 1, 4, 6, 4, 1],
        [ 1, 2, 3, 2, 1]
    ])

higher_weights=[
    240,
    32+36
]

prewitt_x = np.transpose([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
prewitt_y = np.transpose([
    [-1,-1,-1],
    [ 0, 0, 0],
    [ 1, 1 ,1]
])
sobel_x = np.transpose([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
sobel_y = np.transpose([
    [-1,-2,-1],
    [ 0, 0, 0],
    [ 1, 2 ,1]
])
scharr_x = np.transpose([
        [ -3, 0,  3],
        [-10, 0, 10],
        [ -3, 0,  3]
    ])
scharr_y = np.transpose([
        [ -3,-10, -3],
        [  0,  0,  0],
        [  3, 10,  3]
    ])

kernels=[
    (prewitt_x, prewitt_y),
    (sobel_x, sobel_y),
    (scharr_x, scharr_y)
]
summed_weights=[
    6,
    8,
    32
]

def simple_gradient(array, repeats=None, kernels=(sobel_x, sobel_y), weight_sum=8):
    mode = ['nearest', 'nearest']
    if repeats:
        mode[0] = 'wrap' if repeats[0] else 'nearest'
        mode[1] = 'wrap' if repeats[1] else 'nearest'
    return np.stack((
        sp.ndimage.correlate(array, kernels[0], mode=mode[0], cval=0),
        sp.ndimage.correlate(array, kernels[1], mode=mode[1], cval=0)
    ), axis=2)/weight_sum

def simple_coupling(array):
    kernel = np.array([
        [ 1, 2, 1 ],
        [ 2, 0, 2 ],
        [ 1, 2, 1 ]
    ])[:,:,None] * np.transpose(e_i.reshape((3,3,-1)),  axes=(1,0,2))
    return np.stack((
        sp.ndimage.correlate(array, kernel[:,:,0], mode='nearest', origin=(0,0)),
        sp.ndimage.correlate(array, kernel[:,:,1], mode='nearest', origin=(0,0))
    ), axis=2)/8

def basic_stream(flow, ret=None):
    if not isinstance(ret, np.ndarray):
        ret = np.zeros_like(flow)
    w, h = flow.shape[:2]
    for i,v in enumerate(e_i):
        ret[
            max( v[0],0):w+min( v[0],0),
            max( v[1],0):h+min( v[1],0),
            i
        ] = flow[
            max(-v[0],0):w+min(-v[0],0),
            max(-v[1],0):h+min(-v[1],0),
            i
        ]
    return ret

def explicit_stream(flow, ret=None):
    if not isinstance(ret, np.ndarray):
        ret = np.zeros_like(flow)
    ret[:-1,:-1,0] = flow[1:,1:,  0]
    ret[:,:-1,  1] = flow[:,1:,   1]
    ret[1:,:-1, 2] = flow[:-1,1:, 2]
    ret[:-1,:,  3] = flow[1:,:,   3]
    ret[:,:,    4] = flow[:,:,    4]
    ret[1:,:,   5] = flow[:-1,:,  5]
    ret[:-1,1:, 6] = flow[1:,:-1, 6]
    ret[:,1:,   7] = flow[:,:-1,  7]
    ret[1:,1:,  8] = flow[:-1,:-1,8]

    return ret

#define a class to enable testing of internal data (self.trasfers)
class SimpleStreamer:
    def __init__(self, transferes):
        self.transferes = transferes
    def __call__(self, flow, ret):
        for fer in self.transferes:
            ret[*fer[0]] = flow[*fer[1]]
        return ret

def door(pnt):
    return slice(max(pnt, 0), min(pnt, 0) or None)

def gen_periodic_walls(wall_vec, automirror=False):
    if not isinstance(wall_vec[0], list):
        wall_vec = [wall_vec]
    transferes = []
    for vec in wall_vec:
        compare = abs(vec[0])+abs(vec[1]) - 0.5
        for i, v in enumerate(e_i):
            if np.sum(vec*v) > compare:
                # split by coord
                # pair of [target, source]         "if vec[] != 0" [target SLICE, source SLICE]
                x = [min(vec[0],0), min(-vec[0],0)] if vec[0] else [door(v[0]), door(-v[0])]
                y = [min(vec[1],0), min(-vec[1],0)] if vec[1] else [door(v[1]), door(-v[1])]

                transferes.append((
                    (x[0], y[0], i),
                    (x[1], y[1], i)
                ),)
                if automirror:
                    transferes.append((
                        (x[1], y[1], 8-i),
                        (x[0], y[0], 8-i)
                    ))

    return SimpleStreamer(transferes)

def gen_no_slip_walls(wall_vec):
    if not isinstance(wall_vec[0], list):
        wall_vec = [wall_vec]
    transferes = []
    for vec in wall_vec:
        compare = abs(vec[0])+abs(vec[1]) - 0.5
        in_cons = []
        out_cons = []
        for i, v in enumerate(e_i):
            if np.sum(vec*v) > compare:
                in_cons.append(8-i)
                out_cons.append(i)
        x = (slice(0,1) if vec[0] < 0 else slice(-1,None)) if vec[0] else slice(0,None)
        y = (slice(0,1) if vec[1] < 0 else slice(-1,None)) if vec[1] else slice(0,None)
        transferes.append((
            (x,y,tuple( in_cons)),
            (x,y,tuple(out_cons))
        ),)

    return SimpleStreamer(transferes)

def gen_velocity_boundary(wall_vec, vel):
    n_vel = -(wall_vec[0]*vel[0] + wall_vec[1]*vel[1])
    p_vel = -(abs(wall_vec[1])*vel[0] + abs(wall_vec[0])*vel[1]) #äh. hm. questionable.

    half,refl,known = [],[],[]

    for i, v in enumerate(e_i):
        switch = np.sum(v*wall_vec)
        if switch == 0:
            half.append(i)
        elif switch < 0:
            refl.append(i)
        elif switch > 0:
            known.insert(0,i)

    half = np.array(half)
    known = np.array(known)
    refl = np.array(refl)

    line = (
        #WORKAROUND! SHOULD be just min(-wall_vec[0/1],0)
        (slice(0,1) if wall_vec[0] < 0 else slice(-1,None)) if wall_vec[0] else slice(0,None),
        (slice(0,1) if wall_vec[1] < 0 else slice(-1,None)) if wall_vec[1] else slice(0,None)
    )

    def boundary_fulfilment(flow, ret):
        #DUE TO WORKAROUND:
        #keep in mind there will be a dim of length 1 in all selections!
        local_rho = np.sum(ret[*line,half] + 2*ret[*line,known], -1, keepdims=True) \
                      /  (1-n_vel) #ESSENTIAL! scales connecticals in sync with rho-gain!

        v_ = np.array([[[+1,-1]]])* ((ret[*line,half[2]] - ret[*line,half[0]])[:,:,None] + local_rho*p_vel)/2

        #HEAVILY relies on e_i-sorting-order!
        # center reflected   = center towards wall    + correction         workaround
        ret[*line,refl[1]]   = ret[*line,known[1]]   + 2/3*n_vel*local_rho[:,:,0]
        # skip in steps of 2 through `refl` and `known`, i.e. only use 1st and 3rd entry
        ret[*line,refl[::2]] = ret[*line,known[::2]] + 1/6*n_vel*local_rho + v_

        #workaround should NOT affect return-dims!
        return ret

    return boundary_fulfilment

def gen_density_boundary(wall_vec, p, hard_corners=True):

    half,refl,known = [],[],[]
    for i, v in enumerate(e_i):
        switch = np.sum(v*wall_vec)
        if switch == 0:
            half.append(i)
        elif switch < 0:
            refl.append(i)
        elif switch > 0:
            known.insert(0,i) #REVERSE order!

    half = np.array(half)
    known = np.array(known)
    refl = np.array(refl)

    side_range = slice(1,-1) if hard_corners else slice(0,None)

    line = (
        #WORKAROUND! SHOULD be just min(-wall_vec[0/1],0)
        (slice(0,1) if wall_vec[0] < 0 else slice(-1,None)) if wall_vec[0] else side_range,
        (slice(0,1) if wall_vec[1] < 0 else slice(-1,None)) if wall_vec[1] else side_range
    )

    corners = []

    for corner_vector in zip(
        [wall_vec[0]]*2 if wall_vec[0] else [1,-1],
        [wall_vec[1]]*2 if wall_vec[1] else [1,-1]
    ):
        c_refl = []
        c_half = []
        c_known = []
        for i, v in enumerate(e_i):
            switch = np.sum(v * corner_vector)
            if switch == 0 and i != 4:
                c_half.append(i)
            elif switch < 0:
                c_refl.append(i)
            elif switch > 0:
                c_known.insert(0, i)
        corners.append(((min(-corner_vector[0],0), min(-corner_vector[1],0)), c_refl,c_half,c_known),)

    def boundary_fulfilment(flow, ret):
        #DUE TO WORKAROUND:
        #keep in mind there will be a dim of length 1 in all selections!
        #ret[*line, :] = flow[*line, :]
        u_n = p - np.sum(ret[*line, half] + 2*ret[*line,known], -1)
        u_p = (ret[*line,half[2]] - ret[*line,half[0]])/2

        #HEAVILY relies on e_i-sorting-order!
        # center reflected = center towards wall  + correction
        ret[*line,refl[1]] = ret[*line,known[1]] + 2/3*u_n
        # skip in steps of 2 through `refl` and `known`, i.e. only use 1st and 3rd entry
        ret[*line,refl[0]] = ret[*line,known[0]] + 1/6*u_n + u_p
        ret[*line,refl[2]] = ret[*line,known[2]] + 1/6*u_n - u_p
        #                                 u_p needs to be sorted congruent with `refl` and `known`

        return ret

    if not hard_corners:
        return boundary_fulfilment

    def hard_corner_fulfilment(flow, ret):
        ret = boundary_fulfilment(flow, ret)
        for c in corners:
            ret[*c[0], c[2]] = p/2 - (ret[*c[0],4]/2 + np.sum(ret[*c[0],c[3]]))
            ret[*c[0], c[1]] = ret[*c[0],c[3]]
        return ret
    return hard_corner_fulfilment
