import numpy as np
from .utils import *

def vel_distrib(u):
    #Getting this right is IMPORTANT! Can lead to very hard to debug behavior!
    eiu = np.sum(u[:,:,None,:] * e_i[None, None, :,:], 3)
    return (3*eiu + 9/2*eiu**2 - 3/2*np.sum(u**2,-1,keepdims=True))  * weights

class Flat_LBM:

    weights = 1/np.array([36, 9, 36, 9, 9, 9, 36, 9, 36])
    weights[4] *= 4

    e_i = np.array([
        [-1,-1],[0,-1],[1,-1],
        [-1, 0],[0, 0],[1, 0],
        [-1, 1],[0, 1],[1, 1]
    ])
    e_i_abs=np.sqrt([2,1,2,1,7,1,2,1,2])

    ephemerals=[]

    def __init__(self):
        self._boundaries=[]
        self._state = np.zeros((3,3,9),)
        self._reset()

    def _reset(self):
        for attr in self.ephemerals:
            setattr(self, attr, None)

    def __call__(self, flow):
        self.state = flow
        self._reset()
        return self

    def _just_stream(self, flow):
        ret = explicit_stream(flow)
        for condition in self._boundaries:
            ret = condition(flow, ret)
        return ret

    def test_conditions(self):
        flow = np.ones_like(self.state)
        ret = np.full_like(self.state, np.nan)
        ret = explicit_stream(flow, ret)
        for condition in self._boundaries:
            ret = condition(flow, ret)
        assert not np.isnan(ret).any(), f"Not all connecticals were set a value: {np.where(np.isnan(ret))}"
        return True

    def stream(self, relaxed=None):
        self._reset()
        relaxed = self.state if not isinstance(relaxed, np.ndarray) else relaxed
        self.state = self._just_stream(relaxed)
        return self.state

    def add_transferes(self, transferes):
        if callable(transferes):
            self._boundaries.append(transferes)
        else:
            assert len(transferes[0][0])
            self._transferes += transferes

class Compressible_Flow(Flat_LBM):

    ephemerals = Flat_LBM.ephemerals + ['_rho']

    def __init__(self, c=1, relax_time=1, nu=None):
        super().__init__()
        self._rt = nu*3+0.5 if nu else relax_time
        self.nu = nu if nu else 1/6*(2*relax_time-1)
        self.c = c

    def __sub__(self, other):
        return Immiscib_LBM(self, other)

    def relax(self, equilibrated):
        return self.state - 1/self._rt * equilibrated

    def _get_vel_distrib(self, flow):
        return vel_distrib(self._get_vel(flow)/self.rho[:,:,None])

    def _get_vel(self, flow=None):
        flow = self.state if not isinstance(flow, np.ndarray) else flow
        return np.sum(flow[:,:,:,None]*e_i[None,None,:,:],2)

    @property
    def rho(self):
        # _rho is ephemeral
        if not isinstance(self._rho, np.ndarray):
            self._rho = np.sum(self.state,2)
        return self._rho

    def base_equilibrate(self, flow=None):
        return (weights + self._get_vel_distrib(flow)) * self.rho[:,:,None]

class Immiscib_LBM(Compressible_Flow):

    ephemerals = Compressible_Flow.ephemerals + ['_vof', '_phasegrad', '_unitgrad','_phase_norm', '_relax', '_curvature']

    def __init__(self, phase1, phase2):
        super().__init__(relax_time=phase1._rt) #flow is now flow_1+flow_2
        self._r2 = phase2._rt
        self.state = phase1.state+phase2.state
        self.diff =  (phase1.state-phase2.state)/2
        self.beta = 0.999
        self.x_repeat = False
        self.y_repeat = False

    def grad(self, field, fltr=2):
        return simple_gradient(field, (self.x_repeat, self.y_repeat), kernels[fltr], summed_weights[fltr])

    def fine_grad(self, field):
        return simple_gradient(field, (self.x_repeat, self.y_repeat), (higher_scharr_x, higher_scharr_y), 32+36)

    def stream(self, relaxed=None, diff=None):
        super().stream(relaxed)

        diff = self.diff if not isinstance(diff, np.ndarray) else diff
        assert isinstance(diff, np.ndarray)
        self.diff = self._just_stream(diff)
        return self.state, self.diff

    @property
    def VOF(self):
        if not isinstance(self._vof, np.ndarray):
            self._vof = np.sum(self.diff,-1)/self.rho
            #self._vof[self._vof<0] = -1/2
            #self._vof[self._vof>0] = 1/2
        return self._vof

    def _set_gradients(self):
        self._phasegrad = self.grad(self.VOF) #/2 already part of VOF

        self._phase_norm = np.sqrt(np.sum(self._phasegrad**2,2))
        self._phase_norm[self._phase_norm==0]=1
        self._unitgrad = self._phasegrad/self._phase_norm[:,:,None]

    @property
    def phase_grad(self):
        if not isinstance(self._phasegrad, np.ndarray):
            self._set_gradients()
        return self._phasegrad

    @property
    def unit_grad(self):
        if not isinstance(self._unitgrad, np.ndarray):
            self._set_gradients()
        return self._unitgrad

    @property
    def phase_norm(self):
        if not isinstance(self._phase_norm, np.ndarray):
            self._set_gradients()
        return self._phase_norm

    @property
    def interface_curvature(self):
        if not isinstance(self._curvature, np.ndarray):
            #tilting the gradient to the left allows for a more efficient calculation. See below for original.
            #by tilting via transpose and retranspose, arithmatic is avoided, only array-views are used.
            tilted_grad_x = self.grad(self.phase_grad[:,::-1,0].transpose(),1) .transpose((1,0,2),)[:,::-1,:]
            tilted_grad_y = self.grad(self.phase_grad[:,::-1,1].transpose(),1) .transpose((1,0,2),)[:,::-1,:]
            #additionally, phase_grad may be tilted, making formula more similar to pressure-jump. But requirese additional compute.
            scalar_curvature = np.sum(self.phase_grad*(
                self.phase_grad[:,:,0,None]*tilted_grad_y - self.phase_grad[:,:,1,None]*tilted_grad_x
            ),-1)
            #original calculation
            #phase_grad_x = self.grad(self.phase_grad[:,:,0])
            #phase_grad_y = self.grad(self.phase_grad[:,:,1])
            #scalar_curvature    = self.phase_grad[:,:,0]*self.phase_grad[:,:,1]*(phase_grad_x[:,:,1] + phase_grad_y[:,:,0]) \
            #                    - self.phase_grad[:,:,1]**2*phase_grad_x[:,:,0] \
            #                    - self.phase_grad[:,:,0]**2*phase_grad_y[:,:,1]
            #scalar_curvature[self.phase_norm==1]=0
            self._curvature = self.phase_grad * (scalar_curvature/self.phase_norm**3)[:,:,None] 
        return self._curvature

    def normal_pressure_jump(self):
        vel = np.sum(self.state[...,None]*e_i[None,None,:,:],2)/self.rho[:,:,None]
        grad_v_x = self.grad(vel[:,:,0])
        grad_v_y = self.grad(vel[:,:,1])

        du_dnn = (np.sum(grad_v_x*self.phase_grad,-1)*self.phase_grad[:,:,0] + np.sum(grad_v_y*self.phase_grad,-1)*self.phase_grad[:,:,1])/self.phase_norm**2
        #                                              |                mu       c_s^2         |
        return -(2*self.phase_grad*du_dnn[:,:,None]) * ((self.relax-0.5)*self.rho /3 )[:,:,None]

    def full_pressure_jump(self, vel=None):
        vel = vel if isinstance(vel, np.ndarray) else np.sum(self.state[...,None]*e_i[None,None,:,:],2)/self.rho[:,:,None]
        #need *smoothed* gradient to avoid oscillations
        grad_v_x = self.fine_grad(vel[:,:,0])
        grad_v_y = self.fine_grad(vel[:,:,1])
        n = self.phase_grad
        t = np.zeros_like(n)
        #orientation doesn't matter, as t is multiplied twice!
        t[:,:,0]=-n[:,:,1]
        t[:,:,1]=n[:,:,0]

        du_dnn = (np.sum(grad_v_x*n,-1)*n[:,:,0] + np.sum(grad_v_y*n,-1)*n[:,:,1])/self.phase_norm**2
        #each term is multiplied by t ONCE
        du_dnt = (np.sum(grad_v_x*n,-1)*t[:,:,0] + np.sum(grad_v_y*n,-1)*t[:,:,1])/self.phase_norm**2
        du_dtn = (np.sum(grad_v_x*t,-1)*n[:,:,0] + np.sum(grad_v_y*t,-1)*n[:,:,1])/self.phase_norm**2
        #ssound = (0.5+self.VOF) + 1/self._r[0] * (0.5-self.VOF)
        #                                                             |                mu       c_s^2         |
        #t*(du_dtn+du_dnt)[:,:,None] +
        #all multiplied by t for the second time:
        return - (t*(du_dnt+du_dtn)[:,:,None] + 2*n*du_dnn[:,:,None]) * ((self.relax-0.5)*self.rho /3 )[:,:,None]

    @property
    def relax(self):
        if not isinstance(self._relax, np.ndarray):
            #only calc relax-time, first line is linear interp of *mu*, second of relax-time itself
            self._relax = (self._rt-0.5)*(self._r2-0.5)/(self._r2*(0.5+self.VOF) + self._rt*(0.5-self.VOF) - 0.5) + 0.5
            #self._relax = (self._rt+self._r2)/2 + (self._rt - self._r2)*self.VOF
            #more accurate, but more expensive
            #self._relax = (self._rt**(0.5+self.VOF)*self._r2**(0.5-self.VOF))
        return self._relax

    def __call__(self,flow, diff):
        self.diff = diff
        return super().__call__(flow)

    @property
    def colorcos(self):
        return np.sum(self.unit_grad[:,:,None,:]*e_i[None,None,:,:],3)/e_i_abs[None,None,:] # |unit_grad| == 1

    def diff_equilibrate(self, relaxed_flow):
        #relaxed_flow = relaxed_flow if isinstance(relaxed_flow, np.ndarray) else super().equilibrate(self.state)

        #set diff to relaxed_flow 
        diff= self.VOF[:,:,None]*relaxed_flow \
            + self.beta *(1/4-self.VOF[:,:,None]**2)*self.colorcos*(self.rho[:,:,None]*weights) #then recolor according to color-grad
        return relaxed_flow, diff






class DissimilarDensities_LBM(Immiscib_LBM):

    ephemerals = Immiscib_LBM.ephemerals + ['_ssound','_ratio_corrections']

    def __init__(self, phase1, phase2, rho1_to_rho2):
        super().__init__(phase1, phase2)
        self._r = np.ones(9) * rho1_to_rho2
        self._r[4] = (9-5*rho1_to_rho2)/4
        #correct relaxation-time of fluid2
        self._r2 = (self._r2-0.5)*rho1_to_rho2 + 0.5

    def set_dens_rat(self, rat):
        self._r = np.ones(9)*rat
        self._r[4] = (9-5*rat)/4
        self._r2 = (self._r2-0.5)*rat + 0.5

    @property
    def ratio_corrections(self):
        if not isinstance(self._ratio_corrections, np.ndarray):
            self._ratio_corrections = (1+self._r)*0.5 + (1-self._r)*self.VOF[:,:,None]
        return self._ratio_corrections

    def base_equilibrate(self, flow=None):
        flow = self.state if not isinstance(flow, np.ndarray) else flow
        u = self._get_vel(flow)/self.rho[:,:,None]
        eiu = np.sum(u[:,:,None,:] * e_i[None, None, :,:], -1)
        vel_distrib= (3*eiu + 9/2*eiu**2 - 3/2*np.sum(u**2,-1,keepdims=True))  * weights
        shear_correction = self.density_shear_correction(u)

        return ( self.ratio_corrections*weights + vel_distrib)  *self.rho[:,:,None] + shear_correction

    def get_body_force_connecticals(self, force):
        return np.sum(force[:,:,None,:] * e_i[None,None,:,:], -1)*weights[None,None,:]*3*self.ratio_corrections

    def density_shear_correction(self, vel):
        rhograd = self.grad(self.rho)
        pregrid = vel[:,:,:,None] * rhograd[:,:,None,:]
        pregrid += np.transpose(pregrid, (0,1,3,2))
        #             to be in congruence with papers: /8 here
        tensor_product = e_i[:,:,None] * e_i[:,None,:] /8

        #after adding pregrid's transpose, edges are twice required value!
        tensor_product[4,:,:]          = [[-3/2,0],[0,-3/2]] #setting [0,0] and [1,1] values
        tensor_product[(1,3,5,7),:,:] *= 4
        #tensor_product[(0,2,6,8),:,:] *= 1 #just for emphasis
        correction = np.sum(tensor_product[None,None,:,:,:] * pregrid[:,:,None,:,:], (3,4))
        #shouldn't this factor be sound-dependent? :S
        correction *= ((self.relax-0.5)/3)[:,:,None]
        return correction

    def diff_equilibrate(self, relaxed_flow):
        #set diff to relaxed_flow 
        diff= self.VOF[:,:,None]*relaxed_flow \
            + self.beta *(1/4-self.VOF[:,:,None]**2)*self.colorcos*(self.rho[:,:,None]*weights*self.ratio_corrections) #then recolor according to color-grad
        return relaxed_flow, diff

    @property
    def ssound(self):
        if not isinstance(self._ssound, np.ndarray):
            self._ssound = 3*3*self._r[0]/(3*(0.5-self.VOF) + 3*self._r[0]*(0.5+self.VOF))
        return self._ssound
