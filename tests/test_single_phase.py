from ..src.utils import *
from ..src.LBM import *
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

def test_simple_poseuille():
    """
    This test runs fluid through a pipe and evaluates the velocity-cross-section.
    To avoid any complex ability, such as velocity/pressure boundaries, a gravity-like
    force is added at every step, until the wall-friction and fluid-viscosity is equal
    to the gravity-force.
    The flow is laminar in the equilibrium. This system is described by Hagen–Poiseuille
    equation: https://en.wikipedia.org/wiki/Hagen-Poiseuille_equation
    """

    #variable test parameters
    simsteps = 5000
    g = 0.001
    w = 32
    h = 16
    rho = 1
    visualisation = False
    err = []

    #simulator setup
    sim = Compressible_Flow()
    sim.add_transferes(gen_no_slip_walls([[1,0],[-1,0]]))
    sim.add_transferes(gen_periodic_walls([[0,1],[0,-1]]))

    grid = np.ones((w,h,1),) * rho

    #keep in mind half a cell at the borders!
    uy_initial = (np.arange(w))*(w-1-np.arange(w))* g

    if visualisation:
        plt.ion()
    #set up for error-image!
    fig, axes = plt.subplots(nrows=3,ncols=1, figsize=(8,4))
    axes[0].set_ylabel('vel')
    axes[0].set_xlabel(r'$v_x$')
    axes[0].set_title('Verifying the model')
    sc_v = axes[0].scatter(np.zeros(w), np.arange(w),label='LBM')
    axes[0].plot(uy_initial, np.arange(w),'k', label='analytical')
    axes[2].set_xlabel('# run')
    axes[2].set_ylabel(r'$v_{err}$')
    axes[2].set_xlim([0,simsteps])
    sc_err, = axes[2].plot([0,0],[simsteps,0],"k")
    img = axes[1].imshow(np.ones((w,h),),vmin=0,vmax=1)#, cmap='jet')
    fig.colorbar(img)
    #plt.tight_layout()

    #set initial simulator data
    sim(grid*sim.weights)
    error_exit = False

    for n in range(simsteps):
        equilibrated = sim.equilibrate()
        sim.stream(equilibrated + g*sim.rho[:,:,None]*sim.weights*np.sum(e_i * [[0,1]],-1))
        uy_eval = np.sum(equilibrated[:,:,:,None]*e_i[None,None,:,:],2)[:,:,1]
        err.append(np.sum((uy_initial[:,None] - uy_eval)**2)**0.5)

        if False and( n!=0 and err[-1] > err[-2]):
            bad_apples = 0
            if err[-1] - err[-2] > err[-2]:
                error_exit = True
            else:
                for eror, preror in zip(err[-2:-min(9,len(err)-1):-1], err[-3:-min(10,len(err)):-1]):
                    if eror > preror:
                        bad_apples += 1
                if bad_apples>3:
                    error_exit=True

        if visualisation or error_exit:
            if n==0 or error_exit:
                axes[2].set_ylim(0,err[0])
            u_vec = np.sum(np.sum(equilibrated[:,:,:,None]*e_i[None,None,:,:],2),1)/equilibrated.shape[1]
            sc_v.set_offsets(np.stack((u_vec[:,1], np.arange(32)),1))
            sc_err.set_xdata(np.arange(len(err)))
            sc_err.set_ydata(err)
            print(err[-1])
            img.set_data(np.sum(equilibrated[:,:,:]*e_i[None,None,:,1],2) / sim.rho)
            fig.canvas.flush_events()
            if error_exit:
                with open("test_simple_poseuille_errors.pkl", "wb") as pkl:
                    pickle.dump(err, pkl)
                plt.savefig("poseuille_error.png")
                assert False, f"Error behaving unexpectedly in last 10 steps. Last step: {str(n)}, errors stored in test_simple_poseuille_errors.pkl"

def test_moving_boundary():
    """
    Just a box with one conveyor-belt side opposite a no-slip-barrier.
    Laminar in the equilibrium. Mainly test of moving-boundary boundary-condition.
    """

    #variable test parameters
    simsteps = 6000
    g = 0.005
    w = 64
    h = 64
    wall_v = 0.5
    initial_offset_factor = .9 #introduce perturbation in velocit-field
    rho = 1
    visualisation = False
    err = []

    #simulator setup
    sim = Compressible_Flow()
    sim.add_transferes(gen_periodic_walls([[1,0],[-1,0]]))
    sim.add_transferes(gen_no_slip_walls([[0,-1]]))
    assert callable(gen_velocity_boundary([0,1],[0.5,0]))
    sim.add_transferes(gen_velocity_boundary([0,1],[0.5,0]))

    grid = np.ones((w,h,1),) * rho

    temparation = np.arange(h)/(h-1)
    exp_v = (wall_v*temparation)[None,:] 
    v = exp_v[:,:,None]*initial_offset_factor
    eiu = v * e_i[None, None, :,0]
    v_distrib = (1+ 3*eiu + 9*eiu**2/2 - 3*v**2/2)
    initial_data = v_distrib*weights * grid

    sim(initial_data)

    if visualisation:
        plt.ion()
    fig, axes = plt.subplots()
    img = axes.imshow(np.ones((w,h),),vmin=-0.03,vmax=0.03)#, cmap='jet')
    fig.colorbar(img)

    err = []

    for n in range(simsteps):
        equilibrated = sim.equilibrate()
        sim.stream(equilibrated)
        ux_eval = np.sum(sim.state[:,:,:]*e_i[None,None,:,0],2)/sim.rho
        error=float(np.sum((ux_eval - exp_v)**2)**0.5 or 1)
        err.append(error)

        img.set_data(ux_eval - exp_v)
        fig.canvas.flush_events()

    if err[-1]*10 > err[0]:
        with open("test_moving_boundary_errors.pkl", "wb") as pkl:
            pickle.dump(err, pkl)

        assert err[-1] * 10 < err[0], f"Prediction was not met after {simsteps}. Errors stored in test_moving_boundary_errors.pkl"

def test_pressure_poseuille():
    """
    A pipe with pressure differential applied from both sides.
    Pipe is vertical (i.e. extends in y-direction), but visualisations
    are horizontal to better use visualisation-space.

    At P2/P1 > 4 simulation breaks down.
    Set the vmin/vmax values for the imshow appropriately.

    The exact Hagen-Poseuille-solution is valid for incompressible fluids.
    A change in density across the length of the pipe can be moddeled as 
    exponentially expanding (such that rho(y)/rho(y-1) == const).
    Both ends release a pressure wave while equilibriating at the beginning
    of the simulation.
    A small perturbation from optimal velocity distribution can be
    introduced to test for good equilibration and more stable
    error-reduction-test.
    """

    #variable test parameters
    visualisation = False
    simsteps = 6000
    measure_step = 2700 #depends on pipe-length, select moment of low reflections
    w = 32
    h = 1024        #pipe-lenght
    window = 56,105 #highly dependent on pipe-length!
    initial_offset_factor = .9 #introduce perturbation in velocit-field

    #physical system parameters
    #these are PRESSURE-values!
    #need to devide by speed_of_sound**2 (usually 1/3)
    p1=1 #exit pressure
    p2=4.0001 #entrance pressure

    measure_window = slice(*window)
    compare_window = slice(*(np.array(window)+1))

    #kinematic viscosity
    nu = 1/6*(2*1-1) #multiply by density to get dynamic viscosity mu
    err = []

    #simulator setup
    sim = Compressible_Flow()
    sim.add_transferes(gen_no_slip_walls([[1,0],[-1,0]]))

    #/////////////// actual test subject \\\\\\\\\\\\\\\#
    sim.add_transferes(gen_density_boundary([0, 1],p1*3),)
    sim.add_transferes(gen_density_boundary([0,-1],p2*3),)
    #\\\\\\\\\\\\\\\ actual test subject ///////////////#

    #basic velocity profile. Still needs pressure-gradient and 1/rho multiplication.
    expected_crosssection = (np.arange(w))*(w-1-np.arange(w)) /2/nu

    #a bit overkill. In fact, linear increasing density gives more overt sound-vibrations.
    temparation = (p1/p2)**(np.arange(1,h+1)/(h-1))

    #                         approx pressue drop      avg  density
    uy_initial = expected_crosssection * (p2-p1)/(h-1)   *  2 /( (p1+p2)*3 ) #*3 to convert P to rho

    #      prepare density      as constant density gradient across length
    grid = np.ones((w,h,9),) * p2*3 * temparation[None, :, None]
    # (p2 - np.arange(h)/(h-1)*(p2-p1))[None,:,None]*3 #*3 to convert P to rho

    v = uy_initial[:,None]*initial_offset_factor *temparation[None,:]
    eiu = v[:,:,None] * e_i[None, None, :,1]
    v_distrib = (1+ 3*eiu + 9*eiu**2/2 - 3*v[:,:,None]**2/2)

    #set initial simulator data
    sim(grid * v_distrib  * weights)

    #prepare evaluation
    if visualisation:
        plt.ion()

    #set up for error-image
    fig, axes = plt.subplots(nrows=3,ncols=1, figsize=(8,4))
    axes[0].set_ylabel('vel')
    axes[0].set_xlabel(r'$v_y$')
    axes[0].set_title('Verifying the model')
    sc_v = axes[0].scatter(np.zeros(w), np.arange(w),label='LBM')
    #np.average(np.sum(sim_data*e_i[None,None,:,1],2)/np.sum(sim_data,2), 1)
    exp_plot = axes[0].plot(uy_initial, np.arange(w),'k', label='analytical')[0]
    axes[2].set_xlabel('# run')
    axes[2].set_ylabel(r'$v_{err}$')
    axes[2].set_xlim([0,simsteps])
    sc_err, = axes[2].plot([0,0],[simsteps,0],"k")
    img = axes[1].imshow(np.ones((w,h-2),),vmin=-0.3,vmax=0.3)
    #img = axes[1].imshow(np.ones((w,h-2),),vmin=-0.001,vmax=0.001)
    fig.colorbar(img, orientation="horizontal", location="top")
    plt.tight_layout()

    for n in range(simsteps if visualisation else measure_step):
        equilibrated = sim.equilibrate()
        error = np.isnan(equilibrated).any()
        #edgy bounceback:
        sim.stream(equilibrated) #implicit relax=1

        u_eval = np.sum(sim.state[:,:,:,None]*e_i[None,None,:,:],2)
        uy_eval = u_eval[:,:,1]/sim.rho

        prim_pressure_drop = (sim.rho[:,:-1] - sim.rho[:,1:])/3

        #arithmatic avg
        #pressure_drop = (sim.rho[:,:-2]-sim.rho[:,2:])/2/3
        #harmonic avg
        #pressure_drop = 2*prim_pressure_drop[:,:-1]*prim_pressure_drop[:,1:]/(prim_pressure_drop[:,:-1]+prim_pressure_drop[:,1:])
        #geometric avg
        pressure_drop = -np.log(sim.rho[:,2:]/sim.rho[:,:-2])/2 *sim.rho[:,1:-1]/3
        density_corrected = expected_crosssection[:,None]/sim.rho[:,1:-1] #turn /nu into /mu
        exp_velocity = density_corrected*pressure_drop

        err.append(np.sum((exp_velocity[:,compare_window] - uy_eval[:,measure_window])**2)**0.5 or 1)

        if visualisation or error:
            if n==0:
                axes[2].set_ylim(0,err[0]*2)#np.sqrt((p1+p2)/(p2-p1))) #scaling total guesswork, feel free to customize
            if error:
                axes[2].set_ylim(0,max(err))
            exp_plot.set_xdata(np.average(exp_velocity[:,compare_window],1))
            sc_v.set_offsets(np.stack((np.average(uy_eval[:,measure_window],1), np.arange(w)),1))
            sc_err.set_xdata(np.arange(len(err)))
            sc_err.set_ydata(err)
            img.set_data(uy_eval[:,1:-1]-exp_velocity)
            #img.set_data(sim.rho)
            fig.canvas.flush_events()
            if error:
                plt.savefig("poseuille_error.png")
                assert False, "Ran into over/underflows. Last step: "+str(n)
        elif (n+1)==measure_step and err[-1] * 100 > err[0]:
            with open("test_pressure_poseuille_errors.pkl", "wb") as pkl:
                pickle.dump(err, pkl)
            assert err[-1] * 100 < err[0], f"Prediction was not met after {measure_step}. Errors stored in test_pressure_poseuille_errors.pkl"
