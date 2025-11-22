from ..src.utils import *
from ..src.LBM import *
import matplotlib.pyplot as plt

def test_laminar_flow():
    """
    Two fluids of different viscosity slide across eachother.
    Setup is infinite channel by two sides having periodic boundaries.
    One wall is set to constant velocity, the opposing wall uses
    no-slip-condition for simplicity. (aka zero-velocity)

    In this experiment, the velociy linearly decreases with dv/dy
    proportional to local mu. This allows calculating the velocity at
    the interface, which in turn allows calculating the velocity-profile.
    With the exact solution, the total error can be evaluated.

    Free chosen parameters are: nu_ratio, slip_vel, split_height (smaller than h)
    """

    #general parameters
    visualisation = True
    simsteps = 2000
    initial_offset_factor = .9 #introduce perturbation in velocit-field

    #general fluid setup
    relax_1=1
    nu_1=1
    nu_ratio = 5 #nu2 / nu1
    nu_2 = nu_1*nu_ratio
    nu_ratio = nu_ratio

    w=128
    h=128

    #specific experiment setup
    slip_vel = 0.8
    split_height = 40
    surface_tension = 0.1 #should have no effect

    # << setting up slip scenario >>
    #--------------------------------

    # calculate velocity at interface
    equil_vel = nu_1*slip_vel/(split_height) /  ( nu_2/(h-split_height) + nu_1/(split_height))

    # set up vel-vecs everywhere (v_y should be zero.)
    vel_field = np.zeros((w,h,2),)

    #up to split_height, linearly interpolate between slip_vel and equil_vel
    vel_field[:,:split_height,0] = slip_vel - (slip_vel-equil_vel)/(split_height) *np.arange(split_height)[None,:]
    #from split_height onward, interpolate between equil_vel and 0
    vel_field[:,split_height:,0] = equil_vel/(h-split_height) *np.arange(h-split_height,0,-1)[None, :]
    #extract expected profile instead of calculating again.
    compare = vel_field[:,:,0]

    #Turn velocity field into connecticals distributions
    u_distrib = vel_distrib(vel_field*initial_offset_factor)
    #build two fluid distribs, remove where not wanted later
    fluid_1 = u_distrib + weights[None,None,:]
    fluid_2 = fluid_1.copy()# for simplicity


    #remove where no fluid_1 should be
    fluid_1[:,split_height:,:]=0
    #remov where not fluid_2 should be
    fluid_2[:,:split_height,:]=0

    #turn distribs into simulators
    flow1=Compressible_Flow(1,nu=nu_1)
    flow1(fluid_1)
    flow2=Compressible_Flow(1,nu=nu_2)
    flow2(fluid_2)

    #shorthand to turn sims into immiscibility-sim
    immiscible = flow1 - flow2
    #add boundary-conditions
    immiscible.add_transferes(gen_periodic_walls([[1,0]], automirror=True))
    immiscible.add_transferes(gen_no_slip_walls([[0,1]]))
    immiscible.add_transferes(gen_velocity_boundary([0,-1], [slip_vel, 0]))
    immiscible.x_repeat=True

    #sanity checks
    assert immiscible._rt == nu_1*3+0.5
    assert immiscible._r2 == nu_2*3+0.5

    #set up visualisation
    if visualisation:
        plt.ion()

    fig, axes = plt.subplots(nrows=2,ncols=1)#, figsize=(8,4))
    axes[1].set_ylabel('vel')
    plot, = axes[1].plot(np.arange(h), vel_field[0,:,0],'k', label='analytical')
    density, = axes[1].plot(np.arange(h), immiscible.rho[0,:],'k', label='analytical')
    phase, = axes[1].plot(np.arange(h), immiscible.VOF[0,:], 'k')
    axes[1].scatter(np.arange(h), vel_field[0,:,0], label='analytical')
    axes[1].set_ylim([-0.1,slip_vel*5.1])
    img = axes[0].imshow(np.ones((w,h),),vmin=0,vmax=slip_vel*2)#, cmap='jet')
    fig.colorbar(img)

    #for later comparison and error-detection
    total_diff = np.sum(immiscible.diff)
    total_fluid = np.sum(immiscible.state)

    #simulate one step to get first error
    equilibrated = immiscible.base_equilibrate()
    relaxed = immiscible.state + (equilibrated-immiscible.state)/immiscible.relax[:,:,None]
    _ , diff =immiscible.diff_equilibrate(relaxed)
    immiscible.stream(relaxed, diff)

    vel_x = np.sum(relaxed*e_i[None,None,:,0],2)/immiscible.rho
    error = np.sum((vel_x-compare)**2)**0.5
    err = [error]
    error_exit = False
    for n in range(simsteps):
        equilibrated = immiscible.base_equilibrate()

        body_force = np.sum((
                surface_tension * immiscible.interface_curvature + \
                immiscible.normal_pressure_jump
            )[:,:,None,:] * e_i[None,None,:,:], -1)*weights[None,None,:]*3
        relaxed = immiscible.state + (equilibrated-immiscible.state)/immiscible.relax[:,:,None] + body_force
        _ , diff =immiscible.diff_equilibrate(relaxed)
        immiscible.stream(relaxed, diff)

        assert np.isclose(np.sum(diff), total_diff)
        assert np.isclose(np.sum(relaxed), total_fluid)

        vel_x = np.sum(relaxed*e_i[None,None,:,0],2)/immiscible.rho
        error = np.sum((vel_x-compare)**2)**0.5
        err.append(error)

        if ( n!=0 and err[-1] > err[-2]):
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
            #img.set_data(np.sum(diff,-1)+0.5)
            img.set_data(vel_x)
            v_avg = np.average(vel_x, 0)
            plot.set_ydata(v_avg)
            phase.set_ydata(np.sum(relaxed*e_i[None,None,:,0],(0,2))/w+2)
            density.set_ydata(np.sum(immiscible.phase_grad[:,:,1], 0)/w+1.5)
            fig.canvas.flush_events()
            if True and error_exit:
                plt.savefig("immiscible_laminar_error.png")
                assert False, f"Error behaving unexpectedly in last 10 steps. Last step: {str(n)}"
    if err[-1]*10 > err[0]:
        assert err[-1] * 10 < err[0], f"Prediction was not met after {simsteps}."


def test_surface_tension():

    #basic setup
    visualisation = True
    #thorough: simsteps = 20k ... roughly 30 min run
    simsteps = 1000

    w = 128
    h = 128
    surface_tension=0.02
    set_radius = 40

    nu_1=0.8
    nu_2=0.8

    #set up visualisation
    if visualisation:
        plt.ion()

    fig, axes = plt.subplots(nrows=1,ncols=1)#, figsize=(8,4))
    img = axes.imshow(np.ones((w,h),),vmin=0,vmax=0.01)#, cmap='jet')
    fig.colorbar(img)

    fluid_1 = np.ones((w,h,9),)*weights[None,None,:]
    fluid_2 = fluid_1.copy()
    radius_match = np.zeros((w,h,),)
    for i in range(w):
        for j in range(h):
            radius_match[i,j] = ((i-w//2)**2+(j-h//2)**2 or 0.000000000001)**0.5
            if (i-w//2)**2+(j-h//2)**2 > set_radius**2:
                fluid_1[i,j,:]=0
            else:
                fluid_2[i,j,:]=0

    #Viscosity changes relax-time! non-unit relax is important test-pertubation.
    #Choose viscosity to quickly reach equilibrium
    fluid1 = Compressible_Flow(1, nu=nu_1)
    fluid1(fluid_1*2) #change density for test-perturbation
    fluid2 = Compressible_Flow(1, nu=nu_2)
    fluid2(fluid_2*2) #keep same density! Dissimilar densities are in another class.

    bubble_sim = fluid1 - fluid2
    bubble_sim.add_transferes(gen_periodic_walls([[1,0],[0,1],[1,1],[-1,1]], automirror=True))
    #bubble_sim.add_transferes(gen_no_slip_walls([[1,0],[-1,0],[0,1],[0,-1]]))
    total_fluid = np.sum(bubble_sim.state)
    total_diff = np.sum(bubble_sim.diff)
    bubble_sim.x_repeat=True
    bubble_sim.y_repeat=True

    #sanity checks
    assert bubble_sim.test_conditions()
    assert bubble_sim._rt == nu_1*3+0.5
    assert bubble_sim._r2 == nu_2*3+0.5

    measured_surface_tension = 0
    measured_curve = 0

    for n in range(simsteps):
        equilibrated = bubble_sim.base_equilibrate()

        body_force = np.sum((
                surface_tension * bubble_sim.interface_curvature + \
                bubble_sim.normal_pressure_jump()
            )[:,:,None,:] * e_i[None,None,:,:], -1)*weights[None,None,:]*3
        relaxed = bubble_sim.state + (equilibrated-bubble_sim.state)/bubble_sim.relax[:,:,None] + body_force
        #relaxed_diff = bubble_sim.diff + (diff - bubble_sim.diff)/bubble_sim.relax[:,:,None]
        _, diff = bubble_sim.diff_equilibrate(relaxed)

        #assert np.isclose(total_fluid, np.sum(bubble_sim.state))
        #assert np.isclose(total_diff, np.sum(bubble_sim.diff))
        img.set_data(np.sum(bubble_sim.interface_curvature*bubble_sim.phase_grad,-1)/bubble_sim.phase_norm)
        fig.canvas.flush_events()

        outside_density = (bubble_sim.rho[1:5,1:5].sum()+bubble_sim.rho[1:5,-5:-1].sum()+bubble_sim.rho[-5:-1,1:5].sum()+bubble_sim.rho[-5:-1,-5:-1].sum())/64
        inside_density = bubble_sim.rho[w//2-3 : w//2+3,h//2-3:h//2+3].sum()/36
        deltaP = (inside_density-outside_density)/3 # /3 == *c_s**2

        #measure radius by matching with 0-centered circle
        mixture = (0.25-bubble_sim.VOF**2)
        mixture[mixture<0.1] = 0
        radius= (np.sum(mixture*radius_match)/np.sum(mixture) if np.any(mixture!=0) else 0)
        #measure radius by calculating surface area and taking squarroot
        #radius = np.sum(0.5+bubble_sim.VOF)**0.5/np.pi**0.5
        measured_surface_tension = radius*deltaP

        #calculate total curvature, should sum to 2*pi. Larger by about 1%? Maybe long-tail-swings?
        sel = bubble_sim.phase_norm < 1
        total_curv = np.sum(np.sum(bubble_sim.interface_curvature[sel,:]*bubble_sim.phase_grad[sel,:],-1)/bubble_sim.phase_norm[sel])


        #make sure to stream after measurements to avoid confusions
        bubble_sim.stream(relaxed, diff)
        cross_curve = 0
        for i in range(w//2,w-1):
            if sel[i, i]:
                cross_curve += np.sum(bubble_sim.interface_curvature[i,i,:]*bubble_sim.phase_grad[i,i,:],-1)/bubble_sim.phase_norm[i,i]
            if sel[i+1,i]:
                cross_curve += np.sum(bubble_sim.interface_curvature[i+1,i,:]*bubble_sim.phase_grad[i+1,i,:],-1)/bubble_sim.phase_norm[i+1,i]
        cross_curve /= 2
        straight_curve = np.sum(np.sum(bubble_sim.interface_curvature[w//2,h//2:,:][sel[w//2,h//2:],:]*bubble_sim.phase_grad[w//2,h//2:,:][sel[w//2,h//2:],:],-1)/bubble_sim.phase_norm[w//2,h//2:][sel[w//2,h//2:]])
        print(f"{deltaP:<.10f} {radius:<.10f} {deltaP*radius:<.10f} {total_curv:<.5f} {2*np.pi:<.5f} {cross_curve:<.10f} {n}")

    #more sanity-checks than rigorous physics test.
    assert np.isclose(measured_surface_tension, surface_tension, rtol=0.2)
    assert np.isclose(total_curv, np.pi*2, atol=0.002)
    #thorogh checks (run ca. 20k steps)
    #assert np.isclose(measured_surface_tension, surface_tension, rtol=0.005)
    print(total_curv, measured_surface_tension)