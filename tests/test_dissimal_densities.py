from ..src.utils import *
from ..src.LBM import *
import matplotlib.pyplot as plt

def test_laminar_flow():

    mu_1=1
    mu_ratio = 1 #mu2 / mu1
    dens_rat = 2
    mu_2 = mu_1*mu_ratio
    nu_ratio = mu_ratio

    w=128
    h=128

    slip_vel = 0.8
    split_height = 40
    #
    equil_vel = mu_1*slip_vel/(split_height) /  ( mu_2/(h-split_height) + mu_1/(split_height))

    vel_field = np.stack((np.arange(h,0,-1)*np.ones((w,h),),np.zeros((w,h),)),axis=2)

    #up to split_height, mu1 from V to equil_v
    vel_field[:,:split_height,0] = slip_vel - np.arange(split_height)[None,:] * (slip_vel-equil_vel)/(split_height)
    #from split_height onward, mu2 from equil_v to 0
    vel_field[:,split_height:,0] *= equil_vel/(h-split_height)
    compare = vel_field[:,:,0]

    u_distrib = vel_distrib(vel_field)
    fluid_1 = u_distrib + (weights)[None,None,:]
    fluid_2 = fluid_1.copy()#*dens_rat
    #remove where no fluid_1 should be
    fluid_1[:,split_height:,:]=0
    #remov where not fluid_2 should be
    fluid_2[:,:split_height,:]=0

    plt.ion()

    flow1=Compressible_Flow(1,nu=mu_1)
    flow1(fluid_1*dens_rat)
    flow2=Compressible_Flow(1,nu=mu_2)
    flow2(fluid_2)

    immiscible = DissimilarDensities_LBM(flow1, flow2, dens_rat)
    immiscible.add_transferes(gen_periodic_walls([[1,0],[-1,0]]))
    immiscible.add_transferes(gen_no_slip_walls([[0,1]]))
    immiscible.add_transferes(gen_velocity_boundary([0,-1], [slip_vel, 0]))

    assert immiscible._rt == mu_1*3+0.5
    assert immiscible._r2 == mu_2*3*dens_rat+0.5

    fig, axes = plt.subplots(nrows=2,ncols=1)#, figsize=(8,4))
    axes[1].set_ylabel('vel')
    plot, = axes[1].plot(np.arange(h), vel_field[0,:,0],'k', label='analytical')
    density, = axes[1].plot(np.arange(h), immiscible.rho[0,:],'k', label='analytical')
    phase, = axes[1].plot(np.arange(h), immiscible.VOF[0,:], 'k')
    axes[1].scatter(np.arange(h), vel_field[0,:,0], label='analytical')
    axes[1].set_ylim([-0.1,slip_vel*5.1])
    img = axes[0].imshow(np.ones((w,h),),vmin=0,vmax=slip_vel*2)#, cmap='jet')
    fig.colorbar(img)
    equilibrated = immiscible.base_equilibrate()
    relaxed = immiscible.state + (equilibrated-immiscible.state)/immiscible.relax[:,:,None]
    relaxed += np.sum((
                #surface_tension * immiscible.interface_curvature + \
                immiscible.full_pressure_jump()
            )[:,:,None,:] * e_i[None,None,:,:], -1)*weights[None,None,:]*3
    _ , diff =immiscible.diff_equilibrate(relaxed)
    immiscible.stream(relaxed, diff)
    while True:
        equilibrated = immiscible.base_equilibrate()
        relaxed = immiscible.state + (equilibrated-immiscible.state)/immiscible.relax[:,:,None]
        _ , diff =immiscible.diff_equilibrate(relaxed)
        immiscible.stream(relaxed, diff)

        #img.set_data(np.sum(diff,-1)+0.5)
        img.set_data(np.sum(equilibrated*e_i[None,None,:,0],2)/immiscible.rho)
        v = np.sum(np.sum(equilibrated*e_i[None,None,:,0],2)/immiscible.rho, 0)/w
        plot.set_ydata(v)
        phase.set_ydata(np.sum(equilibrated*e_i[None,None,:,0],(0,2))/w+2)
        density.set_ydata(np.sum(immiscible.phase_grad[:,:,1], 0)/w+1.5)
        print(np.sum((np.sum(equilibrated*e_i[None,None,:,0],2)/immiscible.rho-compare)**2)**0.5, np.sum(equilibrated))
        fig.canvas.flush_events()
