from src.utils import *
from src.LBM import *
import matplotlib.pyplot as plt
from textwrap import dedent

#basic setup
just_run=False #set to true if you don't want explainatory interuptions (e.g. if you change some values)
#thorough: simsteps = 20k ... roughly 30 min run
simsteps = 20000
#channel width
w = 128
h = 512
#initial
bubble_radius=30
#interface surface tension
surface_tension=0.008

#gravity acceleration, arbitrary units
g = -0.0008

#density ratio
dens_rat = .15 #rho1/rho2
#kinematic viscosity
nu_1=1
nu_ratio = 1 #mu2 / mu1
nu_2 = nu_1*nu_ratio

#set up visualisation
plt.ion()

fig, axes = plt.subplots(nrows=1,ncols=2)#, figsize=(8,4))
img = axes[1].imshow(np.ones((h,w),),vmin=-0.05,vmax=0.05, origin="lower")#, cmap='jet')
fig.colorbar(img)
rhomap = axes[0].imshow(np.ones((h,w),),vmin=0,vmax=2, origin="lower")#, cmap='jet')
fig.colorbar(rhomap)

fluid_1 = np.ones((w,h,9),)*weights[None,None,:] * np.linspace((-(h-.5)*g*3/2+1)/1,((h-.5)*g*3/2+1)/1, h)[None,:,None]
fluid_2 = fluid_1.copy()
radius_match = np.zeros((w,h,),)
for i in range(w):
    for j in range(h):
        if (i-w//2)**2+(j-w//2)**2 > bubble_radius**2:
            fluid_1[i,j,:]=0
        else:
            fluid_2[i,j,:]=0

#Viscosity changes relax-time! non-unit relax is important test-pertubation.
#Choose viscosity to quickly reach equilibrium
fluid1 = Compressible_Flow(1, nu=nu_1)
fluid1(fluid_1*1*(dens_rat+surface_tension/bubble_radius)) #change density for test-perturbation
fluid2 = Compressible_Flow(1, nu=nu_2)
fluid2(fluid_2*1) #keep same density! Dissimilar densities are in another class.

riser = DissimilarDensities_LBM(fluid1, fluid2, dens_rat)
riser.add_transferes(gen_periodic_walls([[1,0],[1,1],[-1,1]], automirror=True))
riser.add_transferes(gen_no_slip_walls([[0,1],[0,-1]]))
total_fluid = np.sum(riser.state)
total_diff = np.sum(riser.diff)
riser.x_repeat=True
riser.y_repeat=True

#sanity checks
assert riser.test_conditions()
#assert riser._rt == nu_1*3+0.5
#assert riser._r2 == nu_2*3+0.5

measured_surface_tension = 0
measured_curve = 0

def prompt_continue():
    input('>>>> PRESS ENTER TO CONTINUE SIMULATION <<<<')
    print('\033[A                                            ')

for n in range(simsteps):
    equilibrated = riser.base_equilibrate()

            #surface_tension * riser.interface_curvature + 
            #np.stack([np.zeros_like(riser.rho),g*riser.rho],2) 
            #riser.full_pressure_jump()
    body_force = riser.get_body_force_connecticals(
        surface_tension * riser.interface_curvature +
        np.stack([np.zeros_like(riser.rho),g*riser.rho],2)
    )
    relaxed = riser.state + (equilibrated-riser.state)/riser.relax[:,:,None] + body_force/riser.relax[:,:,None]
    #relaxed_diff = riser.diff + (diff - riser.diff)/riser.relax[:,:,None]
    _, diff = riser.diff_equilibrate(relaxed)

    assert np.isclose(total_fluid, np.sum(riser.state)), "The amount of fluid changed!"
    assert np.isclose(total_diff, np.sum(riser.diff))
    img.set_data(np.sum(riser.state[:,:,:] * e_i[None, None, :,1], 2).T/riser.rho.T)
    rhomap.set_data(riser.rho.T)
    fig.canvas.flush_events()

    #calculate total curvature, should sum to 2*pi. Larger by about 1%? Maybe long-tail-swings?
    sel = riser.phase_norm < 1
    total_curv = np.sum(np.sum(riser.interface_curvature[sel,:]*riser.phase_grad[sel,:],-1)/riser.phase_norm[sel])


    #make sure to stream after measurements to avoid confusions
    riser.stream(relaxed, diff)

    if just_run:
        continue
    if n==0:
        print(dedent('''\
            Welcome to the simulation showcase!\n
            What you should see right now is two columns of (compressible) fluid.
            The left shows a density distribution, the right distribution of upwards-velocity.
            The column is set up with heavy fluid, presettled (meaning denser fluid at the bottom) and
            with a bubble of a different, lighter fluid at the bottom. So the situation is basically
            an air-bubble under water, just compressible water.

            Hit [ Ctrl + c ] to stop the simulation at any time.
        '''))
        prompt_continue()
    if n==260:
        print('\033[A'+dedent('''\
            At this point, you should see a wave of fluid moving upwards
            (a yellow area at the top of the right velocity-column), as the presettled fluid still needs to fully settle.
            At the same time, left and right of the bubble fluid is streaming downwards (blue areas next to the bubble).
            This fluid will rush below the bubble and push the bubble upwards soon, but it's still too slow to notice.
            However, hovering the mouse-cursor over the bubble in the velocity-column, you can see positive
            (i.e. upwards) values in the top left corner!
        '''))
        prompt_continue()
    if n==1000:
        print('\033[A'+dedent('''\
            Now the bubble starts to take shape.
            In the left column, the bottom of the bubble has flattened. The right shows clearly an upwards movement
            of the whole bubble. Downward movement at the sides is more pronounced, but the bubble is mostly driven
            upwards from its bottom center. This will drive the shape of the bubble.
        '''))
        prompt_continue()
    if n==2000:
        print('\033[A'+dedent('''\
            A peculiarity of this simulation becomes visible now:
            As the bubble becomes wider, the downward flow to the sides has to speed up. Side walls behave
            periodically (meaning whatever flows out to the left returns from the right), so this downward flow
            is basically being squeezed between two bubbles. Smaller bubbles results in a more dispersed (and 
            thereby slower) downward flow. A solution would be to widen the column, but this
            dramatically increases computation requirements.

            If you look closely, you can see the momentum-exchange at the bottom of the bubble:
            The yellow line just above the interface is bubble-content that just got kicked up by embedding fluid.
            Just below the interface, the embedding fluid crashed into the bubble-wall and got slowed down.
        '''))
        prompt_continue()
    if n==4500:
        print('\033[A'+dedent('''\
            Two trailing streaks at the side of the bubble have formed.
            They are unstable. Surface tension will pull them apart soon. To better observe this, I will print
            the total curvature of the interface. The curvature of any closed path should sum up to 2*pi.
            Due to numeric aberations, total curvature always deviates from this value. But it's a good way to
            measure break up in smaller bubbles.

            You will see raw total curvature, as well as total curvature / 2 pi as an estimate of bubble-number.
        '''))
        prompt_continue()
    if n==6000:
        print('\033[A'+dedent('''                                                                          
            That's all I had prepared for you. You can leave this simulation running and observe the rising
            bubble. It will grow as it reaches lower pressure areas and loose more streaks. The simulation stops
            automatically after 20 000 steps. If you want, there are some obvious changes you could make in the
            simshowcase.py file to observe other behaviors:
             - change the initial bubble size, thereby changing the amount of "air"
             - change the surface tension. (be careful!)
             - change the viscosity of embedding and bubble gas
             - change the column geometry, set the bubble off-center

            be advised: this simulator has no numerical stabilisation and *will* break in edge-cases.
            Tread lightly! ;)
        '''))
        prompt_continue()
    if n < 4500:
        print('\033[A','|/-\\'[n%4])
    if n==4500:
        print('\033[A         ')
    if 4500 <= n:
        print(f"Total Curvature: {total_curv:<.5f} , Bubble Number: {total_curv/2/np.pi:<.5f} , Current Step: {n}")

input("That's it. Hit ENTER to close the simulation-window!")