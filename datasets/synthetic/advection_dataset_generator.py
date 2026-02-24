import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

np.random.seed(17)
def lcm_floats(a, b):
    """Compute the least common multiple of two floats."""
    fa = Fraction(a).limit_denominator(1000000)
    fb = Fraction(b).limit_denominator(1000000)
    lcm_denom = np.lcm(fa.denominator, fb.denominator)
    # Scale numerators to common denominator
    num_a = fa.numerator * (lcm_denom // fa.denominator)
    num_b = fb.numerator * (lcm_denom // fb.denominator)
    lcm_num = np.lcm(num_a, num_b)
    return lcm_num / lcm_denom

class FVGrid:

    def __init__(self, nx, ny, ng=2, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.ng = ng
        self.nx = nx
        self.ny = ny

        # python is zero-based.  Make easy integers to know where the
        # real data lives
        self.ilo = ng
        self.ihi = ng+nx-1
        self.jlo = ng
        self.jhi = ng+ny-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.dy = (ymax - ymin)/(ny)
        self.x = xmin + (np.arange(nx+2*ng)-ng+0.5)*self.dx
        self.y = ymin + (np.arange(ny+2*ng)-ng+0.5)*self.dy
        self.xl = xmin + (np.arange(nx+2*ng)-ng)*self.dx
        self.xr = xmin + (np.arange(nx+2*ng)-ng+1.0)*self.dx
        self.yl = ymin + (np.arange(ny+2*ng)-ng)*self.dy
        self.yr = ymin + (np.arange(ny+2*ng)-ng+1.0)*self.dy

        # meshgrid for plotting
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # storage for the solution
        self.a = self.scratch_array()
        self.ainit = self.scratch_array()

    def period(self, u, v):
        """ return the period for advection with velocities u, v """
        period_x = (self.xmax - self.xmin) / np.abs(u)
        period_y = (self.ymax - self.ymin) / np.abs(v)
        return lcm_floats(period_x, period_y)

    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return np.zeros((self.nx+2*self.ng, self.ny+2*self.ng), dtype=np.float64)

    def fill_BCs(self, atmp):
        """ fill all ghostcells with periodic boundary conditions """

        # x direction
        for n in range(self.ng):
            atmp[self.ilo-1-n, :] = atmp[self.ihi-n, :]
            atmp[self.ihi+1+n, :] = atmp[self.ilo+n, :]

        # y direction
        for n in range(self.ng):
            atmp[:, self.jlo-1-n] = atmp[:, self.jhi-n]
            atmp[:, self.jhi+1+n] = atmp[:, self.jlo+n]

    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if e.shape != (self.nx+2*self.ng, self.ny+2*self.ng):
            return None

        return np.sqrt(self.dx*self.dy*np.sum(e[self.ilo:self.ihi+1, self.jlo:self.jhi+1]**2))
    
    def plot(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        axs[0].contourf(self.X, self.Y, self.ainit.T, levels=20)
        axs[0].set_title("Initial conditions")
        
        axs[1].contourf(self.X, self.Y, self.a.T, levels=20)
        axs[1].set_title("Final solution")
        
        return fig
    
def flux_update_simple_x(gr, u, a):
    """compute x-flux difference for linear advection"""

    # slope in x
    da = gr.scratch_array()
    da[gr.ilo-1:gr.ihi+2, :] = 0.5*(a[gr.ilo:gr.ihi+3, :] - a[gr.ilo-2:gr.ihi+1, :])

    # upwinding means that we take the left state always
    # convection: aint[i,j] = a_{i-1/2,j}
    aint = gr.scratch_array()
    aint[gr.ilo:gr.ihi+2, :] = a[gr.ilo-1:gr.ihi+1, :] + 0.5*da[gr.ilo-1:gr.ihi+1, :]

    flux_diff = gr.scratch_array()
    flux_diff[gr.ilo:gr.ihi+1, :] = u * (aint[gr.ilo:gr.ihi+1, :] - aint[gr.ilo+1:gr.ihi+2, :]) / gr.dx

    return flux_diff

def flux_update_simple_y(gr, v, a):
    """compute y-flux difference for linear advection"""

    # slope in y
    da = gr.scratch_array()
    da[:, gr.jlo-1:gr.jhi+2] = 0.5*(a[:, gr.jlo:gr.jhi+3] - a[:, gr.jlo-2:gr.jhi+1])

    # upwinding means that we take the bottom state always
    # convection: aint[i,j] = a_{i,j-1/2}
    aint = gr.scratch_array()
    aint[:, gr.jlo:gr.jhi+2] = a[:, gr.jlo-1:gr.jhi+1] + 0.5*da[:, gr.jlo-1:gr.jhi+1]

    flux_diff = gr.scratch_array()
    flux_diff[:, gr.jlo:gr.jhi+1] = v * (aint[:, gr.jlo:gr.jhi+1] - aint[:, gr.jlo+1:gr.jhi+2]) / gr.dy

    return flux_diff

def advection_mol(nx, ny, u, v, C, num_periods=1, init_cond=None, flux_update_x=flux_update_simple_x, flux_update_y=flux_update_simple_y):

    # create a grid
    g = FVGrid(nx, ny, ng=2)

    tmax = num_periods * g.period(u, v)

    # setup initial conditions
    init_cond(g)
    g.ainit[:] = g.a[:]

    # compute the timestep
    dt = C * min(g.dx, g.dy) / max(np.abs(u), np.abs(v))

    data = np.zeros((int(tmax//dt)+1, nx, ny))
    data[0,:,:] = g.a[g.ilo:g.ihi+1, g.jlo:g.jhi+1]

    t = 0.0
    n_step = 0
    while t < tmax:
        if t + dt > tmax:
            dt = tmax - t

        # second-order RK integration
        g.fill_BCs(g.a)
        k1x = flux_update_x(g, u, g.a)
        k1y = flux_update_y(g, v, g.a)

        atmp = g.scratch_array()
        atmp[:] = g.a[:] + 0.5 * dt * (k1x[:] + k1y[:])

        g.fill_BCs(atmp)
        k2x = flux_update_x(g, u, atmp)
        k2y = flux_update_y(g, v, atmp)

        g.a[:] += dt * (k2x[:] + k2y[:])
        n_step += 1
        if n_step < data.shape[0]:
            data[n_step,:,:] = g.a[g.ilo:g.ihi+1, g.jlo:g.jhi+1]
        t += dt

    return g,data

def sines(g):
    """Set initial conditions to a sum of sines"""

    X, Y = np.meshgrid(g.x, g.y)
    g.a = (np.sin(2.0*np.pi*X) + np.sin(4.0*np.pi*Y))

def random_poly(g):
    """Set initial conditions to a random polynomial"""

    a_coeffs = np.random.rand(6)
    X, Y = np.meshgrid(g.x, g.y)
    g.a = (a_coeffs[0] + a_coeffs[1]*X + a_coeffs[2]*Y + 
           a_coeffs[3]*X**2 + a_coeffs[4]*X*Y + a_coeffs[5]*Y**2)

"""Parameters for advection
C = CFL Constant = max(|u|,|v|) dt / min(dx,dy)
u = velocity in x
v = velocity in y
nx = number of grid points in x
ny = number of grid points in y"""

C = 0.35 # keep less than 0.5
u = 1.9
v = 0.3
nx = 64
ny = 64
init = random_poly
g,data = advection_mol(nx, ny, u, v, C, init_cond=init)
fig = g.plot()
# plt.show()

# check error, should go down as O(n^2)
# y=[]
# for nx in [32, 64, 128, 256]:
#     g,_ = advection_mol(nx, nx, u, v, C, init_cond=init)
#     temp = g.norm(g.a - g.ainit)
#     y.append(temp)
#     print(f"{nx:3d}: {temp:10.8f}")
# plt.plot(np.log([32, 64, 128, 256]), np.log(y), marker='o')
# plt.show()

np.save("./datasets/synthetic/advection_data_2d.npy", data)