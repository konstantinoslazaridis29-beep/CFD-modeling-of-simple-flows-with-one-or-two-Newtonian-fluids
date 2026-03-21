import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def upwind2_x(f, u, dx):
    diff = np.zeros_like(f)
    mask_pos = u[:, 2:-2] > 0
    mask_neg = ~mask_pos
    # positive velocity (backward biased)
    diff[:, 2:-2][mask_pos] = (3 * f[:, 2:-2][mask_pos] - 4 * f[:, 1:-3][mask_pos] + f[:, 0:-4][mask_pos]) / (2 * dx)
    # negative velocity (forward biased)
    diff[:, 2:-2][mask_neg] = (-3 * f[:, 2:-2][mask_neg] + 4 * f[:, 3:-1][mask_neg] - f[:, 4:][mask_neg]) / (2 * dx)
    # Use central difference for boundary regions
    diff[:, 1] = (f[:, 2] - f[:, 0]) / (2 * dx)
    diff[:, -2] = (f[:, -1] - f[:, -3]) / (2 * dx)
    return diff

def upwind2_y(f, v, dy):
    diff = np.zeros_like(f)
    mask_pos = v[2:-2, :] > 0
    mask_neg = ~mask_pos
    # positive velocity (backward biased)
    diff[2:-2, :][mask_pos] = (3 * f[2:-2, :][mask_pos] - 4 * f[1:-3, :][mask_pos] + f[0:-4, :][mask_pos]) / (2 * dy)
    diff[2:-2, :][mask_neg] = (-3 * f[2:-2, :][mask_neg] + 4 * f[3:-1, :][mask_neg] - f[4:, :][mask_neg]) / (2 * dy)
    # Use central difference for boundary regions
    diff[1, :] = (f[2, :] - f[0, :]) / (2 * dy)
    diff[-2, :] = (f[-1, :] - f[-3, :]) / (2 * dy)
    return diff

'''2nd order central difference'''
def central_difference_x(f, dx):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * dx)
    return diff

def central_difference_y(f, dy):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * dy)
    return diff

def laplace(f, dx, dy):
    diff = np.zeros_like(f)
    diff[1:-1, 1:-1] = (
            (f[1:-1, 0:-2] - 2 * f[1:-1, 1:-1] + f[1:-1, 2:]) / (dx ** 2) +  # d²f/dx²
            (f[0:-2, 1:-1] - 2 * f[1:-1, 1:-1] + f[2:, 1:-1]) / (dy ** 2)  # d²f/dy²
    )
    return diff


def divergence_mac(u, v, dx, dy):
    """
    Correct MAC divergence at cell centers.
    u: (ny, nx+1) - indices [row, col]
    v: (ny+1, nx)
    returns div: (ny, nx)
    """
    # u-term: Difference between right face and left face of EVERY cell
    # u[:, 1:] is the right face, u[:, :-1] is the left face
    div_u = (u[:, 1:] - u[:, :-1]) / dx

    # v-term: Difference between top face and bottom face of EVERY cell
    # v[1:, :] is the top face, v[:-1, :] is the bottom face
    div_v = (v[1:, :] - v[:-1, :]) / dy

    div = div_u + div_v
    return div

def grad_p_x_mac(p, dx):
    """    ∂p/∂x at u-faces p: (ny, nx)  returns gx: (ny, nx+1)    """
    gx = np.zeros((p.shape[0], p.shape[1] + 1))
    gx[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dx
    return gx

def grad_p_y_mac(p, dy):
    """    ∂p/∂y at v-faces  p: (ny, nx)  returns gy: (ny+1, nx)    """
    gy = np.zeros((p.shape[0] + 1, p.shape[1]))
    gy[1:-1, :] = (p[1:, :] - p[:-1, :]) / dy
    return gy


def interp_v_to_u_face(v):
    """Interpolate v from v-faces to u-faces ---- v: (ny+1, nx) -> returns (ny, nx+1)"""
    v_at_u = np.zeros((v.shape[0] - 1, v.shape[1] + 1))

    # Average 4 surrounding v-values to get v at u-face
    v_at_u[:, 1:-1] = 0.25 * (
            v[:-1, :-1] + v[:-1, 1:] +  # Bottom two v-faces
            v[1:, :-1] + v[1:, 1:]  # Top two v-faces
    )

    # Boundaries
    v_at_u[:, 0] = 0.5 * (v[:-1, 0] + v[1:, 0])
    v_at_u[:, -1] = 0.5 * (v[:-1, -1] + v[1:, -1])

    return v_at_u

def interp_u_to_v_face(u):
    """Interpolate u from u-faces to v-faces ---- u: (ny, nx+1) -> returns (ny+1, nx)"""
    u_at_v = np.zeros((u.shape[0] + 1, u.shape[1] - 1))

    # Average 4 surrounding u-values to get u at v-face
    u_at_v[1:-1, :] = 0.25 * (
            u[:-1, :-1] + u[:-1, 1:] +  # Left two u-faces
            u[1:, :-1] + u[1:, 1:]  # Right two u-faces
    )

    # Boundaries
    u_at_v[0, :] = 0.5 * (u[0, :-1] + u[0, 1:])
    u_at_v[-1, :] = 0.5 * (u[-1, :-1] + u[-1, 1:])

    return u_at_v

def interp_cell_to_u_face(f):
    """Interpolate cell-centered values to u-faces ---- f: (ny, nx) -> returns (ny, nx+1)"""
    f_u = np.zeros((f.shape[0], f.shape[1] + 1))
    f_u[:, 1:-1] = 0.5 * (f[:, :-1] + f[:, 1:])
    f_u[:, 0] = f[:, 0]
    f_u[:, -1] = f[:, -1]
    return f_u

def interp_cell_to_v_face(f):
    """Interpolate cell-centered values to v-faces ---- f: (ny, nx) -> returns (ny+1, nx)"""
    f_v = np.zeros((f.shape[0] + 1, f.shape[1]))
    f_v[1:-1, :] = 0.5 * (f[:-1, :] + f[1:, :])
    f_v[0, :] = f[0, :]
    f_v[-1, :] = f[-1, :]
    return f_v

def u_to_cell(u):
    """Interpolate u-velocity from vertical faces to cell centers."""
    # u shape: (ny, nx+1) -> returns (ny, nx)
    return 0.5 * (u[:, :-1] + u[:, 1:])

def v_to_cell(v):
    """Interpolate v-velocity from horizontal faces to cell centers."""
    # v shape: (ny+1, nx) -> returns (ny, nx)
    return 0.5 * (v[:-1, :] + v[1:, :])

def laplace_mac_u(u, dx, dy):
    """Laplacian of u-velocity at u-faces ---- u: (ny, nx+1) -> lap: (ny, nx+1)"""
    lap = np.zeros_like(u)

    # Need interior points in BOTH x and y
    lap[1:-1, 1:-1] = (
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx ** 2 +
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy ** 2)
    return lap

def laplace_mac_v(v, dx, dy):
    """Laplacian of v-velocity at v-faces ---- v: (ny+1, nx) -> lap: (ny+1, nx)"""
    lap = np.zeros_like(v)

    # Need interior points in BOTH x and y
    lap[1:-1, 1:-1] = (
            (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dx ** 2 +
            (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy ** 2
    )
    return lap

def H_sm(phi, epsilon):

    H = np.where(phi < -epsilon, 0.0,
        np.where(phi > epsilon, 1.0,
            0.5 * (1.0 + phi / epsilon + np.sin(np.pi * phi / epsilon) / np.pi)))

    return H

def initialize_phi(Xc, Yc,  Lx, y0, a, k):
    """    Initialize level set function phi for 2D Kelvin–Helmholtz instability.
    Parameters
    Lx : float,     Domain size in x direction
    y0 : float,     Mean interface height
    a : float,      Perturbation amplitude
    k : int,        Number of wavelengths in x-direction
    Returns
    phi : 2D numpy array, shape (nx, ny),   Level set function at cell centers
    x, y : 1D arrays,    Cell center coordinates    """
    # Interface shape
    #y_interface = y0 + a * np.cos(2.0 * np.pi * k * Xc / Lx)
    y_interface = y0 - a/2 + a * Xc / Lx
    #y_interface = y0 - a * (Xc - Lx/2)**2 / Lx

    # Level set: signed distance (approximate)
    phi = Yc - y_interface

    return phi

def interface_correction(phi_advected, dx, dy, nx_int, ny_int,  eps_c, dtau):
    # gradients at cell centers
    phi_x = np.zeros_like(phi_advected)
    phi_y = np.zeros_like(phi_advected)

    phi_x[:, 1:-1] = (phi_advected[:, 2:] - phi_advected[:, :-2]) / (2*dx)
    phi_y[1:-1, :] = (phi_advected[2:, :] - phi_advected[:-2, :]) / (2*dy)

    # f(phi) = phi(1-phi) * n
    q = phi_advected * (1.0 - phi_advected)
    fx = q * nx_int
    fy = q * ny_int

    # numerical fluxes
    F = 0.5*(fx[:, 1:] + fx[:, :-1]) - eps_c*(phi_advected[:, 1:] - phi_advected[:, :-1]) / dx
    G = 0.5*(fy[1:, :] + fy[:-1, :]) - eps_c*(phi_advected[1:, :] - phi_advected[:-1, :]) / dy

    # divergence of fluxes
    div = np.zeros_like(phi_advected)
    div[1:-1, 1:-1] = ((F[1:-1, 1:] - F[1:-1, :-1]) / dx + (G[1:, 1:-1] - G[:-1, 1:-1]) / dy)

    # pseudo-time update
    phi_new = phi_advected.copy()
    phi_new[1:-1, 1:-1] -= dtau * div[1:-1, 1:-1]

    # Neumann BCs (adjust if periodic)
    phi_new[:, 0]  = phi_new[:, -2]     #periodic
    phi_new[:, -1] = phi_new[:, 1]      #periodic
    phi_new[0, :]  = phi_new[1, :]
    phi_new[-1, :] = phi_new[-2, :]

    return phi_new


def compute_slopes_minmod(phi, dx, dy):
    """Calculates Minmod limited slopes for a MAC grid"""
    # Periodic padding for x-gradients (Periodic BCs)
    phi_pad_x = np.pad(phi, ((0, 0), (1, 1)), mode='wrap')
    # Edge padding for y-gradients (Wall BCs)
    phi_pad_y = np.pad(phi, ((1, 1), (0, 0)), mode='edge')

    # Calculate Gradients (Right and Left)
    gr_x = (phi_pad_x[:, 2:] - phi_pad_x[:, 1:-1]) / dx
    gl_x = (phi_pad_x[:, 1:-1] - phi_pad_x[:, :-2]) / dx

    gr_y = (phi_pad_y[2:, :] - phi_pad_y[1:-1, :]) / dy
    gl_y = (phi_pad_y[1:-1, :] - phi_pad_y[:-2, :]) / dy

    # Minmod Limiter Implementation
    def minmod(a, b):
        # Result is 0 if signs are different, else sign(a)*min(|a|, |b|)
        res = np.zeros_like(a)
        mask = a * b > 0
        res[mask] = np.sign(a[mask]) * np.minimum(np.abs(a[mask]), np.abs(b[mask]))
        return res

    sx = minmod(gr_x, gl_x)
    sy = minmod(gr_y, gl_y)

    return sx, sy



def compute_tvd_fluxes(phi, u_vel, v_vel, dx, dy):
    """Computes F and G fluxes using 2nd order TVD reconstruction"""
    sx, sy = compute_slopes_minmod(phi, dx, dy)

    # F flux at (i + 1/2, j) - size (ny, nx-1)
    # U_minus is reconstructed from cell i, U_plus from cell i+1
    phi_minus_x = phi[:, :-1] + (dx / 2.0) * sx[:, :-1]
    phi_plus_x = phi[:, 1:] - (dx / 2.0) * sx[:, 1:]

    # Upwinding based on interface velocity u_{i+1/2, j}
    u_face = u_vel[:, 1:-1]  # Extract interior u-faces
    F = np.maximum(u_face, 0) * phi_minus_x + np.minimum(u_face, 0) * phi_plus_x

    # G flux at (i, j + 1/2) - size (ny-1, nx)
    phi_minus_y = phi[:-1, :] + (dy / 2.0) * sy[:-1, :]
    phi_plus_y = phi[1:, :] - (dy / 2.0) * sy[1:, :]

    v_face = v_vel[1:-1, :]  # Extract interior v-faces
    G = np.maximum(v_face, 0) * phi_minus_y + np.minimum(v_face, 0) * phi_plus_y

    return F, G

def get_F_operator(phi, u_vel, v_vel, dx, dy):
    """The dU/dt = F(U) operator from Eq (A.3)"""
    F, G = compute_tvd_fluxes(phi, u_vel, v_vel, dx, dy)

    dUdt = np.zeros_like(phi)
    # Flux balance (divergence)
    dUdt[:, 1:-1] -= (F[:, 1:] - F[:, :-1]) / dx
    dUdt[1:-1, :] -= (G[1:, :] - G[:-1, :]) / dy

    return dUdt

def advect_phi_tvd(phi_n, u, v, dt, dx, dy):
    """One full time step using TVD-RK2"""
    # Step 1: U* = U^n + dt * F(U^n)
    L_n = get_F_operator(phi_n, u, v, dx, dy)
    phi_star = phi_n + dt * L_n

    # Apply BCs to phi_star here (e.g., periodic)
    phi_star[:, 0] = phi_star[:, -2];    phi_star[:, -1] = phi_star[:, 1]

    # Step 2: U** = U* + dt * F(U*)
    L_star = get_F_operator(phi_star, u, v, dx, dy)
    phi_double_star = phi_star + dt * L_star

    # Apply BCs to phi_double_star here
    phi_double_star[:, 0] = phi_double_star[:, -2];    phi_double_star[:, -1] = phi_double_star[:, 1]

    # Step 3: U^{n+1} = 0.5 * (U^n + U**)
    phi_next = 0.5 * (phi_n + phi_double_star)

    return phi_next


# Video parameters
'''output_dir = "frames"
output_dir_2 = "zoomed in"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_2, exist_ok=True)
save_every = 50'''
root_dir = "simulation_data_KHI"
sub_dirs = ["density", "velocity_u", "velocity_v", "H_sm"]
for sub in sub_dirs:
    path = os.path.join(root_dir, sub)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created: {path}")

velocity_u, velocity_v, Density, Heaviside = [], [], [], []
save_every = 50

# Parameters
L_x = 8.0  # Length of domain in x-direction
L_y = 3.0 # Length of domain in y-direction
N_points_x = 801  # Number of grid points in x-direction
N_points_y = 601  # Number of grid points in y-direction
dt = 0.0005  # Time step
rho_1, rho_2 = 950, 1000    #toluene and water
mu_1, mu_2 = 0.005, 0.008   #toluene and water
hor_vel = 0.4  # Horizontal velocity in the x direction
N_iterations = 14000  # Number of vel iterations (divisible by interval)
N_pressure_iter = 50  # Number of iterations for pressure poisson eq
#N_tau = 1  # Number of interface correction iterations
u_residual, v_residual, pressure_residual, cont_residual, phi_residual = [], [], [], [], []  # residuals L2 norm
u_residual1, v_residual1, pressure_residual1, cont_residual1 = [], [], [], []  # residuals L1 norm
interval = 200  # interval for residual calculations
#p_max = 100.0
g_y = 9.81
rho_ref, mu_ref = rho_2, 0.001     #values for water
nu_ref = mu_ref / rho_ref
u_ref = hor_vel
# Mesh creation with different resolutions in x and y
element_length_x = L_x / (N_points_x - 1)
element_length_y = L_y / (N_points_y - 1)
# coordinates (useful for init / debug)
xc = (np.arange(N_points_x) + 0.5) * element_length_x
yc = (np.arange(N_points_y) + 0.5) * element_length_y
Xc, Yc = np.meshgrid(xc, yc)

k = 1
phi_prev = initialize_phi(Xc, Yc, L_x, L_y / 2, 0.001, k)
L_ref = L_y / k
sigma = 0.001   #typical sigma (surface tension) number value for water toluene at room temperature
Re = u_ref * L_ref / nu_ref
We = rho_ref * u_ref**2 * L_ref / sigma
Fr = u_ref / np.sqrt(L_ref * g_y)
#chat gpt parameters
epsilon = 1.0 * element_length_x      # for H_sm only
eps_c   = 0.5 * element_length_x      # Olsson–Kreiss diffusion
delta_tau    = 0.1 * element_length_x
N_tau   = 2
'''
#my parameters
d = 0.1
epsilon = element_length_x**(1-d) / 2
delta_tau = element_length_x**(1+d) / 2
eps_c = 0.5 * element_length_x
'''
x = np.linspace(0.0, L_x, N_points_x)
y = np.linspace(0.0, L_y, N_points_y)
X, Y = np.meshgrid(x, y)

'''MAC (staggered) grid'''
# Initialize u_prev to match the intended shear immediately
# 1. Get the fluid distribution at the cell centers
H_prev = H_sm(phi_prev, epsilon)
# 2. Interpolate H to the u-faces (staggered locations)
H_at_u = interp_cell_to_u_face(H_prev)
# 3. Apply the shear based on the interpolated H
# Top fluid (H > 0.5) gets +hor_vel, Bottom fluid (H < 0.5) gets -hor_vel
u_prev = -hor_vel + 2.0 * hor_vel * H_at_u
v_prev = np.zeros((N_points_y + 1, N_points_x))
p_prev = np.zeros((N_points_y, N_points_x))
density = H_prev * rho_1 + (1 - H_prev) * rho_2
p_iter = np.zeros((N_points_y, N_points_x))
p_next = np.zeros((N_points_y, N_points_x))
phi_next = np.zeros((N_points_y, N_points_x))
H_next = np.zeros_like(H_prev)

# Create interior mask (exclude boundaries and cylinder) for residual calculations
interior_mask = np.ones((N_points_y, N_points_x), dtype=bool)
interior_mask[0, :] = False  # boundaries
interior_mask[-1, :] = False
interior_mask[:, 0] = False
interior_mask[:, -1] = False


print(f"Simulation started. Instability wavenumber: {k}")

'''Navier-Stokes equations - Incompressible'''
for _ in tqdm(range(N_iterations)):

    t = _ * dt

    # Update properties based on current phi
    density = H_prev * rho_1 + (1 - H_prev) * rho_2
    density_u = interp_cell_to_u_face(density)
    density_v = interp_cell_to_v_face(density)
    viscosity = H_prev * mu_1 + (1 - H_prev) * mu_2
    viscosity_u = interp_cell_to_u_face(viscosity)
    viscosity_v = interp_cell_to_v_face(viscosity)
    # Compute curvature & normal vector (if using surface tension)
    H_dx = central_difference_x(H_prev, element_length_x)
    H_dy = central_difference_y(H_prev, element_length_y)
    grad_mag = np.sqrt(H_dx ** 2 + H_dy ** 2 + 1e-12)
    nx = H_dx / grad_mag
    ny = H_dy / grad_mag
    curvature = -(central_difference_x(nx, element_length_x) + central_difference_y(ny, element_length_y))
    # Assuming curvature, phi_dx, and phi_dy are all at cell centers (shape: 151, 402)
    f_sv_x_center = curvature * H_dx
    f_sv_y_center = curvature * H_dy
    f_sv_at_x = interp_cell_to_u_face(f_sv_x_center)
    f_sv_at_y = interp_cell_to_v_face(f_sv_y_center)

    # Setting variables for NS
    du_prev_dx = upwind2_x(u_prev, u_prev, element_length_x)
    du_prev_dy = central_difference_y(u_prev, element_length_y)
    dv_prev_dx = central_difference_x(v_prev, element_length_x)
    dv_prev_dy = upwind2_y(v_prev, v_prev, element_length_y)
    laplace_u_prev = laplace_mac_u(u_prev, element_length_x, element_length_y)
    laplace_v_prev = laplace_mac_v(v_prev, element_length_x, element_length_y)
    u_at_v = interp_u_to_v_face(u_prev)
    v_at_u = interp_v_to_u_face(v_prev)

    Q = (-u_prev * du_prev_dx - v_at_u * du_prev_dy
         + (mu_ref / density_u) * (laplace_u_prev)
         + (sigma / density_u) * (f_sv_at_x))

    R = (-u_at_v * dv_prev_dx - v_prev * dv_prev_dy
         + (mu_ref / density_v) * (laplace_v_prev)
         + (sigma / density_v) * (f_sv_at_y))

    '''Temporal discretization using RK2 scheme - "Prediction step" -- This is accurate for FDM also (using 2nd order upwind scheme) '''
    ''' Solving the momentum equations without the pressure field '''
    # k1 = dt * f(t_n, y_n) -> where f is the RHS of the NS eq
    k1_u = dt * Q
    k1_v = dt * R

    u_temp = u_prev + k1_u  # y_n + k1
    v_temp = v_prev + k1_v  # y_n + k1

    u_temp[0, :] = 0.0                   ; v_temp[0, :] = 0.0
    u_temp[:, 0] = u_temp[:, -2]         ; v_temp[:, 0] = v_temp[:, -2]
    u_temp[:, -1] = u_temp[:, 1]         ; v_temp[:, -1] = v_temp[:, 1]
    u_temp[-1, :] = 0.0                  ; v_temp[-1, :] = 0.0

    du_temp_dx = upwind2_x(u_temp, u_temp, element_length_x)
    du_temp_dy = central_difference_y(u_temp, element_length_y)
    dv_temp_dx = central_difference_x(v_temp, element_length_x)
    dv_temp_dy = upwind2_y(v_temp, v_temp, element_length_y)
    laplace_u_temp = laplace_mac_u(u_temp, element_length_x, element_length_y)
    laplace_v_temp = laplace_mac_v(v_temp, element_length_x, element_length_y)
    u_temp_at_v = interp_u_to_v_face(u_temp)
    v_temp_at_u = interp_v_to_u_face(v_temp)

    Q_temp = (-u_temp * du_temp_dx - v_temp_at_u * du_temp_dy +
              (mu_ref / density_u) * (laplace_u_temp) +
              (sigma / density_u) * f_sv_at_x)
    R_temp = (-u_temp_at_v * dv_temp_dx - v_temp * dv_temp_dy +
              (mu_ref / density_v) * (laplace_v_temp) +
              (sigma / density_v) * f_sv_at_y)

    # k2 = dt * f(t_n + dt, y_n + k1)
    k2_u = dt * Q_temp
    k2_v = dt * R_temp

    # Runge-Kutta 2nd order
    u_tent = u_prev + 0.5 * (k1_u + k2_u)
    v_tent = v_prev + 0.5 * (k1_v + k2_v)
    '''Boundary conditions for tentative velocities'''
    u_tent[0, :] = 0.0              ; v_tent[0, :] = 0.0
    u_tent[:, 0] = u_tent[:, -2]    ; v_tent[:, 0] = v_tent[:, -2]
    u_tent[:, -1] = u_tent[:, 1]    ; v_tent[:, -1] = v_tent[:, 1]
    u_tent[-1, :] = 0.0             ; v_tent[-1, :] = 0.0

    '''υπολογιζω τις παραγωγους των ενδιαμεσων ταχυτητων '''
    du_tent_dx = central_difference_x(u_tent, element_length_x)
    du_tent_dy = central_difference_y(u_tent, element_length_y)
    dv_tent_dx = central_difference_x(v_tent, element_length_x)
    dv_tent_dy = central_difference_y(v_tent, element_length_y)
    div_tent = divergence_mac(u_tent, v_tent, element_length_x, element_length_y)
    laplace_u_tent = laplace_mac_u(u_tent, element_length_x, element_length_y)
    laplace_v_tent = laplace_mac_v(v_tent, element_length_x, element_length_y)
    u_tent_at_v = interp_u_to_v_face(u_tent)
    v_tent_at_u = interp_v_to_u_face(v_tent)
    '''Pressure correction by solving the pressure poisson eq - "Correction step" '''
    # Ensure dQ and dR both result in (ny, nx)
    Q_tent = (-u_tent * du_tent_dx - v_tent_at_u * du_tent_dy +
              (mu_ref / density_u) * (laplace_u_tent) +
              (sigma / density_u) * f_sv_at_x)
    R_tent = (-u_tent_at_v * dv_tent_dx - v_tent * dv_tent_dy +
              (mu_ref / density_v) * (laplace_v_tent) +
              (sigma / density_v) * f_sv_at_y)

    dQ_dx = (Q_tent[:, 1:] - Q_tent[:, :-1]) / element_length_x
    dR_dy = (R_tent[1:, :] - R_tent[:-1, :]) / element_length_y
    '''Pressure correction by solving the variable-density Poisson eq'''
    # The RHS is (1/dt) * div(u_tent)
    rhs = div_tent / dt
    # We want to solve: div( (1/rho) * grad(p) ) = (1/dt) * div(u_tent)
    # Pre-calculate factors for the Gauss-Seidel iteration
    idx2 = 1.0 / (element_length_x ** 2)
    idy2 = 1.0 / (element_length_y ** 2)

    '''Pressure correction by solving the variable-density Poisson eq'''
    # div_tent shape: (ny, nx)
    p_next[:] = p_prev
    for iter in range(N_pressure_iter):
        p_iter[:] = p_next.copy()

        # Slicing the staggered density to match the interior p[1:-1, 1:-1]
        # Ce (East): uses density at the face between (i,j) and (i+1,j)
        Ce = idx2 / density_u[1:-1, 2:-1]
        # Cw (West): uses density at the face between (i,j) and (i-1,j)
        Cw = idx2 / density_u[1:-1, 1:-2]
        # Cn (North): uses density at the face between (i,j) and (i,j+1)
        Cn = idy2 / density_v[2:-1, 1:-1]
        # Cs (South): uses density at the face between (i,j) and (i,j-1)
        Cs = idy2 / density_v[1:-2, 1:-1]

        # Center coefficient: sum of the neighbor weights
        Cp = -(Ce + Cw + Cn + Cs)

        # Gauss-Seidel / Jacobi update
        p_next[1:-1, 1:-1] = (rhs[1:-1, 1:-1] - (
                Ce * p_iter[1:-1, 2:] +
                Cw * p_iter[1:-1, 0:-2] +
                Cn * p_iter[2:, 1:-1] +
                Cs * p_iter[0:-2, 1:-1]
                )) / Cp

        # Pressure boundary conditions (matching your X-periodic, Y-wall setup)
        p_next[:, 0] = p_next[:, -2]  # Periodic West
        p_next[:, -1] = p_next[:, 1]  # Periodic East
        p_next[0, :] = p_next[1, :]  # Neumann South
        p_next[-1, :] = p_next[-2, :]  # Neumann North


        if np.max(np.abs(p_next - p_iter)) < 1e-7:
            break

    '''υπολογιχω τις παραγωγους της πιεσης για την διορθωση των ταχυτητων'''
    dp_next_dx = grad_p_x_mac(p_next, element_length_x)
    dp_next_dy = grad_p_y_mac(p_next, element_length_y)

    '''Correct the velocities in order to stay incompressible'''
    u_next = u_tent - (dt / density_u) * dp_next_dx
    v_next = v_tent - (dt / density_v) * dp_next_dy

    # Boundary conditions for new velocities
    u_next[0, :] = 0.0              ; v_next[0, :] = 0.0
    u_next[:, 0] = u_next[:, -2]    ; v_next[:, 0] = v_next[:, -2]
    u_next[:, -1] = u_next[:, 1]    ; v_next[:, -1] = v_next[:, 1]
    u_next[-1, :] = 0.0             ; v_next[-1, :] = 0.0

    #Interface advection
    H_tent = advect_phi_tvd(H_prev, u_next, v_next, dt, element_length_x, element_length_y)
    H_tent_dx = central_difference_x(H_tent, element_length_x)
    H_tent_dy = central_difference_y(H_tent, element_length_y)
    grad_mag_int = np.sqrt(H_tent_dx ** 2 + H_tent_dy ** 2 + 1e-12)
    nx_int = H_tent_dx / grad_mag_int
    ny_int = H_tent_dy / grad_mag_int
    #interface correction step
    H_next[:] = H_tent
    for _tau in range(N_tau):
        H_next = interface_correction(H_next, element_length_x, element_length_y, nx_int, ny_int,
                                       eps_c=eps_c, dtau=delta_tau)


    # ---- After correction ----
    if _ % interval == 0:
        # DIAGNOSTIC
        '''du_next_dx = upwind2_x(u_next, u_next, element_length_x)
        du_next_dy = central_difference_y(u_next, element_length_y)
        dv_next_dx = central_difference_x(v_next, element_length_x)
        dv_next_dy = upwind2_y(v_next, v_next, element_length_y)
        laplace_u_next = laplace_mac_u(u_next, element_length_x, element_length_y)
        laplace_v_next = laplace_mac_v(v_next, element_length_x, element_length_y)
        div_next = divergence_mac(u_next, v_next, element_length_x, element_length_y)
        u_next_at_v = interp_u_to_v_face(u_next)
        v_next_at_u = interp_v_to_u_face(v_next)

        rhs_interior = rhs[1:-1, 1:-1]
        print(f"\nIteration {_}")
        # Check different regions
        print("=== DIVERGENCE DISTRIBUTION (steady state) ===")
        print(f"Interior [10:-10, 10:-10]: max={np.max(np.abs(div_next[10:-10, 10:-10]))}")
        print(f"Near inlet [:, 0:5]: max={np.max(np.abs(div_next[:, 0:5]))}")
        print(f"Near outlet [:, -5:]: max={np.max(np.abs(div_next[:, -5:]))}")
        print(f"Near top [0:5, :]: max={np.max(np.abs(div_next[0:5, :]))}")
        print(f"Near bottom [-5:, :]: max={np.max(np.abs(div_next[-5:, :]))}")

        print(f"Div_max tent velocities: {np.max(np.abs(div_tent))}")
        print(f"Div_next (du+dv): [{np.min(div_next):.3f}, {np.max(div_next):.3f}]")
        print(f"RHS_next min/max/mean: {np.min(rhs_interior):.2e}, {np.max(rhs_interior):.2e}, "
              f"{np.mean(np.abs(rhs_interior)):.2e}")
        print(f"RHS_next sum (should be ~0 / idk): {np.sum(rhs_interior):.3e}")
        print(f"Mean div_next: {np.mean(div_next[1:-1, 1:-1])}")  # Should be ~1e-6 or smaller
        print(f"Max abs div_next (interior): {np.max(np.abs(div_next[1:-1, 1:-1]))}")
        # Check pressure Poisson residual
        residual = laplace(p_next, element_length_x, element_length_y) - rhs
        print(f"Poisson residual (raw): {np.max(np.abs(residual))}")
        print(f"Velocity magnitude: u_next={np.max(np.abs(u_next)):.2f}, "
              f"v_next={np.max(np.abs(v_next)):.2f}")'''
        '''RESIDUALS - START'''
        # Compute the residual of the Poisson equation: ∇²p = rhs
        # Residual = rhs - ∇²p (should be close to zero if well converged)
        '''laplace_p = laplace(p_next, element_length_x, element_length_y)
        pressure_res_raw = rhs - laplace_p
        # Calculate L2 and L1 norms only in the interior (fluid region)
        N = np.sum(interior_mask)
        L2_pressure = np.sqrt(np.sum(pressure_res_raw[interior_mask] ** 2) / N)
        L1_pressure = np.sum(np.abs(pressure_res_raw[interior_mask]) / N)
        pressure_residual.append(L2_pressure)
        pressure_residual1.append(L1_pressure)
        # backward difference for time derivative
        du_dt = (u_next - u_prev) / dt
        dv_dt = (v_next - v_prev) / dt
        # Putting the derivatives in the NS eq to find the raw residuals of the velocities
        u_res_raw = du_dt + (-u_next * du_next_dx - v_next_at_u * du_next_dy +
         1.0 / (density_u) * (mu_ref * laplace_u_next) + 1.0 / (density_u) * f_sv_at_x)
        v_res_raw = dv_dt + (-u_next_at_v * dv_next_dx - v_next * dv_next_dy +
         1.0 / (density_v) * (mu_ref * laplace_v_next) + 1.0 / (density_v) * f_sv_at_y +
                             g_y * (density_v - rho_ref) / density_v)
        u_res_raw_c = u_to_cell(u_res_raw)
        v_res_raw_c = v_to_cell(v_res_raw)
        dH_dt = (H_next - H_prev) / dt
        phi_res_raw = dH_dt - get_F_operator(H_next, u_next, v_next,
                                             element_length_x, element_length_y)
        # continuity eq residual
        continuity_raw = divergence_mac(u_next, v_next, element_length_x, element_length_y)
        # L2 norm
        N = np.sum(interior_mask)
        L2_u = np.sqrt(np.sum(u_res_raw_c[interior_mask] ** 2) / N)
        L2_v = np.sqrt(np.sum(v_res_raw_c[interior_mask] ** 2) / N)
        L2_cont = np.sqrt(np.sum(continuity_raw[interior_mask] ** 2) / N)
        L2_phi = np.sqrt(np.sum(phi_res_raw[interior_mask] ** 2) / N)
        u_residual.append(L2_u)
        v_residual.append(L2_v)
        cont_residual.append(L2_cont)
        phi_residual.append(L2_phi)
        # L1 norm for comparison
        L1_u = np.sum(np.abs(u_res_raw_c[interior_mask]) / N)
        L1_v = np.sum(np.abs(v_res_raw_c[interior_mask]) / N)
        L1_cont = np.sum(np.abs(continuity_raw[interior_mask]) / N)
        u_residual1.append(L1_u)
        v_residual1.append(L1_v)
        cont_residual1.append(L1_cont)'''
        '''RESIDUALS - END'''
    '''video making'''
    if _ % save_every == 0:
        '''fig, ax = plt.subplots(2, 2, figsize=(12, 4))
        ax = ax.flatten()

        # Pressure
        c1 = ax[0].contourf(Xc, Yc, p_next, levels=51, cmap='RdBu_r')
        ax[0].contour(Xc, Yc, H_next, levels=[0.5], colors='black', linewidths=1)
        plt.colorbar(c1, ax=ax[0])
        ax[0].set_title(f'Pressure (t={t:.2f}s)')
        ax[0].set_aspect('equal')

        # Velocity
        # 1. Bring u and v to cell centers
        u_c = u_to_cell(u_next)
        v_c = v_to_cell(v_next)

        # 2. Now they have the same shape (ny, nx) and can be combined
        vel_mag = np.sqrt(u_c ** 2 + v_c ** 2)
        max_vel = np.max(vel_mag)
        vel_levels = np.linspace(0, max_vel, 51)
        c2 = ax[1].contourf(Xc, Yc, vel_mag, levels=vel_levels, cmap='viridis')
        ax[1].contour(Xc, Yc, H_next, levels=[0.5], colors='black', linewidths=1)
        ax[1].streamplot(X, Y, u_c, v_c, density=1, color="white", linewidth=0.5)
        plt.colorbar(c2, ax=ax[1])
        ax[1].set_title('Velocity Magnitude')
        ax[1].set_aspect('equal')

        # Density plot
        c3 = ax[2].contourf(Xc, Yc, density, levels=51, cmap='RdBu')
        ax[2].contour(Xc, Yc, H_next, levels=[0.5], colors='black', linewidths=1)
        plt.colorbar(c3, ax=ax[2])
        ax[2].set_title('Density')
        ax[2].set_aspect('equal')

        fig.delaxes(ax[3])
        plt.tight_layout()
        fig.savefig(f"{output_dir}/streamlines_{_:04d}.png", dpi=120, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(2, 2, figsize=(12, 4))
        ax = ax.flatten()
        x1, x2 = 2.5, 5.5  # meters
        y1, y2 = 1.0, 2.0
        ax[0].set_xlim(x1, x2)
        ax[0].set_ylim(y1, y2)

        ax[1].set_xlim(x1, x2)
        ax[1].set_ylim(y1, y2)

        ax[2].set_xlim(x1, x2)
        ax[2].set_ylim(y1, y2)

        # Pressure
        c1 = ax[0].contourf(Xc, Yc, p_next, levels=51, cmap='RdBu_r')
        ax[0].contour(Xc, Yc, H_next, levels=[0.5], colors='black', linewidths=1)
        plt.colorbar(c1, ax=ax[0])
        ax[0].set_title(f'Pressure zoom-in (t={t:.2f}s)')
        ax[0].set_aspect('equal')

        # 2. Now they have the same shape (ny, nx) and can be combined
        c2 = ax[1].contourf(Xc, Yc, vel_mag, levels=vel_levels, cmap='viridis')
        ax[1].contour(Xc, Yc, H_next, levels=[0.5], colors='black', linewidths=1)
        ax[1].streamplot(X, Y, u_c, v_c, density=1, color="white", linewidth=0.5)
        plt.colorbar(c2, ax=ax[1])
        ax[1].set_title('Velocity Magnitude zoom-in')
        ax[1].set_aspect('equal')

        #Density plot
        c3 = ax[2].contourf(Xc, Yc, density, levels=51, cmap='RdBu')
        ax[2].contour(Xc, Yc, H_next, levels=[0.5], colors='black', linewidths=1)
        plt.colorbar(c3, ax=ax[2])
        ax[2].set_title('Density zoom-in')
        ax[2].set_aspect('equal')

        fig.delaxes(ax[3])
        plt.tight_layout()
        fig.savefig(f"{output_dir_2}/streamlines_{_:04d}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)'''
        velocity_u[:] = u_next
        velocity_v[:] = v_next
        Density[:] = density
        Heaviside[:] = H_next
        files = {f"density": Density, f"velocity_u": velocity_u,
                 f"velocity_v": velocity_v, f"H_sm": Heaviside}
        for name, matrix in files.items():
            filename = f"{name}_{_:03d}.txt"
            filepath = os.path.join(root_dir, name, filename)
            # fmt='%.6e' uses scientific notation to preserve precision
            # delimiter=' ' keeps it clean with spaces between columns
            np.savetxt(filepath, matrix, fmt='%.6e', delimiter=' ')



    # Advance in time
    u_prev[:] = u_next  # this way we don't create a new array each iteration, just copy the new values on the old array
    v_prev[:] = v_next
    p_prev[:] = p_next
    H_prev[:] = H_next


# After loop -- video:
print(f"Saved frames to {root_dir}")

# Pressure & velocity field & vorticity plot
plt.figure(1, figsize=(12, 4))
plt.subplot(2, 2, 1)
# Pressure plot
p_max = np.max(p_prev)
pressure_levels = np.linspace(-p_max, p_max, 51)
contour = plt.contourf(X, Y, p_prev, levels=pressure_levels, cmap='RdYlBu_r')
plt.colorbar(contour, label='Pressure')
plt.contour(X, Y, H_prev, levels=[0.5], colors='black', linewidths=1)
# plt.quiver(X[::step_y, ::step_x], Y[::step_y, ::step_x], u_next[::step_y, ::step_x], v_next[::step_y, ::step_x],
#               scale=100, alpha=0.7, color="black")
plt.title("Pressure Field")
plt.xlabel("x")
plt.ylabel("y")
# Add cylinder in the plot
plt.axis('equal')

# Velocity magnitude plot
plt.subplot(2, 2, 2)
# Before plotting results
u_plot = u_to_cell(u_prev)
v_plot = v_to_cell(v_prev)
vel_magnitude = np.sqrt(u_plot ** 2 + v_plot ** 2)
max_vel = np.max(vel_magnitude)  # Add this line!
vel_levels = np.linspace(0, max_vel, 51)  # Fix this!
# For streamplots, X and Y must match the shapes of u_plot and v_plot
plt.xlim(0, L_x)
plt.ylim(0, L_y)
contour2 = plt.contourf(X, Y, vel_magnitude, levels=vel_levels, cmap='viridis')
plt.colorbar(contour2, label='Velocity Magnitude')
plt.contour(X, Y, H_prev, levels=[0.5], colors='black', linewidths=1)
plt.streamplot(X, Y, u_plot, v_plot, density=1, color="white", linewidth=0.5)
plt.title("Velocity Magnitude with Streamlines")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')

#Density plot
plt.subplot(2, 2, 3)
contour3 = plt.contourf(Xc, Yc, density, levels=51, cmap='RdBu')
plt.contour(Xc, Yc, H_next, levels=[0.5], colors='black', linewidths=1)
plt.colorbar(contour3, label='Density')
plt.title('Density')
plt.axis('equal')

plt.subplot(2, 2, 4)
# Adding legend
plt.plot([], [], '', label=f"U_hor = {hor_vel}")
plt.plot([], [], '', label=f"μ = {mu_ref}")      #need to correct this one
plt.plot([], [], '', label=f"Rey = {Re:.2f}")
plt.plot([],[], '', label=f"Web = {We:.2f}")
plt.legend(loc='center', frameon=True, handlelength=0)
plt.axis('equal')
plt.show()

# Velocity and continuity residuals plot
plt.figure(figsize=(10, 5))
iterations = np.arange(len(u_residual))
plt.semilogy(iterations * interval, u_residual, label='u - residual L2')
plt.semilogy(iterations * interval, v_residual, label='v - residual L2')
plt.semilogy(iterations * interval, cont_residual, label='Continuity residual L2')
plt.semilogy(iterations * interval, pressure_residual, label='Pressure residual L2')
plt.semilogy(iterations * interval, phi_residual, label='Φ (H) residual L2')
plt.semilogy(iterations * interval, u_residual1, label='u - residual L1')
plt.semilogy(iterations * interval, v_residual1, label='v - residual L1')
plt.semilogy(iterations * interval, cont_residual1, label='Continuity residual L1')
plt.semilogy(iterations * interval, pressure_residual1, label='Pressure residual L1')
plt.title('Residuals')
plt.xlabel('Iterations')
plt.ylabel('Velocity residuals (L2 & L1 norms)')
plt.legend()
plt.grid(True)
plt.show()
