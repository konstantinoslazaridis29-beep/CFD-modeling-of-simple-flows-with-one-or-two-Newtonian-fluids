import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.ndimage import binary_dilation

def divergence(u, v, dx, dy):
    """Compute the discrete divergence ∂u/∂x + ∂v/∂y."""
    div = np.zeros_like(u)
    div[1:-1, 1:-1] = ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                       (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))
    return div

# Defining functions
'''2nd order upwind for the convection terms - same as the 2nd order accurate forward diff'''
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

# Parameters
L_x = 4.0  # Length of domain in x-direction
L_y = 2.5 # Length of domain in y-direction
N_points_x = 401  # Number of grid points in x-direction
N_points_y = 251  # Number of grid points in y-direction
dt = 0.0005  # Time step
density = 1.0  # Density of air
viscosity = 0.001  # Kinematic Viscosity
hor_vel = 3.5  # Horizontal velocity in the x direction (left boundary)
N_iterations = 10000  # Number of vel iterations (divisible by interval)
N_pressure_iter = 50  # Number of iterations for pressure poisson eq
R = 0.1  # radius of cylinder
x_cylinder = 1  # x_cylinder
y_cylinder = L_y / 2  # y_cylinder
u_residual, v_residual, pressure_residual, cont_residual = [], [], [], []  # residuals L2 norm
u_residual1, v_residual1, pressure_residual1, cont_residual1 = [], [], [], []  # residuals L1 norm
interval = 200  # interval for residual calculations
p_max = 20.0


# Cylinder placement
theta = np.linspace(0, 2 * np.pi, 100)
x_cyl = x_cylinder + R * np.cos(theta)
y_cyl = y_cylinder + R * np.sin(theta)

# Mesh creation with different resolutions in x and y
element_length_x = L_x / (N_points_x - 1)
element_length_y = L_y / (N_points_y - 1)

x = np.linspace(0.0, L_x, N_points_x)
y = np.linspace(0.0, L_y, N_points_y)

X, Y = np.meshgrid(x, y)

'''Setting every initial condition for every cell to zero, except for the horizontal velocity'''
u_prev = np.zeros_like(X) * hor_vel
'''faster convergence to steady state, pressure solver more efficient when velocity field is closer to divergence-free
    physical realism, numerical stability, computational cost'''
v_prev = np.zeros_like(X)
p_prev = np.zeros_like(X)
p_iter = np.zeros_like(X)
p_next = np.zeros_like(X)
p_new = np.zeros_like(X)

# Create cylinder mask for efficient boundary application
'''This is the correct one!'''
cylinder_mask = np.zeros_like(X, dtype=bool)
for i in range(N_points_y):
    for j in range(N_points_x):
        x_d = j * element_length_x
        y_d = i * element_length_y
        distance = np.sqrt((x_d - x_cylinder) ** 2 + (y_d - y_cylinder) ** 2)
        if distance <= R:
            cylinder_mask[i, j] = True

# surface and fluid masks
fluid_mask = ~cylinder_mask
surface_mask = fluid_mask & (binary_dilation(cylinder_mask))
surface_indices = np.argwhere(surface_mask)
found_neighbor = np.zeros_like(X, dtype=bool)
angle_phi = []
pi_quarter = np.pi / 4
three_pi_quarter = 3 * np.pi / 4

for i, j in surface_indices:
    xj = j * element_length_x
    yi = i * element_length_y
    nx = xj - x_cylinder
    ny = yi - y_cylinder
    angle = np.arctan2(ny, nx)
    angle_phi.append(angle)
    # Select direction based on angle
    if -pi_quarter <= angle <= pi_quarter:  # right side
        found_neighbor[i, j + 1] = True
    elif pi_quarter < angle < three_pi_quarter:  # top side
        found_neighbor[i + 1, j] = True
    elif angle <= -three_pi_quarter or angle >= three_pi_quarter:  # left side
        found_neighbor[i, j - 1] = True
    elif -three_pi_quarter < angle < -pi_quarter:  # bottom side
        found_neighbor[i - 1, j] = True
    else:
        continue  # skip if undefined

neighbor_indices = np.argwhere(found_neighbor)

# Create interior mask (exclude boundaries and cylinder) for residual calculations
# -- we want to calculate the residuals only in the interior of the control volume
interior_mask = np.ones((N_points_y, N_points_x), dtype=bool)
interior_mask[0, :] = False  # boundaries
interior_mask[-1, :] = False
interior_mask[:, 0] = False
interior_mask[:, -1] = False
interior_mask[cylinder_mask] = False

# Ramp-up settings
T_ramp = 1.5      # seconds until inlet reaches hor_vel
def inlet_velocity(t):
    if t >= T_ramp:
        return hor_vel
    return hor_vel * (0.5 * (1 - np.cos(np.pi * t / T_ramp)))


'''Navier-Stokes equations - Incompressible'''
for _ in tqdm(range(N_iterations)):

    t = _ * dt
    U_in = inlet_velocity(t)

    # Setting the variable names
    '''x and y derivatives needed for later'''
    du_prev_dx = upwind2_x(u_prev, u_prev, element_length_x)
    du_prev_dy = upwind2_y(u_prev, v_prev, element_length_y)
    dv_prev_dx = upwind2_x(v_prev, u_prev, element_length_x)
    dv_prev_dy = upwind2_y(v_prev, v_prev, element_length_y)
    laplace_u_prev = laplace(u_prev, element_length_x, element_length_y)
    laplace_v_prev = laplace(v_prev, element_length_x, element_length_y)

    '''Temporal discretization using RK2 scheme - "Prediction step" -- This is accurate for FDM also (using 2nd order upwind scheme) '''
    ''' Solving the momentum equations without the pressure field '''
    '''New time discretization (Heun / RK2)'''
    # k1 = dt * f(t_n, y_n) -> where f is the RHS of the NS eq
    k1_u = dt * (-(u_prev * du_prev_dx + v_prev * du_prev_dy) + viscosity * laplace_u_prev)
    k1_v = dt * (-(u_prev * dv_prev_dx + v_prev * dv_prev_dy) + viscosity * laplace_v_prev)

    u_temp = u_prev + k1_u  # y_n + k1
    v_temp = v_prev + k1_v  # y_n + k1

    u_temp[0, :] = u_temp[1, :]
    u_temp[:, 0] = U_in
    u_temp[:, -1] = u_temp[:, -2]
    u_temp[-1, :] = u_temp[-2, :]
    v_temp[0, :] = 0.0
    v_temp[:, 0] = 0.0
    v_temp[:, -1] = v_temp[:, -2]
    v_temp[-1, :] = 0.0
    u_temp[cylinder_mask] = 0.0
    v_temp[cylinder_mask] = 0.0

    du_temp_dx = upwind2_x(u_temp, u_temp, element_length_x)
    du_temp_dy = upwind2_y(u_temp, v_temp, element_length_y)
    dv_temp_dx = upwind2_x(v_temp, u_temp, element_length_x)
    dv_temp_dy = upwind2_y(v_temp, v_temp, element_length_y)
    laplace_u_temp = laplace(u_temp, element_length_x, element_length_y)
    laplace_v_temp = laplace(v_temp, element_length_x, element_length_y)

    # k2 = dt * f(t_n + dt, y_n + k1)
    k2_u = dt * (-(u_temp * du_temp_dx + v_temp * du_temp_dy) + viscosity * laplace_u_temp)
    k2_v = dt * (-(u_temp * dv_temp_dx + v_temp * dv_temp_dy) + viscosity * laplace_v_temp)

    # Runge-Kutta 2nd order
    u_tent = u_prev + 0.5 * (k1_u + k2_u)
    v_tent = v_prev + 0.5 * (k1_v + k2_v)
    '''Boundary conditions for tentative velocities'''
    u_tent[0, :] = u_tent[1, :]
    u_tent[:, 0] = U_in
    u_tent[:, -1] = u_tent[:, -2]
    u_tent[-1, :] = u_tent[-2, :]
    v_tent[0, :] = 0.0
    v_tent[:, 0] = 0.0
    v_tent[:, -1] = v_tent[:, -2]
    v_tent[-1, :] = 0.0

    u_tent[cylinder_mask] = 0.0  # Dirichlet BCs - correct
    v_tent[cylinder_mask] = 0.0

    '''derivatives of tentative velocities '''
    du_tent_dx = central_difference_x(u_tent, element_length_x)
    du_tent_dy = central_difference_y(u_tent, element_length_y)
    dv_tent_dx = central_difference_x(v_tent, element_length_x)
    dv_tent_dy = central_difference_y(v_tent, element_length_y)
    div_tent = divergence(u_tent, v_tent, element_length_x, element_length_y)
    div_temp = divergence(u_temp, v_temp, element_length_x, element_length_y)

    '''Pressure correction by solving the pressure poisson eq - "Correction step" '''
    # calculating the rhs of pressure Poisson with the tentatvie velocities
    rhs = density * (1 / dt) * div_tent     #this is the correct way and is the same as Barba's

    p_next[:] = p_prev
    denominator = 2 * (element_length_y ** 2 + element_length_x ** 2)
    for iter in range(N_pressure_iter):
        p_iter[:] = p_next
        '''Poisson for different resolution in x and y axis'''
        p_next[1:-1, 1:-1] = ((p_iter[1:-1, 0:-2] + p_iter[1:-1, 2:]) * element_length_y**2 +
                              (p_iter[0:-2, 1:-1] + p_iter[2:, 1:-1]) * element_length_x ** 2 -
                              rhs[1:-1, 1:-1] * element_length_x**2 * element_length_y**2) / denominator

        # Pressure boundary conditions - Neumann type
        p_next[:, -1] = 0.0
        p_next[0, :] = p_next[1, :]
        p_next[:, 0] = p_next[:, 1]
        p_next[-1, :] = p_next[-2, :]

        # Neumann conditions cylinder
        for (i, j), (ni, nj) in zip(surface_indices, neighbor_indices):
            p_next[i, j] = p_next[ni, nj]

        if np.max(np.abs(p_next - p_iter)) < 1e-6:
            break


    '''pressure derivatives for velocity correction'''
    dp_next_dx = central_difference_x(p_next, element_length_x)
    dp_next_dy = central_difference_y(p_next, element_length_y)

    '''Correct the velocities in order to stay incompressible'''
    u_next = u_tent - (dt / density) * dp_next_dx
    v_next = v_tent - (dt / density) * dp_next_dy

    # Boundary conditions for new velocities
    u_next[0, :] = u_next[1, :]
    u_next[:, 0] = U_in
    u_next[:, -1] = u_next[:, -2]
    u_next[-1, :] = u_next[-2, :]
    v_next[0, :] = 0.0
    v_next[:, 0] = 0.0
    v_next[:, -1] = v_next[:, -2]
    v_next[-1, :] = 0.0

    u_next[cylinder_mask] = 0.0
    v_next[cylinder_mask] = 0.0

    # ---- After correction ----
    if _ % interval == 0:
        # DIAGNOSTIC
        du_next_dx = upwind2_x(u_next, u_next, element_length_x)
        du_next_dy = upwind2_y(u_next, v_next, element_length_y)
        dv_next_dx = upwind2_x(v_next, u_next, element_length_x)
        dv_next_dy = upwind2_y(v_next, v_next, element_length_y)
        laplace_u_next = laplace(u_next, element_length_x, element_length_y)
        laplace_v_next = laplace(v_next, element_length_x, element_length_y)
        div_next = divergence(u_next, v_next, element_length_x, element_length_y)

        rhs_interior = rhs[1:-1, 1:-1]
        '''RESIDUALS - START'''
        '''L2 method for residual calculation'''
        # Compute the residual of the Poisson equation: ∇²p = rhs
        # Residual = rhs - ∇²p (should be close to zero if well converged)
        laplace_p = laplace(p_next, element_length_x, element_length_y)

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
        u_res_raw = du_dt + u_next * du_next_dx + v_next * du_next_dy + (
                    1 / density) * dp_next_dx - viscosity * laplace_u_next
        v_res_raw = dv_dt + u_next * dv_next_dx + v_next * dv_next_dy + (
                    1 / density) * dp_next_dy - viscosity * laplace_v_next
        # continuity eq residual
        continuity_raw = divergence(u_next, v_next, element_length_x, element_length_y)
        # L2 norm
        N = np.sum(interior_mask)
        L2_u = np.sqrt(np.sum(u_res_raw[interior_mask] ** 2) / N)
        L2_v = np.sqrt(np.sum(v_res_raw[interior_mask] ** 2) / N)
        L2_cont = np.sqrt(np.sum(continuity_raw[interior_mask] ** 2) / N)
        u_residual.append(L2_u)
        v_residual.append(L2_v)
        cont_residual.append(L2_cont)
        # L1 norm for comparison
        L1_u = np.sum(np.abs(u_res_raw[interior_mask]) / N)
        L1_v = np.sum(np.abs(v_res_raw[interior_mask]) / N)
        L1_cont = np.sum(np.abs(continuity_raw[interior_mask]) / N)
        u_residual1.append(L1_u)
        v_residual1.append(L1_v)
        cont_residual1.append(L1_cont)
        '''RESIDUALS - END'''

    # Advance in time
    u_prev[:] = u_next  # this way we don't create a new array each iteration, just copy the new values on the old array
    v_prev[:] = v_next
    p_prev[:] = p_next
