"""
Density-based safe navigation (paper-faithful) for a single finite-height cylindrical obstacle.

Assumption:
- Workspace/state coordinates are in millimeters (mm).
- Cylinder height is specified physically as:
    height = 22 inches
  and is converted below to millimeters.

Implements:
- f(t), \bar f(t) from (5)-(6)
- Phi from (7), Psi from (8)
- rho from (4) with V(x)=||x-x_goal||^2
- controller k(x)=∇rho(x) from (11)
- near-goal modification from Remark 2
- constrained control from (15)-(16)

Reference: "Safe Navigation Using Density Functions" (Zheng, Narayanan, Vaidya)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# Save waypoints to a text file
def save_waypoints_to_file(waypoints, filename="cylinderobstacleworkingcodewaypoints.txt"):
    with open(filename, "w") as f:
        for i, wp in enumerate(waypoints):
            f.write(f"{i}: {wp}\n")
    print(f"Waypoints saved to {filename}")


# ============================================================
# User settings / parameters to tune
# ============================================================
x0 = np.array([180.0, -590.0, 180.0])   # mm
xg = np.array([460.0,  590.0, 630.0])   # mm

N = 20000
dt = 0.01

# Density tuning (paper Remark 3)
alpha = 0.2
theta = 1e-4
kappa = 1e8
u_max = 250.0
MAX_STEP = 2.0

# Goal neighborhood for Remark-2 modification
goal_delta = 25.0
stop_radius = 5.0


# ============================================================
# Obstacle geometry: finite-height cylinder along z
# ============================================================
c = 0.5 * (x0 + xg)
cx, cy, cz = c[0], c[1], c[2]

# Radius based on workspace size (same as your original code)
ws_min = np.minimum(x0, xg)
ws_max = np.maximum(x0, xg)
ws_size = np.linalg.norm(ws_max - ws_min)
r = 0.06 * ws_size

# Sensing radius in xy-plane
R = 3.0 * r

# Finite cylinder height: 22 inches converted to mm
inch_to_mm = 25.4
h_cyl = 22.0 * inch_to_mm        # 558.8 mm
hz_cyl = h_cyl / 2.0             # 279.4 mm

# Sensing height scaling in z
inflate_z = 1.5
Hz = inflate_z * hz_cyl

# Transition denominator
HS_DEN = max(R**2 - r**2, Hz - hz_cyl)


# ============================================================
# Smooth step functions f and fbar (paper (5)-(6))
# ============================================================
def f(t):
    """Elementary C^∞ function: f(t) = exp(-1/t) for t>0 else 0."""
    if t <= 0.0:
        return 0.0
    return np.exp(-1.0 / t)

def fprime(t):
    """Derivative of f for t>0: f'(t)=exp(-1/t)*(1/t^2), else 0."""
    if t <= 0.0:
        return 0.0
    et = np.exp(-1.0 / t)
    return et * (1.0 / (t * t))

def fbar(t):
    """Smooth step: fbar(t) = f(t) / ( f(t) + f(1-t) )."""
    a = f(t)
    b = f(1.0 - t)
    denom = a + b
    if denom == 0.0:
        return 0.0
    return a / denom

def fbarprime(t):
    """Derivative of fbar(t) using quotient rule."""
    a = f(t)
    b = f(1.0 - t)
    ap = fprime(t)
    bp = -fprime(1.0 - t)

    denom = a + b
    if denom == 0.0:
        return 0.0
    return (ap * denom - a * (ap + bp)) / (denom * denom)


# ============================================================
# Finite cylinder obstacle and sensing set
#
# Obstacle:
#   radial term:   d_xy^2 - r^2
#   vertical term: |z-cz| - hz_cyl
#   h(x) = max(radial term, vertical term)
#
# Sensing set:
#   radial term:   d_xy^2 - R^2
#   vertical term: |z-cz| - Hz
#   s(x) = max(radial sensing term, vertical sensing term)
# ============================================================
def d2_xy(x):
    return (x[0] - cx)**2 + (x[1] - cy)**2

def h_fun(x):
    radial = d2_xy(x) - r**2
    vertical = abs(x[2] - cz) - hz_cyl
    return max(radial, vertical)

def s_fun(x):
    radial = d2_xy(x) - R**2
    vertical = abs(x[2] - cz) - Hz
    return max(radial, vertical)

def grad_h(x):
    radial = d2_xy(x) - r**2
    vertical = abs(x[2] - cz) - hz_cyl

    if radial >= vertical:
        return np.array([
            2.0 * (x[0] - cx),
            2.0 * (x[1] - cy),
            0.0
        ])
    else:
        return np.array([
            0.0,
            0.0,
            np.sign(x[2] - cz)
        ])


# ============================================================
# Phi, Psi per (7)-(8) and their gradients
# ============================================================
def Phi_and_gradPhi(x):
    hx = h_fun(x)
    sx = s_fun(x)

    # inside obstacle
    if hx <= 0.0:
        return 0.0, np.zeros(3)

    # outside sensing region
    if sx > 0.0:
        return 1.0, np.zeros(3)

    # transition region
    tau = hx / HS_DEN
    phi = fbar(tau)
    dphi_dtau = fbarprime(tau)

    gtau = grad_h(x) / HS_DEN
    gphi = dphi_dtau * gtau

    return phi, gphi

def Psi_and_gradPsi(x):
    Phi, gPhi = Phi_and_gradPhi(x)
    Psi = Phi + theta
    gPsi = gPhi
    return Psi, gPsi


# ============================================================
# Density rho(x)=Psi(x)/V(x)^alpha, V=||x-x_goal||^2
# ============================================================
def V_and_gradV(x):
    e = x - xg
    V = float(np.dot(e, e))
    gV = 2.0 * e
    return V, gV

def rho_and_gradrho(x, V_eps=1e-9):
    Psi, gPsi = Psi_and_gradPsi(x)
    V, gV = V_and_gradV(x)
    V = max(V, V_eps)

    Va = V**alpha
    rho = Psi / Va
    grad_rho = (gPsi / Va) - (alpha * Psi * gV) / (V**(alpha + 1.0))
    return rho, grad_rho


# ============================================================
# Near-goal modification (Remark 2)
# ============================================================
def controller_u(x):
    _, grad_rho = rho_and_gradrho(x)

    e = x - xg
    dist = np.linalg.norm(e)

    tau = (goal_delta - dist) / goal_delta
    chi = fbar(tau)

    u = (1.0 - chi) * grad_rho - chi * e

    # global speed gain
    u = kappa * u

    # saturation
    infn = np.max(np.abs(u))
    if infn > u_max:
        u = (u / infn) * u_max

    return u


# ============================================================
# Simulate (Euler) with step cap
# ============================================================
traj = np.zeros((N, 3), dtype=float)
traj[0] = x0

for k in range(N - 1):
    xk = traj[k]
    u = controller_u(xk)

    step = dt * u
    nstep = np.linalg.norm(step)
    if nstep > MAX_STEP:
        step = step * (MAX_STEP / nstep)

    traj[k + 1] = xk + step

    if np.linalg.norm(traj[k + 1] - xg) < stop_radius:
        traj = traj[:k + 2]
        break


# ============================================================
# Diagnostics: did we enter obstacle?
# ============================================================
inside = np.array([
    (d2_xy(x) <= r**2) and (abs(x[2] - cz) <= hz_cyl)
    for x in traj
])

dxy = np.sqrt((traj[:, 0] - cx)**2 + (traj[:, 1] - cy)**2)

print("traj shape:", traj.shape)
print("entered obstacle? :", bool(np.any(inside)))
print("min(dxy - r):", float(np.min(dxy - r)))
print("workspace assumed units: mm")
print("cylinder radius [mm]:", float(r))
print("cylinder height [mm]:", float(h_cyl))
print("cylinder half-height [mm]:", float(hz_cyl))


# ============================================================
# Waypoints
# ============================================================
def extract_waypoints(tr, dist=20.0):
    wps = [tr[0].tolist()]
    acc = 0.0
    for i in range(1, len(tr)):
        acc += float(np.linalg.norm(tr[i] - tr[i - 1]))
        if acc >= dist:
            wps.append(tr[i].tolist())
            acc = 0.0
    wps.append(tr[-1].tolist())
    return np.array(wps, dtype=float)

waypoints = extract_waypoints(traj, dist=20.0)


# ============================================================
# Plot: finite-height cylinder + trajectory + waypoints + start/goal
# ============================================================
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Cylinder surface for plotting
theta_grid = np.linspace(0, 2 * np.pi, 60)
z_grid = np.linspace(cz - hz_cyl, cz + hz_cyl, 2)
TH, ZZ = np.meshgrid(theta_grid, z_grid)

Xc = r * np.cos(TH) + cx
Yc = r * np.sin(TH) + cy
Zc = ZZ

ax.plot_surface(Xc, Yc, Zc, alpha=0.5, linewidth=0)

# Optional: draw sensing ring at z = cz
Xs = R * np.cos(theta_grid) + cx
Ys = R * np.sin(theta_grid) + cy
Zs = np.full_like(theta_grid, cz)
ax.plot(Xs, Ys, Zs, linewidth=2)

# Trajectory
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=3, label="Trajectory")

# Waypoints
ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=40, label="Waypoints")

# Start/Goal
ax.scatter(*x0, s=70, label="Start")
ax.scatter(*xg, s=70, label="Goal")

# Axis limits
xmin = min(np.min(traj[:, 0]), x0[0], xg[0], cx - R)
xmax = max(np.max(traj[:, 0]), x0[0], xg[0], cx + R)
ymin = min(np.min(traj[:, 1]), x0[1], xg[1], cy - R)
ymax = max(np.max(traj[:, 1]), x0[1], xg[1], cy + R)
zmin = min(np.min(traj[:, 2]), x0[2], xg[2], cz - Hz)
zmax = max(np.max(traj[:, 2]), x0[2], xg[2], cz + Hz)

mx = 0.05 * (xmax - xmin + 1e-9)
my = 0.05 * (ymax - ymin + 1e-9)
mz = 0.05 * (zmax - zmin + 1e-9)

ax.set_xlim(xmin - mx, xmax + mx)
ax.set_ylim(ymin - my, ymax + my)
ax.set_zlim(zmin - mz, zmax + mz)

for i in range(len(waypoints)):
    print("X:", waypoints[i])

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")
ax.grid(True)
ax.legend()
ax.view_init(elev=45, azim=45)
plt.tight_layout()
plt.show()