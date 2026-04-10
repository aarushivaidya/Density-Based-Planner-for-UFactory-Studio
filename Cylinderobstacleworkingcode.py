"""
Density-based safe navigation (paper-faithful) for a single cylindrical obstacle.

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
x0 = np.array([180.0, -590.0, 180.0])
xg = np.array([460.0,  590.0, 630.0])

N = 20000
dt = 0.01

# Density tuning (paper Remark 3)

alpha = 0.2
theta = 1e-4          # try 0.2, 0.5, 1.0
kappa = 1e8          # global speed gain; try 1e7..1e10
u_max = 250.0
MAX_STEP = 2.0


# Goal neighborhood for Remark-2 modification
goal_delta = 25.0      # radius of neighborhood; try: 10..40
stop_radius = 5.0      # stop when within this distance of goal

# Obstacle geometry: cylinder along z, defined by circle in (x,y)
# We'll place obstacle at midpoint (like your original)
c = 0.5 * (x0 + xg)
cx, cy, cz = c[0], c[1], c[2]

# radius based on workspace size (like you had)
ws_min = np.minimum(x0, xg)
ws_max = np.maximum(x0, xg)
ws_size = np.linalg.norm(ws_max - ws_min)
r = 0.06 * ws_size

# Sensing radius R (this is sk(x)=0 boundary). THIS IS THE MAIN TUNING KNOB.
# If you see "goes through obstacle", increase R and/or decrease MAX_STEP.
R = 3.0 * r            # try: 2.5*r, 3*r, 4*r

# Cylinder height for plotting
h_cyl = ws_size


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
    """Smooth step: \bar f(t) = f(t) / ( f(t) + f(1-t) )."""
    a = f(t)
    b = f(1.0 - t)
    denom = a + b
    if denom == 0.0:
        return 0.0
    return a / denom

def fbarprime(t):
    """Derivative of \bar f(t) using quotient rule."""
    a = f(t)
    b = f(1.0 - t)
    ap = fprime(t)
    # b(t)=f(1-t) -> b'(t) = -f'(1-t)
    bp = -fprime(1.0 - t)

    denom = (a + b)
    if denom == 0.0:
        return 0.0
    # (a'*(a+b) - a*(a'+b')) / (a+b)^2
    return (ap * denom - a * (ap + bp)) / (denom * denom)


# ============================================================
# Obstacle sets h(x) and s(x) (paper (2)-(3))
# For a cylinder: use xy-circle level sets.
#   h(x) = d^2 - r^2  -> Xu = {h<=0}
#   s(x) = d^2 - R^2  -> sensing boundary {s=0}
# In transition region: Xu excluded, but inside sensing boundary: 0<h<=? see paper.
# ============================================================
def d2_xy(x):
    return (x[0] - cx)**2 + (x[1] - cy)**2

def h_fun(x):
    return d2_xy(x) - r**2

def s_fun(x):
    return d2_xy(x) - R**2

def grad_h(x):
    # gradient wrt (x,y,z)
    return np.array([2.0*(x[0]-cx), 2.0*(x[1]-cy), 0.0])

# Note: for circular sensing region, h-s = (d^2-r^2)-(d^2-R^2)=R^2-r^2 is constant.
HS_DEN = (R**2 - r**2)


# ============================================================
# Phi, Psi per (7)-(8) and their gradients
# Phi(x)=0 in obstacle, =1 outside sensing region, else =phi(x) in sensing ring.
# phi(x)= fbar( h/(h-s) ) = fbar( h/(R^2-r^2) ) for this circular case.
# ============================================================
def Phi_and_gradPhi(x):
    hx = h_fun(x)
    sx = s_fun(x)

    # inside obstacle
    if hx <= 0.0:
        return 0.0, np.zeros(3)

    # outside sensing region => Phi=1
    # sensing region is {s<=0} excluding obstacle; so outside means s>0
    if sx > 0.0:
        return 1.0, np.zeros(3)

    # transition ring: 0 < h and s <= 0
    tau = hx / HS_DEN  # since h-s=R^2-r^2
    phi = fbar(tau)
    dphi_dtau = fbarprime(tau)

    # grad tau = grad h / (R^2-r^2)
    gtau = grad_h(x) / HS_DEN
    gphi = dphi_dtau * gtau

    return phi, gphi


def Psi_and_gradPsi(x):
    Phi, gPhi = Phi_and_gradPhi(x)
    Psi = Phi + theta
    gPsi = gPhi
    return Psi, gPsi


# ============================================================
# Density rho(x)=Psi(x)/V(x)^alpha, V=||x-xg||^2 (shifted target)
# And grad rho per (11) expanded:
# grad rho = (grad Psi)/V^alpha - alpha*Psi*(grad V)/V^(alpha+1)
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

    # grad rho
    grad_rho = (gPsi / Va) - (alpha * Psi * gV) / (V**(alpha + 1.0))
    return rho, grad_rho


# ============================================================
# Near-goal modification (Remark 2): blend between ∇rho and -e
# xdot = [1 - fbar(tau)] ∇rho  - fbar(tau) * e
# We choose tau = (goal_delta - ||e||)/goal_delta so that near goal, tau>0 and fbar ~ 1
# ============================================================
def controller_u(x):
    _, grad_rho = rho_and_gradrho(x)

    e = x - xg
    dist = np.linalg.norm(e)

    tau = (goal_delta - dist) / goal_delta
    chi = fbar(tau)

    u = (1.0 - chi) * grad_rho - chi * e

    # ---- speed gain (inside function!) ----
    u = kappa * u

    # saturation (paper eqn 16)
    infn = np.max(np.abs(u))
    if infn > u_max:
        u = (u / infn) * u_max

    return u



# ============================================================
# Simulate (Euler) with step cap to avoid crossing sensing ring in one step
# ============================================================
traj = np.zeros((N, 3), dtype=float)
traj[0] = x0

for k in range(N-1):
    xk = traj[k]
    u = controller_u(xk)

    step = dt * u
    nstep = np.linalg.norm(step)
    if nstep > MAX_STEP:
        step = step * (MAX_STEP / nstep)

    traj[k+1] = xk + step

    if np.linalg.norm(traj[k+1] - xg) < stop_radius:
        traj = traj[:k+2]
        break


# ============================================================
# Diagnostics: did we enter obstacle?
# For cylinder: inside if d_xy < r
# ============================================================
dxy = np.sqrt((traj[:, 0]-cx)**2 + (traj[:, 1]-cy)**2)
print("traj shape:", traj.shape)
print("min(dxy - r):", float(np.min(dxy - r)), "  (should be >= 0 for strict avoidance)")
print("min dxy:", float(np.min(dxy)), " r:", float(r), " R:", float(R))

# If min(dxy-r) is negative:
# - Increase R (wider sensing region)
# - Decrease MAX_STEP
# - Decrease u_max (stronger saturation)
# - Decrease theta (sharper transition)


# ============================================================
# Waypoints (same idea as your original)
# ============================================================
def extract_waypoints(tr, dist=20.0):
    wps = [tr[0].tolist()]
    acc = 0.0
    for i in range(1, len(tr)):
        acc += float(np.linalg.norm(tr[i] - tr[i-1]))
        if acc >= dist:
            wps.append(tr[i].tolist())
            acc = 0.0
    wps.append(tr[-1].tolist())
    return np.array(wps, dtype=float)

waypoints = extract_waypoints(traj, dist=20.0)


# ============================================================
# Plot: cylinder + trajectory + waypoints + start/goal
# ============================================================
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Cylinder surface for plotting
theta_grid = np.linspace(0, 2*np.pi, 60)
z_grid = np.linspace(0, 1, 2)
TH, ZZ = np.meshgrid(theta_grid, z_grid)
Xc = r * np.cos(TH) + cx
Yc = r * np.sin(TH) + cy
Zc = ZZ * h_cyl + (cz - h_cyl/2)

ax.plot_surface(Xc, Yc, Zc, alpha=0.5, linewidth=0)

# Plot sensing boundary too (optional, helps debug)
Rs = R
Xs = Rs * np.cos(theta_grid) + cx
Ys = Rs * np.sin(theta_grid) + cy
Zs = np.full_like(theta_grid, cz)
ax.plot(Xs, Ys, Zs, linewidth=2)  # ring at z=cz (just a visual cue)

# Trajectory
ax.plot(traj[:,0], traj[:,1], traj[:,2], linewidth=3, label="Trajectory")

# Waypoints

ax.scatter(waypoints[:,0], waypoints[:,1], waypoints[:,2], s=40, label="Waypoints")


# Start/Goal
ax.scatter(*x0, s=70, label="Start")
ax.scatter(*xg, s=70, label="Goal")

# Axis limits include obstacle + traj + start/goal
xmin = min(np.min(traj[:,0]), x0[0], xg[0], cx - R)
xmax = max(np.max(traj[:,0]), x0[0], xg[0], cx + R)
ymin = min(np.min(traj[:,1]), x0[1], xg[1], cy - R)
ymax = max(np.max(traj[:,1]), x0[1], xg[1], cy + R)
zmin = min(np.min(traj[:,2]), x0[2], xg[2], cz - h_cyl/2)
zmax = max(np.max(traj[:,2]), x0[2], xg[2], cz + h_cyl/2)

mx = 0.05*(xmax-xmin + 1e-9)
my = 0.05*(ymax-ymin + 1e-9)
mz = 0.05*(zmax-zmin + 1e-9)

ax.set_xlim(xmin-mx, xmax+mx)
ax.set_ylim(ymin-my, ymax+my)
ax.set_zlim(zmin-mz, zmax+mz)

for i in range(len(waypoints)):
    print("X:", waypoints[i])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.grid(True)
ax.legend()
ax.view_init(elev=45, azim=45)
plt.tight_layout()
plt.show()

