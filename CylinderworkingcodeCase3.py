import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def save_waypoints_to_file(waypoints, filename="twocylinderobstaclewaypoints.txt"):
    with open(filename, "w") as f:
        for i, wp in enumerate(waypoints):
            f.write(f"{i}: {wp}\n")
    print(f"Waypoints saved to {filename}")


# ============================================================
# User settings / parameters to tune
# ============================================================
x0 = np.array([180.0, -590.0, 180.0])   # mm
xg = np.array([460.0,  590.0, 630.0])   # mm

N = 25000
dt = 0.01

# Density tuning
alpha = 0.40
theta = 1e-4
kappa = 6e6
u_max = 250.0
MAX_STEP = 2.0

# Goal neighborhood for Remark-2 modification
goal_delta = 100.0
stop_radius = 5.0

# Small tangential steering to break the deadlock in front of cylinders
swirl_gain = 0.18


# ============================================================
# Obstacle geometry: two finite-height cylinders along z
# Place them front/back in the XY path direction, but at SAME z
# ============================================================
dxy = xg[:2] - x0[:2]
dxy_norm = np.linalg.norm(dxy)
edir_xy = dxy / max(dxy_norm, 1e-9)

# Choose a common z level so both obstacles are in one horizontal line
cz_common = 0.5 * (x0[2] + xg[2])

# Place cylinder centers along the XY start-goal line only
c1_xy = x0[:2] + 0.35 * dxy
c2_xy = x0[:2] + 0.65 * dxy

cx1, cy1, cz1 = c1_xy[0], c1_xy[1], cz_common
cx2, cy2, cz2 = c2_xy[0], c2_xy[1], cz_common

# Radius based on workspace size
ws_min = np.minimum(x0, xg)
ws_max = np.maximum(x0, xg)
ws_size = np.linalg.norm(ws_max - ws_min)
r = 0.055 * ws_size

# Reduced sensing radius in xy-plane
R = 1.55 * r

# Cylinder height: 22 inches converted to mm
inch_to_mm = 25.4
h_cyl = 22.0 * inch_to_mm        # 558.8 mm
hz_cyl = h_cyl / 2.0             # 279.4 mm

# Reduced sensing height scaling in z
inflate_z = 1.05
Hz = inflate_z * hz_cyl

# Transition denominator
HS_DEN = max(R**2 - r**2, Hz - hz_cyl)


# ============================================================
# Smooth step functions
# ============================================================
def f(t):
    if t <= 0.0:
        return 0.0
    return np.exp(-1.0 / t)

def fprime(t):
    if t <= 0.0:
        return 0.0
    et = np.exp(-1.0 / t)
    return et * (1.0 / (t * t))

def fbar(t):
    a = f(t)
    b = f(1.0 - t)
    denom = a + b
    if denom == 0.0:
        return 0.0
    return a / denom

def fbarprime(t):
    a = f(t)
    b = f(1.0 - t)
    ap = fprime(t)
    bp = -fprime(1.0 - t)

    denom = a + b
    if denom == 0.0:
        return 0.0
    return (ap * denom - a * (ap + bp)) / (denom * denom)


# ============================================================
# Cylinder helper functions
# ============================================================
def d2_xy(x, cx, cy):
    return (x[0] - cx)**2 + (x[1] - cy)**2

def h_single_cyl(x, cx, cy, cz):
    radial = d2_xy(x, cx, cy) - r**2
    vertical = abs(x[2] - cz) - hz_cyl
    return max(radial, vertical)

def s_single_cyl(x, cx, cy, cz):
    radial = d2_xy(x, cx, cy) - R**2
    vertical = abs(x[2] - cz) - Hz
    return max(radial, vertical)

def grad_h_single_cyl(x, cx, cy, cz):
    radial = d2_xy(x, cx, cy) - r**2
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

def active_obstacle_center(x):
    h1 = h_single_cyl(x, cx1, cy1, cz1)
    h2 = h_single_cyl(x, cx2, cy2, cz2)
    if h1 <= h2:
        return np.array([cx1, cy1, cz1]), 1
    return np.array([cx2, cy2, cz2]), 2


# ============================================================
# Union of two obstacles
# ============================================================
def h_fun(x):
    h1 = h_single_cyl(x, cx1, cy1, cz1)
    h2 = h_single_cyl(x, cx2, cy2, cz2)
    return min(h1, h2)

def s_fun(x):
    s1 = s_single_cyl(x, cx1, cy1, cz1)
    s2 = s_single_cyl(x, cx2, cy2, cz2)
    return min(s1, s2)

def grad_h(x):
    h1 = h_single_cyl(x, cx1, cy1, cz1)
    h2 = h_single_cyl(x, cx2, cy2, cz2)

    if h1 <= h2:
        return grad_h_single_cyl(x, cx1, cy1, cz1)
    else:
        return grad_h_single_cyl(x, cx2, cy2, cz2)


# ============================================================
# Phi, Psi and gradients
# ============================================================
def Phi_and_gradPhi(x):
    hx = h_fun(x)
    sx = s_fun(x)

    if hx <= 0.0:
        return 0.0, np.zeros(3)

    if sx > 0.0:
        return 1.0, np.zeros(3)

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
# Density rho(x)=Psi(x)/V(x)^alpha
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
# Tangential steering around active obstacle
# ============================================================
def tangential_bias(x):
    Phi, _ = Phi_and_gradPhi(x)
    sx = s_fun(x)

    # Only act inside sensing region and outside obstacle
    if sx > 0.0 or Phi <= 0.0 or Phi >= 1.0:
        return np.zeros(3)

    c_active, _ = active_obstacle_center(x)
    rel = np.array([x[0] - c_active[0], x[1] - c_active[1], 0.0])

    nr = np.linalg.norm(rel[:2])
    if nr < 1e-9:
        return np.zeros(3)

    # Tangent in XY plane
    tangent = np.array([-rel[1], rel[0], 0.0]) / nr

    # Choose the tangent direction that aligns better with motion toward goal in XY
    gxy = np.array([xg[0] - x[0], xg[1] - x[1], 0.0])
    if np.dot(tangent, gxy) < 0.0:
        tangent = -tangent

    # strongest near obstacle, fades away outside sensing shell
    strength = swirl_gain * (1.0 - Phi)
    return strength * tangent


# ============================================================
# Near-goal modification + tangential steering
# ============================================================
def controller_u(x):
    _, grad_rho = rho_and_gradrho(x)

    e = x - xg
    dist = np.linalg.norm(e)

    tau = (goal_delta - dist) / goal_delta
    chi = fbar(tau)

    u_core = (1.0 - chi) * grad_rho - chi * e
    u_swirl = tangential_bias(x)

    u = u_core + u_swirl
    u = kappa * u

    infn = np.max(np.abs(u))
    if infn > u_max:
        u = (u / infn) * u_max

    return u


# ============================================================
# Simulate
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
# Diagnostics
# ============================================================
inside = np.array([
    (
        ((d2_xy(x, cx1, cy1) <= r**2) and (abs(x[2] - cz1) <= hz_cyl))
        or
        ((d2_xy(x, cx2, cy2) <= r**2) and (abs(x[2] - cz2) <= hz_cyl))
    )
    for x in traj
])

print("traj shape:", traj.shape)
print("entered obstacle? :", bool(np.any(inside)))
print("workspace assumed units: mm")
print("cylinder radius [mm]:", float(r))
print("sensing radius [mm]:", float(R))
print("cylinder height [mm]:", float(h_cyl))
print("cylinder 1 center [mm]:", [cx1, cy1, cz1])
print("cylinder 2 center [mm]:", [cx2, cy2, cz2])


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
# Plot
# ============================================================
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

theta_grid = np.linspace(0, 2 * np.pi, 60)
z_grid = np.linspace(cz_common - hz_cyl, cz_common + hz_cyl, 2)
TH, ZZ = np.meshgrid(theta_grid, z_grid)

# Cylinder 1
Xc1 = r * np.cos(TH) + cx1
Yc1 = r * np.sin(TH) + cy1
Zc1 = ZZ
ax.plot_surface(Xc1, Yc1, Zc1, alpha=0.5, linewidth=0)

# Cylinder 2
Xc2 = r * np.cos(TH) + cx2
Yc2 = r * np.sin(TH) + cy2
Zc2 = ZZ
ax.plot_surface(Xc2, Yc2, Zc2, alpha=0.5, linewidth=0)

# Optional sensing rings
Xs1 = R * np.cos(theta_grid) + cx1
Ys1 = R * np.sin(theta_grid) + cy1
Zs1 = np.full_like(theta_grid, cz1)
ax.plot(Xs1, Ys1, Zs1, linewidth=2)

Xs2 = R * np.cos(theta_grid) + cx2
Ys2 = R * np.sin(theta_grid) + cy2
Zs2 = np.full_like(theta_grid, cz2)
ax.plot(Xs2, Ys2, Zs2, linewidth=2)

# Trajectory
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=3, label="Trajectory")

# Waypoints
ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=40, label="Waypoints")

# Start/Goal
ax.scatter(*x0, s=70, label="Start")
ax.scatter(*xg, s=70, label="Goal")

xmin = min(np.min(traj[:, 0]), x0[0], xg[0], cx1 - R, cx2 - R)
xmax = max(np.max(traj[:, 0]), x0[0], xg[0], cx1 + R, cx2 + R)
ymin = min(np.min(traj[:, 1]), x0[1], xg[1], cy1 - R, cy2 - R)
ymax = max(np.max(traj[:, 1]), x0[1], xg[1], cy1 + R, cy2 + R)
zmin = min(np.min(traj[:, 2]), x0[2], xg[2], cz_common - Hz)
zmax = max(np.max(traj[:, 2]), x0[2], xg[2], cz_common + Hz)

mx = 0.05 * (xmax - xmin + 1e-9)
my = 0.05 * (ymax - ymin + 1e-9)
mz = 0.05 * (zmax - zmin + 1e-9)

ax.set_xlim(xmin - mx, xmax + mx)
ax.set_ylim(ymin - my, ymax + my)
ax.set_zlim(zmin - mz, zmax + mz)

ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel("z [mm]")
ax.grid(True)
ax.legend()
ax.view_init(elev=45, azim=45)
plt.tight_layout()
plt.show()