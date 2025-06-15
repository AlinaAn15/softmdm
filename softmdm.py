import numpy as np

def InitializeU(m):
    return np.ones(m) / m

def ComputeB(pts):
    return 0.5 * (np.linalg.norm(pts, axis=1)**2)

def ComputeX(pts, u):
    return (u @ pts) / u.sum()

def ComputeD(pts, x, b):
    return pts @ x - b

def ActiveIndices(u, c):
    return [i for i, ui in enumerate(u) if 0 < ui < c]

def FindIprime(pts, x, b, u, c):
    idx = ActiveIndices(u, c)
    d = ComputeD(pts, x, b)
    return idx[np.argmin(d[idx])]

def FindIdoublePrime(pts, x, b, u, c):
    idx = ActiveIndices(u, c)
    d = ComputeD(pts, x, b)
    return idx[np.argmax(d[idx])]

def ComputeDelta(pts, x, b, u, c):
    ip = FindIprime(pts, x, b, u, c)
    idp = FindIdoublePrime(pts, x, b, u, c)
    d = ComputeD(pts, x, b)
    return d[idp] - d[ip]

def DistanceTilde(pts, ip, idp, Delta):
    diff = pts[ip] - pts[idp]
    return Delta / np.dot(diff, diff)

def StepSize(pts, u, ip, idp, c, Delta):
    dt = DistanceTilde(pts, ip, idp, Delta)
    return min(dt, u[idp], c - u[ip])

def UpdateUAndX(pts, u, x, ip, idp, step):
    u_new = u.copy()
    u_new[ip] += step
    u_new[idp] -= step
    x_new = ComputeX(pts, u_new)
    return u_new, x_new

def Q(pts, u, b):
    return 0.5 * (u @ (pts @ pts.T) @ u) - b @ u

def SoftMDM(pts, c, tol = 0):
    m = len(pts)
    b = ComputeB(pts)
    u = InitializeU(m)
    x = ComputeX(pts, u)

    history = []

    while True:
        ip = FindIprime(pts, x, b, u, c)
        idp = FindIdoublePrime(pts, x, b, u, c)
        Delta = ComputeDelta(pts, x, b, u, c)
        q = Q(pts, u, b)
        rad = np.sqrt(-2 * q)
        history.append({"u": u.copy(), "x": x.copy(), "Delta": Delta, "radius": rad})

        if Delta <= tol:
            break

        step = StepSize(pts, u, ip, idp, c, Delta)
        u, x = UpdateUAndX(pts, u, x, ip, idp, step)

    mu = Q(pts, u, b)
    radius = np.sqrt(-2 * mu)
    support_indices = [i for i, ui in enumerate(u) if 0 < ui < c]

    return {
        "center": x,
        "radius": radius,
        "weights": u,
        "mu": mu,
        "support_indices": support_indices,
        "history": history
    }
