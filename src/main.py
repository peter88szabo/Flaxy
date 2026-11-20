"""Flaxy numerical specification. Run: python3 flaxy_reference.py (NumPy only).
Atomic units. Deterministic synthetic parameters and fixtures, not a chemical PES.
"""
import itertools
import numpy as np

class Jet:
    """Scalar value, Cartesian gradient, and Cartesian Hessian."""
    __array_priority__ = 1000

    def __init__(self, v, g, h):
        self.v, self.g, self.h = float(v), g, h

    def __add__(self, b):
        if isinstance(b, Jet):
            return Jet(self.v+b.v, self.g+b.g, self.h+b.h)
        return Jet(self.v+b, self.g, self.h)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.g, -self.h)

    def __sub__(self, b):
        return self + (-b)

    def __rsub__(self, b):
        return (-self) + b

    def __mul__(self, b):
        if not isinstance(b, Jet):
            return Jet(self.v*b, self.g*b, self.h*b)
        h = self.h*b.v + self.v*b.h
        cross = np.outer(self.g, b.g)
        return Jet(self.v*b.v, self.g*b.v+self.v*b.g, h+cross+cross.T)

    __rmul__ = __mul__

    def unary(self, f, fp, fpp):
        return Jet(f, fp*self.g, fp*self.h+fpp*np.outer(self.g, self.g))

    def __pow__(self, p):
        v = self.v
        return self.unary(v**p, p*v**(p-1), p*(p-1)*v**(p-2))

    def __truediv__(self, b):
        return self * (b**-1)

def val(x):
    return x.v if isinstance(x, Jet) else x

def exp(x):
    e = np.exp(val(x))
    return x.unary(e, e, e) if isinstance(x, Jet) else e

def rising(x, a, b):
    if val(x) <= a:
        return 0.
    if val(x) >= b:
        return 1.
    u = (x-a)/(b-a)
    return u**3 * (10 + u*(-15 + 6*u))

def sub(a, b):
    return [x-y for x, y in zip(a, b)]

def mean(a, b):
    return [(x+y)*0.5 for x, y in zip(a, b)]

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def length(x):
    return dot(x, x)**0.5

def unit(x):
    r = length(x)
    return [a/r for a in x]

def seed(x):
    v = np.asarray(x).ravel()
    eye, zeros = np.eye(v.size), np.zeros((v.size, v.size))
    return np.array([Jet(a, g, zeros) for a, g in zip(v, eye)],
                    dtype=object).reshape(-1, 3).tolist()

def dq_shape(a, b, n):
    u, v, d = dot(a, n), dot(b, n), dot(a, b)
    return (15*u*v*v - 3*u - 6*d*v)*0.5

def entrance_basis(a, b, n, r):
    u, v = dot(a, n), dot(b, n)
    return [dq_shape(a, b, n)*r**-4, -r**-6,
            -(3*u*u-1)*0.5*r**-6, -(3*v*v-1)*0.5*r**-6]

COEF = np.array([0.12, 7., 0.8, -0.3])
HIDS = (1, 2, 3)
PERMS = list(itertools.permutations(HIDS))
PAIRS = list(itertools.combinations(range(4), 2))
RANKS = [1, 2, 2, 2, 2, 2, 1]
rng = np.random.default_rng(441)
CORES = [rng.normal(0, 0.8, (a, 2, b))
         for a, b in zip(RANKS[:-1], RANKS[1:])]

def morse(r, req, depth):
    return depth*((1-exp(-1.2*(r-req)))**2-1)

def partition(x, tail=1., coef=COEF):
    """Return fixed contribution and weight multiplying the inner TT."""
    o = x[0]
    gs, branches = [], []
    for i in HIDS:
        j, k = [t for t in HIDS if t != i]
        oh, hh = sub(x[i], o), sub(x[k], x[j])
        # OH+H2. Both reference origins are geometric midpoints.
        disp = sub(mean(x[j], x[k]), mean(o, x[i]))
        r = length(disp) if val(dot(disp, disp)) > 0 else 0.
        gate = (rising(r, 4., 8.) * (1-rising(length(oh), 2.4, 3.4))
                * (1-rising(length(hh), 2.2, 3.2)))
        if val(gate) > 0:
            f = entrance_basis(unit(oh), unit(hh), unit(disp), r)
            energy = (morse(length(oh), 1.8, 0.15)
                      + morse(length(hh), 1.4, 0.17) + tail*dot(coef, f))
            gs.append(gate)
            branches.append(energy)
        # H2O+H. O is the water origin for this illustrative channel.
        oj, ok = sub(x[j], o), sub(x[k], o)
        r = length(oh)
        gate = (rising(r, 4., 8.) * (1-rising(length(oj), 2.4, 3.4))
                * (1-rising(length(ok), 2.4, 3.4)))
        if val(gate) > 0:
            water = (morse(length(oj), 1.8, 0.18)
                     + morse(length(ok), 1.8, 0.18)
                     + 0.02*(dot(unit(oj), unit(ok))+0.25)**2)
            delta = -4*r**-6
            gs.append(gate)
            branches.append(water + tail*delta)
    core = 1.
    for g in gs:
        core = core*(1-g)
    z = core + sum(gs)
    return sum(g*b for g, b in zip(gs, branches))/z, core/z

def tt(x, cores):
    v = [1.]
    for (i, j), g in zip(PAIRS, cores):
        y = exp(-length(sub(x[i], x[j]))/2)
        phi = [exp(-((y-c)**2)/(2*0.18**2)) for c in (0.18, 0.55)]
        v = [sum(v[a]*g[a,k,b]*phi[k] for a in range(g.shape[0])
                 for k in range(2)) for b in range(g.shape[2])]
    return v[0]

def symtt(x, cores):
    return sum(tt([x[0]]+[x[i] for i in p], cores) for p in PERMS)/6

def surface(x, cores=CORES, tail=1.):
    base, w = partition(x, tail)
    if val(w) == 0:
        return base
    return base+w*symtt(x, cores)

def obs(y):
    return np.r_[y.v, -y.g]  # energy and forces

def check_local_solve():
    random = np.random.default_rng(9)
    active = 2
    shape = CORES[active].shape
    design, target = [], []
    for _ in range(15):
        x = seed(random.normal(0, 1.45, (4, 3)))
        base, w = partition(x)
        if val(w) < 1e-8:
            continue
        target.append(obs(surface(x)-base))
        cols = []
        for a in range(np.prod(shape)):
            g = np.zeros(shape)
            g.flat[a] = 1.
            trial = CORES.copy()
            trial[active] = g
            cols.append(obs(w*symtt(x, trial)))
        design.append(np.stack(cols, axis=1))
    A, b = np.vstack(design), np.concatenate(target)
    sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=1e-12)
    assert rank == np.prod(shape)
    return np.max(np.abs(sol-CORES[active].ravel()))

def check_tail_fit():
    random = np.random.default_rng(316)
    rows = []
    for _ in range(70):
        a, b, n = [unit(random.normal(size=3)) for _ in range(3)]
        rows.append(entrance_basis(a, b, n, random.uniform(8, 18)))
    A = np.array(rows)
    scales = np.linalg.norm(A[:60], axis=0)
    # Noiseless analytic fixture, with ten unseen orientation/radius pairs.
    y = A @ COEF
    c = np.linalg.lstsq(A[:60]/scales, y[:60], rcond=None)[0]/scales
    return np.max(np.abs(c-COEF)), np.max(np.abs(A[60:]@c-y[60:]))

def check_capture():
    # Synthetic atom + homonuclear rotor, frozen axis along or across R.
    # Each cut has C6(u)=12*(1+0.4*P2(u)); this isolates radial matching.
    max_barrier_err = 0.
    for u in [0., 1.]:
        c6 = 12*(1+0.4*(3*u*u-1)/2)
        for rb in [7., 13., 22.]:
            mu = 900.
            ell2 = 6*mu*c6/rb**4
            analytic = 2*c6/rb**6
            for rin, rout in [(5., 9.), (6., 10.)]:
                def effective(r):
                    w = 1-rising(r, rin, rout)
                    # A C2 compact residual confined to R < 5.
                    inner = -c6/r**6 + 0.002*(1-rising(r, 3., 5.))
                    return ((1-w)*(-c6/r**6)+w*inner
                            +ell2/(2*mu*r*r))
                # Search across the nominal join and beyond, on this outer cut.
                lo, hi = 5.1, 2*rb
                for _ in range(80):
                    a, b = (2*lo+hi)/3, (lo+2*hi)/3
                    if effective(a) < effective(b):
                        lo = a
                    else:
                        hi = b
                err = abs(effective((lo+hi)/2)-analytic)/analytic
                max_barrier_err = max(max_barrier_err, err)
    return max_barrier_err

def check_geometry():
    # Rigid motion, and an exit with a stretched, intact water molecule.
    x = np.array([[0.,0.,0.], [1.75,0.,0.], [5.7,0.6,-0.2],
                  [6.2,1.8,0.25]])
    q, _ = np.linalg.qr(np.random.default_rng(92).normal(size=(3,3)))
    y, yr = surface(seed(x)), surface(seed(x@q+2.9))
    errors = [abs(y.v-yr.v),
              np.max(abs(y.g.reshape(-1,3)@q-yr.g.reshape(-1,3)))]
    exit_x = np.array([[0.,0.,0.], [1.95,0.,0.], [-0.4,1.6,0.],
                       [0.,0.,18.]])
    base, w = partition(seed(exit_x))
    assert val(w) == 0 and abs(surface(seed(exit_x)).v-base.v) < 1e-14
    return max(errors)

def feature_cache(x):
    """For each orbit term cache feature values and Cartesian derivatives."""
    pairs = {}
    for i, j in PAIRS:
        vec = x[i]-x[j]
        r = np.linalg.norm(vec)
        y = np.exp(-r/2)
        dy = np.zeros_like(x)
        dy[i] = -0.5*y*vec/r
        dy[j] = -dy[i]
        offsets = y-np.array([0.18, 0.55])
        f = np.exp(-0.5*(offsets/0.18)**2)
        df = dy.reshape(-1,1)*(-offsets*f/0.18**2)
        pairs[i,j] = f, df
    cache = []
    for p in PERMS:
        ids = [0, *p]
        cache.append([pairs[tuple(sorted((ids[i], ids[j])))]
                      for i,j in PAIRS])
    return cache

def attach(v, dv, m, dm):
    """Multiply a row environment by its next core and propagate its Jacobian."""
    return v@m, dv@m+np.einsum('i,xij->xj', v, dm)

def local_design(cached, cores, j, count):
    """Return all energy and force columns for a one- or two-site update."""
    w, dw, orbit = cached
    total, dtotal = 0., 0.
    for fs in orbit:
        l, r = np.ones(1), np.ones(1)
        dl, dr = np.zeros((12,1)), np.zeros((12,1))
        for s in [*range(j), *range(5,j+count-1,-1)]:
            f, df = fs[s]
            m = np.einsum('k,akb->ab', f, cores[s])
            dm = np.einsum('xk,akb->xab', df, cores[s])
            if s < j:
                l, dl = attach(l,dl,m,dm)
            else:
                r, dr = attach(r,dr,m.T,dm.transpose(0,2,1))
        # Cartesian product l (x) phi_j [(x) phi_j+1] (x) r.
        for f, df in [*fs[j:j+count], (r, dr)]:
            dl = (dl[:,:,None]*f + l[None,:,None]*df[:,None,:]).reshape(12,-1)
            l = np.outer(l,f).ravel()
        total, dtotal = total+l, dtotal+dl
    energy = total/len(orbit)
    return np.vstack((w*energy, -(w*dtotal/len(orbit)+np.outer(dw,energy))))

def training_record(x):
    xx = seed(x)
    base, w = partition(xx)
    wgrad = w.g if isinstance(w, Jet) else np.zeros(12)
    return (val(w), wgrad, feature_cache(x)), obs(surface(xx)-base)

def validation_loss(records, cores):
    g = cores[0].ravel()
    errors = [local_design(data,cores,0,1)@g-target
              for data,target in records]
    return np.mean(np.square(errors))

def canonicalize_from_right(cores):
    for j in range(len(cores)-1,0,-1):
        r, b, s = cores[j].shape
        q, t = np.linalg.qr(cores[j].reshape(r,b*s).T)
        cores[j] = q.T.reshape(-1,b,s)
        cores[j-1] = np.tensordot(cores[j-1],t.T,axes=1)

def two_site_fit(train, valid, initial, sweeps=2, rank_cap=2,
                 svd_tol=1e-9):
    """Dense two-site solver, equal row weights, with a proximal update."""
    cores = [g.copy() for g in initial]
    canonicalize_from_right(cores)
    best = [g.copy() for g in cores]
    best_loss = validation_loss(valid, cores)
    for _ in range(sweeps):
        for forward in (True, False):
            sites = range(5) if forward else range(4,-1,-1)
            for j in sites:
                A = np.vstack([local_design(d,cores,j,2) for d,_ in train])
                y = np.concatenate([t for _,t in train])
                prior = np.tensordot(cores[j], cores[j+1], axes=1).ravel()
                gamma = 0.01*np.linalg.norm(A, ord=2)
                A = np.vstack((A,gamma*np.eye(prior.size)))
                y = np.r_[y,gamma*prior]
                k = np.linalg.lstsq(A,y,rcond=1e-12)[0]
                a,b,_ = cores[j].shape
                _,c,d = cores[j+1].shape
                u,s,vt = np.linalg.svd(k.reshape(a*b,c*d),full_matrices=False)
                cut = svd_tol*np.linalg.norm(s)
                rank = next(t for t in range(1,len(s)+1)
                            if np.linalg.norm(s[t:]) <= cut)
                rank = min(rank, rank_cap)
                u, s, vt = u[:,:rank], s[:rank], vt[:rank]
                if forward:
                    vt = s[:,None]*vt
                else:
                    u = u*s
                cores[j] = u.reshape(a,b,rank)
                cores[j+1] = vt.reshape(rank,c,d)
        score = validation_loss(valid, cores)
        if score < best_loss:
            best_loss, best = score, [g.copy() for g in cores]
    return best, best_loss

def check_sweep():
    rand = np.random.default_rng(126)
    data = [training_record(rand.normal(0,1.4,(4,3))) for _ in range(24)]
    # Perturb the teacher for a reproducible local-convergence test.
    initial = [g+rand.normal(0,0.12,g.shape) for g in CORES]
    before = validation_loss(data[18:],initial)
    _, after = two_site_fit(data[:18],data[18:],initial)
    assert after < 0.25*before
    return before, after



def main():
    x = np.array([[0.,0.,0.], [1.75,0.,0.], [5.7,0.6,-0.2],
                  [6.2,1.8,0.25]])
    y = surface(seed(x))
    v = np.linspace(-1, 1, 12)
    v /= np.linalg.norm(v)
    dx = v.reshape(4, 3)*1e-4
    yp, ym = surface(seed(x+dx)), surface(seed(x-dx))
    eg = abs((yp.v-ym.v)/2e-4-y.g@v)
    eh = np.max(np.abs((yp.g-ym.g)/2e-4-y.h@v))
    errs = np.zeros(3)
    for p in PERMS:
        ids = [0, *p]
        idx = np.arange(12).reshape(4, 3)[ids].ravel()
        t = surface(seed(x[ids]))
        errs = np.maximum(errs, [abs(t.v-y.v), max(abs(t.g-y.g[idx])),
                                 np.max(abs(t.h-y.h[np.ix_(idx,idx)]))])
    far = x.copy()
    far[2:,0] += 12.
    b, w = partition(seed(far))
    assert val(w) == 0 and abs(surface(seed(far)).v-b.v) < 1e-14
    ec, et = check_tail_fit()
    ea = check_local_solve()
    before, after = check_sweep()
    print("validation row MSE before/after two sweeps", before, after)
    eb, er = check_capture(), check_geometry()
    print("capture relative barrier, rigid-motion errors", eb, er)
    assert max(eb, er) < 1e-10
    print("mixed inner weight", val(partition(x)[1]))
    print("permutation E,g,H errors", errs)
    print("directional gradient, Hessian FD errors", eg, eh)
    print("tail coefficient, held-out energy errors", ec, et)
    print("recovered active core error", ea)
    assert max(errs) < 1e-10 and max(eg,eh) < 1e-7
    assert max(ec,et,ea) < 1e-8

if __name__ == "__main__":
    main()




