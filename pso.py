import numpy as np

def function_sphere(x):
    return np.sum(x**2, axis=1)

# global best
def pso(nparts, ndims, niter, function, lb, ub, W, C1, C2):
    # Initialize the particles
    x = np.random.random((nparts, ndims)) * (ub - lb) + lb
    v = np.zeros((nparts, ndims))
    fx = function(x)
    pbest = x.copy()
    pbest_fx = fx.copy()
    idx = np.argmin(fx)
    gbest = x[idx].copy()

    while niter > 0:
        niter = niter - 1
        v = W*v + C1*np.random.random((nparts, ndims)) * (pbest - x) + C2*np.random.random((nparts, ndims)) * (gbest - x)
        x = x + v
        fx = function(x)
        idx = fx < pbest_fx
        pbest[idx] = x[idx]
        pbest_fx[idx] = fx[idx]
        idx = np.argmin(pbest_fx)
        gbest = pbest[idx].copy()
        print((gbest*gbest).sum())

    return gbest



if __name__ == "__main__":
    nparts = 10
    ndims = 2
    niter = 20 
    lb = -10
    ub = 10
    W = 0.7 # Frans Van den Bergh
    C1 = 1.4
    C2 = 1.4
    gbest = pso(nparts, ndims, niter, function_sphere, lb, ub, W, C1, C2)
    #print(gbest)
    #print((gbest*gbest).sum())

