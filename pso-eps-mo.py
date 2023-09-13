import numpy as np
from epsdominance import update_non_dominated_solutions, is_non_dominated
import random

def function_schafer_1(x):
    f1 = np.sum(x**2, axis=1)
    f2 = np.sum((x-2)**2, axis=1)
    return np.column_stack((f1, f2))


def pso(nparts, ndims, niter, function, lb, ub, W, C1, C2):
    # Initialize the particles
    x = np.random.random((nparts, ndims)) * (ub - lb) + lb
    v = np.zeros((nparts, ndims))
    fx = function(x)
    pbest = x.copy()
    pbest_fx = fx.copy()
    gbest = []
    gbest_fx = []

    for pb_x, pb_fx in zip(pbest, pbest_fx):
        update_non_dominated_solutions(gbest, pb_x, gbest_fx, pb_fx)


    while niter > 0:
        niter = niter - 1
        selected_gbest = random.choices(gbest, k=nparts)
        selected_gbest = np.vstack(selected_gbest)
        v = W*v + C1*np.random.random((nparts, ndims)) * (pbest - x) + C2*np.random.random((nparts, ndims)) * (selected_gbest - x)
        x = x + v

        idx = x < lb
        x[idx] = lb

        idx = x > ub
        x[idx] = ub

        fx = function(x)

        idx = (fx[:,0] <= pbest_fx[:,0]) & (fx[:,1] <= pbest_fx[:,1])
        #idx = is_non_dominated(fx, ) 
        #print(idx)

        pbest[idx] = x[idx]
        pbest_fx[idx] = fx[idx]
        for pb_x, pb_fx in zip(x, fx):
            update_non_dominated_solutions(gbest, pb_x, gbest_fx, pb_fx)

    return gbest, gbest_fx


if __name__ == "__main__":
    nparts = 10
    ndims = 1
    niter = 200 
    lb = -10.0
    ub = 10.0
    W = 0.7 # Frans Van den Bergh
    C1 = 1.4
    C2 = 1.4
    gbest,gbest_fx = pso(nparts, ndims, niter, function_schafer_1, lb, ub, W, C1, C2)
    print(gbest_fx)
    np.savetxt('pareto.txt', gbest_fx)

    #print(gbest)
    #print((gbest*gbest).sum())

