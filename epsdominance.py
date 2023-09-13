import numpy as np
eps = np.array([0.05, 0.05])

lb_fx = np.array([0.0, 0.0])

def compute_eps(f):
    return np.floor((f - lb_fx) / eps)

def is_dominated(a, b):
    """
    Check if a is dominated by b.
    """
    a = compute_eps(a)
    b = compute_eps(b)
    return np.all(a >= b) and np.any(a > b)

def is_non_dominated(a, b):
    return not is_dominated(a, b) and not is_dominated(b, a)

def update_non_dominated_solutions(solutions_x, new_solution_x, solutions_fx, new_solution_fx):
    """
    Update and return the list of non-dominated solutions with the new solution.
    """
    # Initialize lists to hold the new set of non-dominated solutions
    new_solutions_x = []
    new_solutions_fx = []
    
    # Flag to check if the new solution is dominated or not
    new_solution_dominated = False
    
    for i in range(len(solutions_x)):
        # If the existing solution is dominated by the new solution, skip it
        # if np.all(compute_eps(new_solution_fx) == compute_eps(solutions_fx[i])): # Si pertenecen a la misma caja, tenemos que tener un metodo diferente para hacer una comparacion de grano fino (vectores, dominancia, distancias).
        if is_dominated(solutions_fx[i], new_solution_fx):
            continue
            
        # If the new solution is dominated by an existing solution, flag it
        if is_dominated(new_solution_fx, solutions_fx[i]) or np.all(compute_eps(new_solution_fx) == compute_eps(solutions_fx[i])):
            new_solution_dominated = True
        
        # Otherwise, keep the existing solution
        new_solutions_x.append(solutions_x[i])
        new_solutions_fx.append(solutions_fx[i])
    
    # If the new solution is not dominated, add it
    if not new_solution_dominated:
        new_solutions_x.append(new_solution_x)
        new_solutions_fx.append(new_solution_fx)
    solutions_x[:] = new_solutions_x
    solutions_fx[:] = new_solutions_fx
    return new_solutions_x, new_solutions_fx




if __name__ == "__main__":
    # Initialize with some non-dominated solutions

    schafer_f1 = lambda x: np.column_stack(((x**2).sum(axis=1), ((x - 2)**2).sum(axis=1)))
    solutions_x = []
    solutions_fx = []
    new_solutions_x = np.array([
        [1.0, 4.0],
        [2.0, 3.0],
        [4.0, 1.0],
        [1.0, 5.0],
        [0.0, 3.0],
        [5.0, 0.0],
        [3.0, 2.0]
    ])
    print(new_solutions_x)
    new_solutions_fx = schafer_f1(new_solutions_x)
    print(new_solutions_fx)

    
    # Update non-dominated solutions
    for new_solution_x, new_solution_fx in zip(new_solutions_x, new_solutions_fx):
        print("New solution to consider (x):", new_solution_x, " fX:", new_solution_fx)
        update_non_dominated_solutions(solutions_x, new_solution_x, solutions_fx, new_solution_fx)
        print(f"New non-dominated solutions are:")
        for x,fx in zip(solutions_x, solutions_fx):
            print('x: ', x, ' fx:', fx)

    print("Final non-dominated solutions are:")
    print(solutions_x)
    print(solutions_fx)
