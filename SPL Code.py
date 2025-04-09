import numpy as np
import sympy as syp
import picos as pic
import matplotlib.pyplot as plt
import pickle
import time
import os

def load_output_probabilities_noisy():
    filename = '''insert file'''
    with open(filename, 'rb') as file:
        loaded_dictionary = pickle.load(file)
    return loaded_dictionary, filename

output_probs, filename = load_output_probabilities_noisy()
# print(output_probs)

eta_val = 0.8965
tilt_val = 0.0
phi_val = np.pi/2
t_vals = np.arange(0.96, 0.98, 0.01)

for key, value in output_probs.items():
    if value != 0:
        [syp.var(str(i), **i.assumptions0) for i in value.atoms(syp.Symbol)]

numeric_p = {}
negative_val_keys = []
for t_val in t_vals:
    for key, value in output_probs.items():
        try:
            numeric_p[(key, t_val)] = float(complex(value.subs([(t,t_val), (eta,eta_val),(phi,phi_val), (tilt,tilt_val) ])).real)
        except:
            
            try:
                numeric_p[(key, t_val)] = float(complex(value.subs([(t,t_val)])).real)
            except:
                
                try:
                    numeric_p[(key, t_val)] = float(complex(value.subs([(t,t_val),(phi,phi_val)])).real)
                except:
                    
                    try:
                        numeric_p[(key, t_val)] = float(complex(value.subs([(t,t_val),(tilt,tilt_val)])).real)
                    except:
                        
                        try:
                            numeric_p[(key, t_val)] = float(value)
                        except:   
                            
                            print("Error in key: ", key)
                            print("Value: ", value)
                            print(value.atoms())
                    
        if numeric_p[(key, t_val)] < 0:
            negative_val_keys.append((key, t_val))

p_len = len(output_probs)
outcomes_per_party = round(p_len**(1/3))
X_v = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '15', '16', '20']
all_outcome_indices = ['0'] + X_v
# print("X_v =", X_v)

total_sum = 0
for (key, t_val), value in numeric_p.items():
    if any(outcome in key for outcome in all_outcome_indices):
        total_sum += value
print("Total sum over all_outcome_indices:", total_sum)

for t_val in t_vals:
    i_max = len(X_v)
    j_max = len(X_v)
    k_max = len(X_v)
    s_max = 2
    
    obj = pic.RealVariable("obj")

    Q = pic.RealVariable("Q", i_max * j_max * k_max * s_max)# define variables for distribution vector q and reshape it
    q = {(i, j, k, s): Q[i * j_max * k_max * s_max + j * k_max * s_max + k * s_max + s]
        for i in range(i_max)
            for j in range(j_max)
                for k in range(k_max)
                    for s in range(s_max)}
    # print(q)

    constraints = [] # list of needed constraints
    # 0.Constraint:
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                for s in range(s_max):
                    constraints.append(q[i, j, k, s] >= 0)
    sum_over_ijks = sum(q[i, j, k, s] for i in range(i_max) for j in range(j_max) for k in range(k_max) for s in range(s_max))
    constraints.append(sum_over_ijks == 1)

    # 1.Constraint:
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                sum_over_s = sum(q[i, j, k, s] for s in range(s_max))
                constraints.append(sum_over_s == 1/(0.25 + 3 * tilt_val**2) * numeric_p[(X_v[i], X_v[j], X_v[k]), t_val])

    # 2.Constraint:
    def D_i(i):
        return (0.5 - tilt_val)/(0.5 + tilt_val) * sum(numeric_p[(X_v[i], '0', X_v[k]), t_val] for k in range(len(X_v))) - sum(numeric_p[(X_v[i], X_v[j], '0'), t_val] for j in range(len(X_v)))
    def D_j(j):
        return (0.5 - tilt_val)/(0.5 + tilt_val) * sum(numeric_p[(X_v[i], X_v[j], '0'), t_val] for i in range(len(X_v))) - sum(numeric_p[('0', X_v[j], X_v[k]), t_val] for k in range(len(X_v)))
    def D_k(k):
        return (0.5 - tilt_val)/(0.5 + tilt_val) * sum(numeric_p[('0', X_v[j], X_v[k]), t_val] for j in range(len(X_v))) - sum(numeric_p[(X_v[i], '0', X_v[k]), t_val] for i in range(len(X_v)))

    for i in range(i_max):
        sum_over_jk_0 = sum(q[i, j, k, 0] for j in range(j_max) for k in range(k_max))
        sum_over_jk_1 = sum(q[i, j, k, 1] for j in range(j_max) for k in range(k_max))
        constraints.append(sum_over_jk_1 - (0.5 - tilt_val)/(0.5 + tilt_val) * sum_over_jk_0 == 1/(0.25 + 3 * tilt_val**2) * D_i(i))
    for j in range(j_max):
        sum_over_ik_0 = sum(q[i, j, k, 0] for i in range(i_max) for k in range(k_max))
        sum_over_ik_1 = sum(q[i, j, k, 1] for i in range(i_max) for k in range(k_max))
        constraints.append(sum_over_ik_1 - (0.5 - tilt_val)/(0.5 + tilt_val) * sum_over_ik_0 == 1/(0.25 + 3 * tilt_val**2) * D_j(j))
    for k in range(k_max):
        sum_over_ij_0 = sum(q[i, j, k, 0] for i in range(i_max) for j in range(j_max))
        sum_over_ij_1 = sum(q[i, j, k, 1] for i in range(i_max) for j in range(j_max))
        constraints.append(sum_over_ij_1 - (0.5 - tilt_val)/(0.5 + tilt_val) * sum_over_ij_0 == 1/(0.25 + 3 * tilt_val**2) * D_k(k))

    constraints.append(obj >= 0)

    problem = pic.Problem() # initializes a new optimization problem
    constraints_string = [] # adding constraints to the problem
    for c in constraints:
        # print(c)
        problem.add_constraint(c)
        constraints_string.append(str(c))

    # Rewrite constraints string such that it is more readable.
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                for t in range(s_max):
                     constraints_string = [c.replace(f"[{i * j_max * k_max * s_max + j * k_max * s_max + k * s_max + t}]", f"[{i},{j},{k},{t}]") for c in constraints_string]
    problem.set_objective("min", obj)
    
    problem.options.solver = "mosek"
    solution = problem.solve(verbosity=False, apply_solution = True, primals=False)
    # print(solution.status)       
    print(solution)