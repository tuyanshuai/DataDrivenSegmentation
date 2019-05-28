import numpy as np
from numpy import pi as pi
from matplotlib import pyplot

import pde_adjoint_solver_dirichlet as model

n = 65
SNR = 10


# Part 1: examine the solver part with examples with analytic solution

#u = np.zeros((n-1)**2)
#def force(x, y):
#    return 2*pi**2*np.sin(pi*x)*np.sin(pi*y)
#def true_solu(x, y):
#    return np.sin(pi*x)*np.sin(pi*y)

#u = np.zeros((n-1)**2)
#def force(x,y):
#    return 2*y*(1-y)+2*x*(1-x)
#def true_solu(x,y):
#    return x*(1-x)*y*(1-y)

#u = np.zeros((n-1)**2)
#def force(x,y):
#    return (2+pi**2*y*(1-y))*np.sin(pi*x)
#def true_solu(x,y):
#    return np.sin(pi*x)*y*(1-y)

#u = np.zeros((n-1,n-1)) # u[i, j] stores the value of u(x, y), with x=(i+0.5)/(n-1), y=(j+0.5)/(n-1)
#for i in range(n-1):
#    for j in range(n-1):
#        u[i,j] = np.log(1+10*(j+0.5)/(n-1))    # u(x,y) = log(1+10*y)
#u = u.reshape((-1))
#def force(x,y):
#    return (-8+40*y)*x*(1-x)+2*y*(1-y)*(1+10*y)
#def true_solu(x,y):
#    return x*(1-x)*y*(1-y)
#
#modelSolver = model.pde_adj_solver(n, 1, SNR, force)
#solution = modelSolver.solve(u).reshape((n,n))
#y_p, x_p = (np.arange(n+1)-0.5)/(n-1), (np.arange(n+1)-0.5)/(n-1)
#pmesh_y, pmesh_x = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))
#true_solution = true_solu(pmesh_x, pmesh_y) # true_solu[i,j] should store true_solu(x, y), with x=(i+0.5)/(n-1), y=(j+0.5)/(n-1)
#y_u, x_u = np.arange(n)/(n-1), np.arange(n)/(n-1)
#
#pyplot.figure()
#pyplot.pcolor(x_u, y_u, u.reshape((n-1,n-1)).T) # To plot u correctly, u[i, j] should store true_solu(x, y), with x=(j+0.5)/(n-1), y=(i+0.5)/(n-1)
#pyplot.colorbar()
#pyplot.show()
#
#diff = solution-true_solution
#pyplot.figure()
#pyplot.pcolor(x_p, y_p, diff.T)
#pyplot.colorbar()
#pyplot.show()
#
#pyplot.figure()
#pyplot.pcolor(x_p, y_p, true_solution.T)
#pyplot.colorbar()
#pyplot.show()
#
#pyplot.figure()
#pyplot.pcolor(x_p, y_p, solution.T)
#pyplot.colorbar()
#pyplot.show()
#
#print('L2 error: %5.3e'%(np.linalg.norm(diff)/(n-1)))


# Part 2: Finite difference to exam the adjoint gradient

#import utility
#
#def force(x, y):
#    return 20/pi*(np.exp(-20*((x-0.3)**2+(y-0.3)**2))+np.exp(-20*((x-0.3)**2+(y-0.7)**2)))
#
#manualSeed = 20180629
#synthetic_u = utility.read_image('symmetry', n)
#Sscale = np.linalg.norm(synthetic_u, ord=np.inf) 
#modelSolver = model.pde_adj_solver(n, Sscale, SNR, force, True)
#true_solution = modelSolver.solve(synthetic_u)
#observation, noise = utility.construct_ob(true_solution, Sscale, n, SNR, manualSeed, True)
#u0 = np.random.normal(0,1, (n-1)**2)
#p0, F0, duF0 = modelSolver.lnprob(u0, observation)
#
#fnerror = np.zeros((6,5))
#for i in range(6):
#    idx = np.random.randint(0, (n-1)**2, 1)[0]
#    mask = np.eye(1, (n-1)**2, idx)[0]
#    for j in range(5):
#        u_new = u0 + 10**(-j-2)*mask
#        p_new, F_new, duF_new = modelSolver.lnprob(u_new, observation)
#        fnerror[i,j] = F_new[0] - ( F0[0] + (u_new-u0)@duF0[0] )
#        print('Test %d, position %d, stepsize %5.3e, Fnerror %5.3e'%(i, idx, 10**(-j-2), fnerror[i,j]))
#        
#absfnerror = np.abs(fnerror)
#rate = np.zeros((6,4))
#for i in range(4):
#    rate[:,i] = absfnerror[:,i]/absfnerror[:,i+1]
#
#pyplot.figure()
#pyplot.semilogy(absfnerror.T)
#pyplot.show()
#
#pyplot.figure()
#pyplot.plot(rate.T)
#pyplot.show()



# Part 3: Symmetric property of the model

#import utility
#
#def force(x, y):
#    return 20/pi*(np.exp(-20*((x-0.3)**2+(y-0.3)**2))+np.exp(-20*((x-0.3)**2+(y-0.7)**2)))
#
#manualSeed = 20180629
#synthetic_u = utility.read_image('symmetry', n)
#Sscale = np.linalg.norm(synthetic_u, ord=np.inf) 
#modelSolver = model.pde_adj_solver(n, Sscale, SNR, force, True)
#true_solution = modelSolver.solve(synthetic_u)
#observation, noise = utility.construct_ob(true_solution, Sscale, n, SNR, manualSeed, True, True)
#
#u0 = np.random.normal(0, 1, (n-1)**2)
##u0 = synthetic_u.copy()
#p0, F0, duF0 = modelSolver.lnprob(u0, observation)
#p0 = p0[0].reshape((n, n))
#duF0 = duF0[0].reshape((n-1, n-1))
#u0 = u0.reshape((n-1,n-1))
#sym_u = np.zeros((n-1,n-1))
#for i in range(n-1):
#    sym_u[:, i] = u0[:, n-2-i]
#p_sym, F_sym, duF_sym = modelSolver.lnprob(sym_u.reshape(-1), observation)
#p_sym = p_sym[0].reshape((n, n))
#duF_sym = duF_sym[0].reshape((n-1, n-1))
#
#y_u, x_u = np.arange(n)/(n-1), np.arange(n)/(n-1)
#y_p, x_p = (np.arange(n+1)-0.5)/(n-1), (np.arange(n+1)-0.5)/(n-1)
#
#pyplot.figure()
#pyplot.pcolor(x_u, y_u, u0.T)
#pyplot.colorbar()
#pyplot.show()
#
#pyplot.figure()
#pyplot.pcolor(x_p, y_p, p0.T)
#pyplot.colorbar()
#pyplot.show()
#
#pyplot.figure()
#pyplot.pcolor(x_u, y_u, sym_u.T)
#pyplot.colorbar()
#pyplot.show()
#
#pyplot.figure()
#pyplot.pcolor(x_p, y_p, p_sym.T)
#pyplot.colorbar()
#pyplot.show()
#
#p_sym_sym = np.zeros((n,n))
#for i in range(n):
#    p_sym_sym[:, i] = p_sym[:, n-1-i]
#duF_sym_sym = np.zeros((n-1,n-1))
#for i in range(n-1):
#    duF_sym_sym[:, i] = duF_sym[:, n-2-i]
#print('Symmetric difference %5.3e'%(np.linalg.norm(p0-p_sym_sym)))
#print('Symmetric difference %5.3e'%(np.linalg.norm(F0[0]-F_sym[0])))
#print('Symmetric difference %5.3e'%(np.linalg.norm(duF0[0]-duF_sym_sym[0])))


# Part 4, profile

import utility

def force(x, y):
    return 20/pi*(np.exp(-20*((x-0.3)**2+(y-0.3)**2))+np.exp(-20*((x-0.3)**2+(y-0.7)**2)))

manualSeed = 20180629
synthetic_u = utility.read_image('symmetry', n)
Sscale = np.linalg.norm(synthetic_u, ord=np.inf) 
modelSolver = model.pde_adj_solver(n, Sscale, SNR, force)
true_solution = modelSolver.solve(synthetic_u)
observation, noise = utility.construct_ob(true_solution, Sscale, n, SNR, manualSeed, True, True)

u = np.random.normal(0, 1, (64, (n-1)**2))
p, F, duF = modelSolver.lnprob(u, observation)