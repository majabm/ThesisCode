# -*- coding: utf-8 -*-

import numpy as np
from Test.C2_Problems import FRB, HeavyTop, Pendulum1Fold, Pendulum2Fold
from Test.C2_Problems import PendulumNFold


class RKMK:
    Problem = FRB("SO(3)")
    N = 1000
    t0 = 0
    tf = 5 
    h = 0.01
    dim = 3
    
    coordinate_map = "exp"
    method = "Lie Euler"
    
    sol = np.zeros((1001,dim))
    time = np.zeros(1001)
    
    def __init__(self, prob, coord_map, N = None, tf = None):
        self.h = (self.tf-self.t0)/self.N
        self.Problem = prob
        self.coordinate_map = coord_map
        self.dim = prob.dim
        
        if N:
            self.set_N(N)
        if tf:
            self.set_tf(tf)

    def set_N(self, new_N):
        self.N = new_N
        self.h = (self.tf-self.t0)/self.N
        self.sol = np.zeros((self.N,3))
        self.time = np.zeros(self.N)
        
    def set_tf(self, new_tf):
        self.tf = new_tf
        self.h = (self.tf-self.t0)/self.N
        
    def set_coord_map(self, new_map):
        self.coordinate_map = new_map
    
    def get_solution(self):
        return self.sol
            
    def get_time(self):
        return self.time
    
    def print(self):
        self.Problem.problem_info()
        print("Coordinate map:", self.coordinate_map)
        print()
        self.Problem.print_parameteres()
        print("End time:", self.tf)
        print("Steps:", self.N)
        print("Step length:", self.h)
            
    def make_plot_name(self, type_of_plot):
        name = type_of_plot + ", "
        name = name + self.Problem.name + ", " + self.Problem.Lie_group_name + ", "
        name = name + self.coordinate_map + ", " + self.method
        name = name + ", tf=" + str(self.tf) + ", N=" + str(self.N)
        return name
    
    def make_order_plot_name(self, N0, Nf, Nref):
        name = "Order plot, "
        name = name +self.Problem.name + ", " + self.Problem.Lie_group_name + ", "
        name = name + self.coordinate_map + ", " + self.method
        name = name + ", tf=" + str(self.tf) + ", N0=" + str(N0) + ", Nf=" + str(Nf)
        name = name + ", Nref=" + str(Nref)
        return name
    
    def make_comparison_plot_name(self, type_of_plot, N0, Nf, Nref):
        name = type_of_plot + ", "
        name = name + self.Problem.name + ", " + self.method
        name = name + ", tf=" + str(self.tf) + ", N0=" + str(N0) + ", Nf=" + str(Nf)
        name = name + ", Nref=" + str(Nref)
        return name
        
    def numerical_approximation(self, method_name): 
        t = np.linspace(0,self.tf,self.N+1)
        y = np.zeros((self.N+1,self.dim))
        y[0,:] = self.Problem.get_init()
        method = self.identify_method(method_name)
        for i in range(self.N):
            y[i+1] = method(y[i])
        self.sol = y
        self.time = t
    
    def identify_method(self, name):
        if name in ["lie_euler", "Lie_Euler", "Lie_euler", "lie_Euler",
                    "lie-euler", "Lie-Euler", "Lie-euler", "lie-Euler",
                    "Lie Euler", "Lie euler", "lie Euler", "lie euler",
                    "LieEuler", "Lieeuler", "lieEuler", "lieeuler",
                    "le", "LE", "Le", "lE"]:
            self.method = "Lie Euler"
            return self.lie_euler
        if name in ["improved lie euler", "improved Lie Euler",
                    "impr lie euler", "impr Lie Euler",
                    "impr", "Impr", "ile", "iLE"]:
            self.method = "Heun Euler"
            return self.improved_lie_euler
        if name in ["rkmk4", "mkrk4", "MKRK4", "RKMK4", "mk4", "MK4, rkmk, mkrk"]:
            self.method = "RKMK4"
            return self.rkmk4
        else:
            print("Error: Method identification")

    def function(self, omega, y):
        A = omega.Phi(self.coordinate_map)   # A in G
        B = self.Problem.Lambda(A, y)    # B in M
        C = self.Problem.f(B)    # C in g 
        D = omega.dPhiinv(C, self.coordinate_map)   # D in g
        E = D.mult_w_scalar(self.h)   # E in g
        return E
    
    def function_k1(self, y):
        zero = self.Problem.Lie_alg(np.zeros(self.dim))
        A = zero.get_Lie_group_id()
        B = self.Problem.Lambda(A, y)    
        C = self.Problem.f(B)   
        E = C.mult_w_scalar(self.h)
        return E
    
    def lie_euler(self, y0):
        # input: y0 - initial value
        # output: value after one step
        zero = self.Problem.Lie_alg(np.zeros(self.dim))
        k1 = self.function(zero, y0) 
        E = k1.Phi(self.coordinate_map)
        F = self.Problem.Lambda(E, y0)
        return F
    
    def improved_lie_euler(self, y0):
        # input: m0 - initial value
        # output: value after one step
        k1 = self.function_k1(y0)
        k2 = self.function(k1, y0)
        O1 = k1+k2
        O1 = O1.mult_w_scalar(1/2)
        E = O1.Phi(self.coordinate_map)
        F = self.Problem.Lambda(E, y0)
        return F
    
    def rkmk4(self, y0):
        k1 = self.function_k1(y0)
        k2 = self.function(k1.mult_w_scalar(1/2), y0)  
        k3 = self.function(k2.mult_w_scalar(1/2), y0)  
        k4 = self.function(k3, y0)
        
        O1 = k1 + k2.mult_w_scalar(2) + k3.mult_w_scalar(2) + k4
        O1 = O1.mult_w_scalar(1/6)
        E = O1.Phi(self.coordinate_map)
        F = self.Problem.Lambda(E, y0)
        return F

