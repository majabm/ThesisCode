# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt
from Master.Liegrupper import SO3, so3, Sp1, sp1, SE3, se3, UDQ, udq
from Master.Liegrupper import SE3_Squared, se3_Squared, SE3N, se3N
from Master.Liegrupper import UDQ_2, udq_2, UDQ_N, udq_N
from Master.Hjelpefunksjoner import identify_lie_group, to_hat_matrix

class FRB: 
    name = "Euler's free rigid body"
    manifold = "S^2"
    Lie_group_name = "SO3" 
    Lie_group = SO3
    Lie_alg = so3
    
    m0 = np.asarray([3,4,3]) 
    dim = 3
    I = np.diag([1,2,3])
    I_inv = np.diag([1,1/2,1/3])
    
    def __init__(self, lie_group):
        self.Lie_group, self.Lie_alg = identify_lie_group(lie_group)
        self.Lie_group_name = self.Lie_group.name

    def set_parameters(self, y0, other):
        self.m0 = y0
        
    def set_lie_group(self, new):
        self.Lie_group, self.Lie_alg = identify_lie_group(new)
        self.Lie_group_name = self.Lie_group.name
        
    def get_init(self):
        return self.m0
    
    def problem_info(self):
        print("Problem:", self.name)
        print("Manifold:", self.manifold)
        print("Lie group:", self.Lie_group.name)
    
    def print_parameteres(self):
        print("Initial value:", self.m0)
        print("Inertia:", np.asarray([self.I[0,0], self.I[1,1], self.I[2,2]]))
    
    def print_options(self):
        print("Lie group options:")
        print("SO(3): exp cay ccsk")
        print("Sp(1): exp cay ccsk")
        
    def func_so3(self, a):  
        b = - self.I_inv.dot(a)
        return self.Lie_alg(b)
    
    def func_sp1(self, a):
        b = - self.I_inv.dot(a) / 2
        return self.Lie_alg(b)
    
    def f(self, a):
        if self.Lie_group_name in ["SO(3)", "SO3"]:
            return self.func_so3(a)
        elif self.Lie_group_name == "Sp(1)":
            return self.func_sp1(a)
        else:
            print("Error, frb, f")
        
    def Lambda_SO3(self, A, b):   
        #A: 3x3 matrix in SO(3)
        #b: vecton in manifold
        A = A.get_mx()
        return A.dot(b)
    
    def Lambda_Sp1(self, Q, b):
        q0, q = Q.get_components()
        B = 2*(q.dot(b))*q + (q0**2 - q.dot(q))*b + 2*q0 * np.cross(q, b)
        return B
    
    def Lambda(self, A, b):
        if self.Lie_group_name in ["SO(3)", "SO3"]:
            return self.Lambda_SO3(A, b)
        elif self.Lie_group_name == "Sp(1)":
            return self.Lambda_Sp1(A, b)
        else:
            print("Error, frb, Lambda")


class Pendulum1Fold:
    name = "1-fold 3D pendulum"
    manifold = "TS^2"
    Lie_group_name = "SE(3)" 
    Lie_group = SE3
    Lie_alg = se3

    q0 = np.asarray([sqrt(2)/2,0,sqrt(2)/2])
    w0 = np.asarray([0,1,0])
    #q0 = np.asarray([0,1,0])
    #w0 = np.asarray([1,0,1])
    y0 = np.concatenate((q0, w0))
    dim = 6
    c = 10   # c = g/L, g gravitational constant, L length of pendulum
    
    def __init__(self, lie_group):
        self.Lie_group, self.Lie_alg = identify_lie_group(lie_group)
        self.Lie_group_name = self.Lie_group.name

    def set_lie_group(self, new):
        self.Lie_group, self.Lie_alg = identify_lie_group(new)
        self.Lie_group_name = self.Lie_group.name
        
    def get_init(self):
        return self.y0
    
    def set_init(self, y0):
        self.y0 = y0
    
    def problem_info(self):
        print("Problem:", self.name)
        print("Manifold:", self.manifold)
        print("Lie group:", self.Lie_group.name)
    
    def print_parameteres(self):
        print("Initial value:", self.q0, self.w0)
        #print("Inertia:", np.asarray([self.I[0,0], self.I[1,1], self.I[2,2]]))
        print("some stuff")
        
    def print_options(self):
        print("Lie group options:")
        print("SE3: exp cay ccsk")
        print("Unit dual quaternions: exp ccsk")
        
    def func_se3(self, y):  
        q, w = np.array_split(y, 2)
        A = w
        h = self.c * np.cross(q, np.asarray([0,0,1]))  
        a = np.cross(q, h)
        return self.Lie_alg(np.concatenate((A, a)))
    
    def func_udq(self, y): 
        q, w = np.array_split(y, 2)
        A = w / 2
        h = self.c * np.cross(q, np.asarray([0,0,1]))  
        a = np.cross(q, h) / 2
        return self.Lie_alg(np.concatenate((A, a)))
    
    def f(self, a):
        if self.Lie_group_name in ["SE(3)", "SE3"]:
            return self.func_se3(a)
        if self.Lie_group_name == "Unit dual quaternions":
            return self.func_udq(a)
        else:
            print("Error, 1-fold pendulum, f")
        
    def Lambda_SE3(self, U, V):
        #U: in SE(3)
        #V:  in manifold
        #result: in M
        R, r = U.get_mx(), U.get_vec()    
        u, v = np.array_split(V, 2)   
        Ru = R.dot(u)
        C = Ru
        c = R.dot(v) + np.cross(r, Ru) 
        return np.concatenate((C, c))
    
    def Lambda_UDQ(self, Q, b): 
        B = UDQ(0, b[0:3], 0, b[3:6])  
        Qc = Q.conjugate()
        A = B*Qc
        C = Q*A
        c0, c, d0, d = C.get_components()
        return np.concatenate((c, d))
    
    def Lambda(self, A, b):
        if self.Lie_group_name in ["SE(3)", "SE3"]:
            return self.Lambda_SE3(A, b)
        if self.Lie_group_name == "Unit dual quaternions":
            return self.Lambda_UDQ(A, b)
        else:
            print("Error, 1-fold pendulum, Lambda")


class Pendulum2Fold:
    name = "2-fold 3D pendulum"
    manifold = "(TS^2)^2"
    Lie_group_name = "(SE(3))^2" 
    Lie_group = SE3_Squared
    Lie_alg = se3_Squared

    q0 = np.asarray([sqrt(2)/2,0,sqrt(2)/2])
    w0 = np.asarray([0,1,0])
    #q0 = np.asarray([0,1,0])
    #w0 = np.asarray([1,0,1])
    y0 = np.concatenate((q0, w0, q0, w0))
    dim = 12
    m1 = 1
    m2 = 1
    L1 = 1
    L2 = 1 
    g = 10
    # everything is 1 and its a miracle
    
    def __init__(self, lie_group):
        self.Lie_group, self.Lie_alg = identify_lie_group(lie_group)
        self.Lie_group_name = self.Lie_group.name
    
    def set_lie_group(self, new):
        self.Lie_group, self.Lie_alg = identify_lie_group(new)
        self.Lie_group_name = self.Lie_group.name
        
    def get_init(self):
        return self.y0
    
    def set_init(self, y0):
        self.y0 = np.concatenate((y0, y0))
        self.q0 = y0[0:3]
        self.w0 = y0[3:6]
        
    def set_init_1(self, y0):
        self.y0[0:6] = y0
        
    def set_init_2(self, y0):
        self.y0[6:12] = y0
    
    def problem_info(self):
        print("Problem:", self.name)
        print("Manifold:", self.manifold)
        print("Lie group:", self.Lie_group.name)
    
    def print_parameteres(self):
        print("Initial value:", self.q0, self.w0, "change this I guess")
        #print("Inertia:", np.asarray([self.I[0,0], self.I[1,1], self.I[2,2]]))
        print("some stuff")
        
    def print_options(self):
        print("Lie group options:")
        print("SE(3)^2: exp cay ccsk")
        print("Unit dual quaternions: exp ccsk")
        
    def find_R_v0(self, q1, q2):
        r11 = np.eye(3)
        r12 = to_hat_matrix(q2).dot(to_hat_matrix(q1))
        r21 = to_hat_matrix(q1).dot(to_hat_matrix(q2))
        r22 = np.eye(3)
        return np.block([[r11, r21], [r12, r22]])
    
    def find_R(self, q1, q2):
        q1_hat = to_hat_matrix(q1)
        q2_hat = to_hat_matrix(q2)
        r11 = (self.m1+self.m2)*self.L1**2 * np.eye(3)
        r12 = self.m2*self.L1*self.L2 * np.transpose(q1_hat).dot(q2_hat)
        r21 = r12.T
        r22 = self.m2*self.L2**2 * np.eye(3)
        return np.block([[r11, r12], [r21, r22]])
    
    def find_h_v0(self, q1, q2, w1, w2):
        g = self.g  
        u1 = -np.linalg.norm(w2)**2 * to_hat_matrix(q2) 
        u1 += g * to_hat_matrix(np.asarray([0,0,1]))
        u1 = u1.dot(q1)
        u2 = -np.linalg.norm(w1)**2 * to_hat_matrix(q1)
        u2 += g * to_hat_matrix(np.asarray([0,0,1]))
        u2 = u2.dot(q2)
        u = np.concatenate((u1, u2))
        R = self.find_R(q1, q2)
        R_inv = np.linalg.inv(R)
        h = R_inv.dot(u)
        return np.split(h, 2)
    
    def find_h(self, q1, q2, w1, w2):
        e3 = np.asarray([0,0,1])
        M12 = self.m2*self.L1*self.L2 * np.eye(3)
        b1 = w2.dot(w2) * M12.dot(np.cross(q1, q2)) 
        b1 -= (self.m1+self.m2)*self.g*self.L1 * np.cross(q1, e3)
        b2 = w1.dot(w1) * M12.dot(np.cross(q2, q1)) 
        b2 -= self.m2*self.g*self.L2 * np.cross(q2, e3)
        b = -np.concatenate((b1, b2))
        R = self.find_R(q1, q2)
        R_inv = np.linalg.inv(R) #!!!
        h = R_inv.dot(b)
        return np.split(h, 2)
        
    def func_se3(self, y):  
        q1, w1, q2, w2 = np.array_split(y, 4)
        h1, h2 = self.find_h(q1, q2, w1, w2)
        A1 = w1
        B1 = np.cross(q1, h1)
        A2 = w2
        B2 = np.cross(q2, h2)
        return self.Lie_alg(np.concatenate((A1, B1, A2, B2)))
    
    def func_udq(self, y):  
        q1, w1, q2, w2 = np.array_split(y, 4)
        h1, h2 = self.find_h(q1, q2, w1, w2)
        A1 = w1
        B1 = np.cross(q1, h1)
        A2 = w2
        B2 = np.cross(q2, h2)
        return self.Lie_alg(np.concatenate((A1, B1, A2, B2))/2)
    
    def f(self, a):
        if self.Lie_group_name in ["SE(3)^2", "SE3^2"]:
            return self.func_se3(a)
        if self.Lie_group_name == "(UDQ)^2":
            return self.func_udq(a)
        else:
            print("Error: 2-fold pendulum, f")
        
    def Lambda_SE3(self, U, V):   
        #U: in SE(3)
        #V:  in manifold
        #result: in M
        R, r = U.get_mx(), U.get_vec()    
        u, v = np.array_split(V, 2)   
        Ru = R.dot(u)
        r_hat = to_hat_matrix(r)
        C = Ru
        c = R.dot(v) + r_hat.dot(Ru)
        return np.concatenate((C, c))
    
    def Lambda_SE3_2(self, A, B):
        A1, A2 = A.get_elements()
        B1, B2 = np.split(B, 2)
        C1, C2 = self.Lambda_SE3(A1, B1), self.Lambda_SE3(A2, B2)
        return np.concatenate((C1, C2))
    
    def Lambda_UDQ(self, Q, b):
        B = UDQ(0, b[0:3], 0, b[3:6])
        Qc = Q.conjugate()
        C = Q*(B*Qc)
        c0, c, d0, d = C.get_components()
        return np.concatenate((c, d))
    
    def Lambda_UDQ_2(self, A, B): 
        A1, A2 = A.get_UDQs()
        B1, B2 = np.split(B, 2)
        C1, C2 = self.Lambda_UDQ(A1, B1), self.Lambda_UDQ(A2, B2) 
        return np.concatenate((C1, C2))
    
    def Lambda(self, A, b):
        if self.Lie_group_name in ["SE(3)^2", "SE3^2"]:
            return self.Lambda_SE3_2(A, b)
        elif self.Lie_group_name == "(UDQ)^2":
            return self.Lambda_UDQ_2(A, b)
        else:
            print("Error. 2-fold pendulum, Lambda")    


class PendulumNFold:
    name = "N-fold 3D pendulum"
    manifold = "(TS^2)^N"
    Lie_group_name = "(SE(3))^N" 
    
    Lie_group = SE3N
    Lie_alg = se3N

    # Various values for N=1
    N = 1
    dim = 6
    q00 = np.asarray([sqrt(2)/2,0,sqrt(2)/2])
    w00 = np.asarray([0,1,0])
    #q00 = np.asarray([0,1,0])
    #w00 = np.asarray([1,0,1])
    y00 = np.concatenate((q00, w00))
    m0 = 1
    L0 = 1 
    g = 10
    
    def __init__(self, lie_group, N):
        self.Lie_group, self.Lie_alg = identify_lie_group(lie_group)
        self.Lie_group_name = self.Lie_group.name
        self.N = N
        self.dim = 6*N
        self.y0 = np.concatenate(([self.y00 for i in range(self.N)]))
        self.m = [self.m0 for i in range(self.N)]
        self.L = [self.L0 for i in range(self.N)]
        self.M = self.find_M()
        
    def get_init(self):
        return self.y0
    
    def set_init(self, y0):
        self.y00 = y0
        self.y0 = np.concatenate(([self.y00 for i in range(self.N)]))
        
    def set_init_i(self, y0, i):
        # Note: All positions relative to pendulum above, not x-y-z coordinates
        self.y0[i*6:(i+1)*6] = y0
        
    def set_lie_group(self, new):
        self.Lie_group, self.Lie_alg = identify_lie_group(new)
        self.Lie_group_name = self.Lie_group.name
    
    def set_m(self, vc):
        self.m = vc
        
    def set_L(self, vc):
        self.L = vc
    
    def problem_info(self):
        print("Problem:", self.name, "with N =", self.N)
        print("Manifold:", self.manifold)
        print("Lie group:", self.Lie_group.name)
    
    def print_parameteres(self): 
        print("Initial value:", self.q00, self.w00, "change this I guess")
        #print("Inertia:", np.asarray([self.I[0,0], self.I[1,1], self.I[2,2]]))
        print("some stuff")
        
    def print_options(self): 
        print("Lie group options:")
        print("SE(3)^N: exp cay ccsk")
        print("Unit dual quaternions: exp ccks")
        
    def find_M(self):
        N = self.N
        m = self.m
        L = self.L
        A = np.random.rand(N,N,3,3)
        I = np.eye(3)
        for i in range(N):
            for j in range(N):
                k = max(i, j)
                summ = sum(m[k:N]) 
                A[i,j] = summ * L[i] * L[j] * I
        return A
    
    def find_R_blocks(self, y):
        N = self.N
        m = self.m
        L = self.L
        I = np.eye(3)
        A = np.random.rand(N,N,3,3) 
        for i in range(N):
            summ = sum(m[i:N])
            A[i,i] = summ * L[i]**2 * I
            qi = y[i*6:i*6+3]
            qi_hat = to_hat_matrix(qi)
            for j in range(i+1, N):
                summ = sum(m[j:N]) 
                qj = y[j*6:j*6+3]
                qj_hat = to_hat_matrix(qj)
                A[i,j] = summ * L[i]*L[j] * qi_hat.dot(qj_hat)
                A[j,i] = A[i,j].T
        return A
    
    def find_R(self, y):
        N = self.N
        m = self.m
        L = self.L
        I = np.eye(3)
        A = np.eye(3*N)
        for i in range(N):
            summ = sum(m[i:N])
            A[i*3:i*3+3,i*3:i*3+3] = summ * L[i]**2 * I
            qi = y[i*6:i*6+3]
            qi_hat = to_hat_matrix(qi)
            for j in range(i+1, N):
                summ = sum(m[j:N]) 
                qj = y[j*6:j*6+3]
                qj_hat = to_hat_matrix(qj)
                B = summ * L[i]*L[j] * qi_hat.dot(qj_hat)
                A[i*3:i*3+3,j*3:j*3+3] = -B   
                A[j*3:j*3+3,i*3:i*3+3] = -B.T  
        return A
    
    def find_b(self, y):
        N = self.N
        m = self.m
        L = self.L
        g = self.g
        M = self.M
        e3 = np.asarray([0,0,1])
        b = np.zeros(3*N)
        for i in range(N):
            qi = y[i*6:i*6+3]   
            summ = sum(m[i:N])
            bi = -summ *g*L[i]* np.cross(qi,e3)
            for j in range(N):
                if i != j:
                    qj = y[j*6:j*6+3] 
                    wj = y[j*6+3:(j+1)*6]
                    Mij = M[i,j]
                    bj = wj.dot(wj) * Mij.dot(np.cross(qi,qj))
                    bi = bi + bj
            b[i*3:i*3+3] = bi
        #print(b)
        return b
        
    def find_h(self, y): 
        R = self.find_R(y) 
        R_inv = np.linalg.inv(R)
        b = - self.find_b(y)
        h = R_inv.dot(b)
        return h
    
    def func_se3N(self, y):   
        h = self.find_h(y)  
        C = np.zeros(self.dim)
        for i in range(self.N):
            qi = y[i*6 : i*6+3]
            wi = y[i*6+3 : (i+1)*6]
            hi = h[3*i:3*i+3]
            A = wi
            B = np.cross(qi, hi)
            C[i*6:(i+1)*6] = np.concatenate((A, B))
        return self.Lie_alg(C)
    
    def func_udqN(self, y): 
        h = self.find_h(y)  
        C = np.zeros(self.dim)
        for i in range(self.N):
            qi = y[i*6 : i*6+3]
            wi = y[i*6+3 : (i+1)*6]
            hi = h[3*i:3*i+3]
            A = wi
            B = np.cross(qi, hi)
            C[i*6:(i+1)*6] = np.concatenate((A, B)) / 2
        return self.Lie_alg(C)
    
    def f(self, a):
        if self.Lie_group_name == "SE(3)^N":
            return self.func_se3N(a)
        if self.Lie_group_name == "(UDQ)^N":
            return self.func_udqN(a)
        else:
            print("Error: N-fold pendulum, f")
        
    def Lambda_SE3(self, U, V):   
        #U: in SE(3)
        #V:  in manifold
        #result: in M
        R, r = U.get_mx(), U.get_vec()    
        u, v = np.array_split(V, 2)   
        Ru = R.dot(u)
        r_hat = to_hat_matrix(r)
        C = Ru
        c = R.dot(v) + r_hat.dot(Ru)
        return np.concatenate((C, c))
    
    def Lambda_SE3N(self, A, b):  # A i G, b i M

        C = np.zeros(6*self.N)
        A = A.get_elements()
        for i in range(self.N):
            U = A[i]
            V = b[i*6:(i+1)*6]
            W = self.Lambda_SE3(U, V)
            C[i*6:(i+1)*6] = W
        return C
    
    def Lambda_UDQ(self, Q, b):
        B = UDQ(0, b[0:3], 0, b[3:6])
        Qc = Q.conjugate()
        C = Q*(B*Qc)
        c0, c, d0, d = C.get_components()
        return np.concatenate((c, d))
    
    def Lambda_UDQN(self, A, b):  
        C = np.zeros(6*self.N)
        A = A.get_elements()
        for i in range(self.N):
            U = A[i]
            V = b[i*6:(i+1)*6]
            W = self.Lambda_UDQ(U, V)
            C[i*6:(i+1)*6] = W
        return C
    
    def Lambda(self, A, b):
        if self.Lie_group_name == "SE(3)^N":
            return self.Lambda_SE3N(A, b)
        elif self.Lie_group_name == "(UDQ)^N":
            return self.Lambda_UDQN(A, b)
        else:
            print("Error. N-fold pendulum, Lambda")  
          
