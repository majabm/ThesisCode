# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as sl
from math import sin, cos, tan, sqrt
from mpmath import cot

from Hjelpefunksjoner import identify_lie_group, to_hat_matrix, from_hat_matrix, normalize



class SO3:
    name = "SO(3)"
    
    A = np.eye(3)
    
    def __init__(self, mx):
        self.A = mx
        
    def print(self):
        print(self.A)
        
    def get_mx(self):
        return self.A
    
    def identity(self):
        return SO3(np.eye(3))

class so3:
    u = np.zeros(3)
    
    def __init__(self, vec):
        self.u = vec
        
    def set_vec(self, vec):
        self.u = vec
        
    def get_vec(self):
        return self.u
    
    def get_Lie_group_id(self):
        return SO3(np.eye(3))
    
    def print(self):
        print(self.u)
        
    def __add__(self, other):
        vec = self.get_vec() + other.get_vec()
        return so3(vec)
    
    def mult_w_scalar(self, a):
        return so3(a * self.u)
    
    def norm(self):
        # 2-norm 
        return np.linalg.norm(self.u)
    
    def exp(self):
        A = to_hat_matrix(self.get_vec())
        a = np.linalg.norm(A, 2) 
        TOL = 1e-3
        
        if a > TOL:
            c1 = sin(a)/a
            c2 = (1-cos(a)) / a**2
            
        else:
            c1 = 1 - a**2/6 + a**4/120
            c2 = 1/2 - a**2/24 + a**4/720 
            
        B = np.eye(3) + c1*A + c2 * A.dot(A) 
        return SO3(np.asarray(B))
    
    def dexpinv(self, v):
        a = np.linalg.norm(self.u)
        
        TOL = 1e-3
        U = to_hat_matrix(self.get_vec())
        I = np.eye(3)
        
        if a > TOL: 
            c = (1 - (a/2) * (1/tan(a/2))) / a**2
        
        else:
            c = 1/12 + a**2/720 + a**4/30240 
            
        V = I - 1/2*U + c*U.dot(U)
        B = V.dot(v.get_vec())   
        return so3(B)
    
    def cay(self):
        A = to_hat_matrix(self.get_vec())
        a = self.norm()
        
        c = 1 + a**2/4
        B = np.eye(3) + 1/c * (A + A.dot(A)/2)
        return SO3(np.asarray(B))
    
    def dcayinv(self, v):
        I = np.eye(3)
        U = to_hat_matrix(self.get_vec())
        a = np.linalg.norm(U, 2) 
        A = (1+1/4*a**2)*I - 1/2 * U + 1/4 * U.dot(U) 
        B = A.dot(v.get_vec())  
        return so3(B)
    
    def ccsk(self):
        a1, a2, a3 = self.get_vec()
        E1 = np.eye(3)
        c1, c2, c3 = cos(a1), cos(a2), cos(a3)
        s1, s2, s3 = sin(a1), sin(a2), sin(a3)
        E1[1,1] = c1  
        E1[2,2] = c1
        E1[1,2] = -s1
        E1[2,1] = s1
        E2 = np.eye(3)
        E2[0,0] = c2
        E2[2,2] = c2
        E2[0,2] = s2
        E2[2,0] = -s2
        E3 = np.eye(3)
        E3[0,0] = c3 
        E3[1,1] = c3
        E3[0,1] = -s3
        E3[1,0] = s3
        E = E1.dot(E2.dot(E3))
        return SO3(E)
    
    def dccskinv(self, v):
        x1, x2, x3 = self.get_vec()
        c1, c2 = cos(x1), cos(x2)
        s1, s2 = sin(x1), sin(x2)
        t2 = tan(x2)
        y = v.get_vec()
        A = np.eye(3)
        A[1,1] = c1
        A[2,2] = c1/c2
        A[0,1] = s1*s2/c2 #s1*t2
        A[0,2] = -s2*c1/c2#-c1*t2
        A[1,2] = s1
        A[2,1] = -s1/c2
        B = A.dot(y) 
        return so3(B)
    
    def ccsk321(self):
        a1, a2, a3 = self.get_vec()
        E1 = np.eye(3)
        c1, c2, c3 = cos(a1), cos(a2), cos(a3)
        s1, s2, s3 = sin(a1), sin(a2), sin(a3)
        E1[1,1] = c1  
        E1[2,2] = c1
        E1[1,2] = -s1
        E1[2,1] = s1
        E2 = np.eye(3)
        E2[0,0] = c2
        E2[2,2] = c2
        E2[0,2] = s2
        E2[2,0] = -s2
        E3 = np.eye(3)
        E3[0,0] = c3 
        E3[1,1] = c3
        E3[0,1] = -s3
        E3[1,0] = s3
        E = E3.dot(E2.dot(E1))
        return SO3(E)
    
    def dccskinv321(self, v):
        x1, x2, x3 = self.get_vec()
        c1, c2, c3 = cos(x1), cos(x2), cos(x3)
        s1, s2, s3 = sin(x1), sin(x2), sin(x3)
        t2 = tan(x2)
        y = v.get_vec()
        A = np.eye(3)
        A[0,0] = c3/c2
        A[1,1] = c3
        A[0,1] = s3/c2 
        A[1,0] = -s3
        A[2,1] = s2*s3/c2
        A[2,0] = c3*s2/c2
        B = A.dot(y) 
        return so3(B)
    
    def Phi(self, name):
        if name in ["exp", "Exp"]:
            return self.exp()
        elif name == 'cay':
            return self.cay()
        elif name == "ccsk":
            return self.ccsk()
        else:
            print('Error in coordinate map')
        
    def dPhiinv(self, v, name):
        if name in ["exp", "Exp"]:
            return self.dexpinv(v)
        elif name == 'cay':
            return self.dcayinv(v)
        elif name == "ccsk":
            return self.dccskinv(v)
        else:
            print('Error in differential of coordinate map')
            
    def Phi_SYM(self, nr):
        if nr==1:
            return self.ccsk()
        elif nr==2:
            return self.ccsk321()
        else:
            print('Error in coordinate map (SYM)')
        
    def dPhiinv_SYM(self, v, nr):
        if nr==1:
            return self.dccskinv(v)
        elif nr==2:
            return self.dccskinv321(v)
        else:
            print('Error in differential of coordinate map (SYM)')


class Sp1:
    name = "Sp(1)"
    
    q0 = 1
    q = np.zeros(3)
    Q = [q0, q]
    
    def __init__(self, q0, q):
        self.q0 = q0
        self.q = q
        self.Q = [q0, q]
        
    def __mul__(self, other):  
        a0, a = self.q0, self.q
        b0, b = other.get_components()
        q0 = a0*b0 - a.dot(b)
        q = a0*b + b0*a + np.cross(a, b)
        return Sp1(q0, q)
    
    def __add__(self, other):
        a0, a = self.q0, self.q
        b0, b = other.get_components()
        return Sp1(a0+b0, a+b)
        
    def print(self):
        print(self.q0, "+", self.q[0],"i + ", self.q[1],"j + ", self.q[2],"k")
        print(self.rotational_matrix())
        
    def get_components(self):
        return self.q0, self.q
    
    def conjugate(self):
        return Sp1(self.q0, -self.q)
    
    def rotational_matrix(self):
        I = np.eye(3)
        q0 = self.q0
        q = self.q    
        R = (q0**2 - (np.linalg.norm(q))**2) * I
        R += 2 * (q0 * to_hat_matrix(q) + np.outer(q,q))
        return R
    
    def identity(self):
        return Sp1(1, np.zeros(3))
                
class sp1:
    q = np.zeros(3)
    
    def __init__(self, vec):
        self.q = vec
        
    def set_vec(self, vec):
        self.q = vec
        
    def get_vec(self):
        return self.q
    
    def get_Lie_group_id(self):
        return Sp1(1, np.zeros(3))
        
    def print(self):
        print(0, "+", self.q[0],"i + ", self.q[1],"j + ", self.q[2],"k")
        
    def __add__(self, other):
        vec = self.get_vec() + other.get_vec()
        return sp1(vec)
    
    def mult_w_scalar(self, a):
        return sp1(a * self.get_vec())
    
    def norm(self):
        # 2-norm 
        return np.linalg.norm(self.q)
    
    def exp(self):
        x = self.get_vec() 
        alpha = np.linalg.norm(x) 
        TOL = 1e-3
        if alpha > TOL:
            q0 = cos(alpha)
            q = ( sin(alpha) / (alpha) ) * x 
        else:
            q0 = 1 - alpha**2/2 + alpha**4/(4*3*2)
            a = 1 - alpha**2/6 + alpha**4/(5*4*3*2)
            q = a*x     
        return Sp1(q0, q)
    
    def dexpinv(self, v): 
        u = 2*self.get_vec()    
        a = np.linalg.norm(u)
        TOL = 1e-3
        U = to_hat_matrix(u)
        I = np.eye(3)
        if a > TOL: 
            c = (1 - (a/2) * (1/tan(a/2))) / a**2
        else:
            c = 1/12 + a**2/720 + a**4/30240 
        V = I - 1/2*U + c*U.dot(U)
        B = V.dot(v.get_vec())   
        return sp1(B)
    
    def cay(self):
        x = self.get_vec() 
        alpha = np.linalg.norm(x)
        q0 = (4 - alpha**2) / (4 + alpha**2)
        q = (4 / (4 + alpha**2) ) * x
        return Sp1(q0, q)
    
    def dcayinv(self, v):
        u = self.get_vec()
        U = to_hat_matrix(u)
        alpha = np.linalg.norm(u)
        A = (1+alpha**2/4)*np.eye(3) - U + U.dot(U)/2 
        v = v.get_vec()
        return sp1(A.dot(v))
    
    def ccsk(self):
        x1, x2, x3 = self.get_vec()
        c1, c2, c3 = cos(x1), cos(x2), cos(x3)
        s1, s2, s3 = sin(x1), sin(x2), sin(x3)
        q0 = c1*c2*c3 - s1*s2*s3 
        q1 = c1*s2*s3 + s1*c2*c3
        q2 = c1*s2*c3 - s1*c2*s3
        q3 = c1*c2*s3 + s1*s2*c3
        q = np.asarray([q1,q2,q3]) 
        return Sp1(q0, q)
    
    def dccskinv(self, v): 
        x = so3(2*self.get_vec())
        y = x.dccskinv(v)
        return sp1(y.get_vec())
    
    def Phi(self, name):
        if name in ["exp", "Exp"]:
            return self.exp()
        elif name == 'cay':
            return self.cay()
        elif name == "ccsk":
            return self.ccsk()
        else:
            print('Error in coordinate map')
        
    def dPhiinv(self, v, name):
        if name in ["exp", "Exp"]:
            return self.dexpinv(v)
        elif name == 'cay':
            return self.dcayinv(v)
        elif name == "ccsk":
            return self.dccskinv(v)
        else:
            print('Error in differential of coordinate map')
            
def Lambda(Q, b):
    q0, q = Q.get_components()
    B = 2*(q.dot(b))*q + (q0**2 - q.dot(q))*b + 2*q0 * np.cross(q, b)
    return B

class SE3:
    name = "SE(3)"
    
    A = np.eye(3)
    a = np.zeros(3)
    
    def __init__(self, elem):
        self.A = elem[0]
        self.a = elem[1]
        
    def __mul__(self, other):
        A, a = self.get_mx(), self.get_vec()
        B, b = other.get_mx(), other.get_vec()
        C = A.dot(B)
        c = A.dot(b) + a
        return SE3([C,c])
        
    def get_mx(self):
        return self.A

    def get_vec(self):
        return self.a

    def print(self):
        print(self.A, self.a)
        
    def identity(self):
        return SE3([np.eye(3), np.zeros(3)])

class se3:
    u = np.zeros(6)
    
    def __init__(self, x):
        self.u = x
        
    def get_mx(self):
        return self.u[0:3]
    
    def get_vec(self):
        return self.u[3:6]
    
    def get_components(self):
        return self.get_mx(), self.get_vec()
    
    def get(self):
        return self.u
    
    def get_Lie_group_id(self):
        return SE3([np.eye(3), np.zeros(3)])
        
    def print(self):
        print(self.u)
        
    def __add__(self, other): 
        g, u = self.get_mx(), self.get_vec()
        h, v = other.get_mx(), other.get_vec()
        return se3(np.concatenate((g+h, u+v))) 
    
    def mult_w_scalar(self, a):   
        return se3(a * self.u)

    def exp(self):  
        A, a = self.get_mx(), self.get_vec()
        alpha = np.linalg.norm(A)
        U = to_hat_matrix(A)
        TOL = 1e-3
        if alpha > TOL:
            c1 = sin(alpha)/alpha
            c2 = (1-cos(alpha)) / alpha**2
            
        else:
            c1 = 1 - alpha**2/6 + alpha**4/120
            c2 = 1/2 - alpha**2/24 + alpha**4/720 
            
        V = np.eye(3) + c1*U + c2 * U.dot(U) 
        
        TOL = 1e-3
        
        A = U
        B = V
        
        if alpha > TOL:
            c1 = (1 - cos(alpha)) / (alpha**2)
            c2 = (alpha - sin(alpha)) / (alpha**3)
            
        else: 
            c1 = 1/2 - alpha**2/24 + alpha**4/720 
            c2 = 1/6 - alpha**2/120 + alpha**4/5040 
        
        b = (np.eye(3) + c1*A + c2*A@A).dot(a)

        return SE3([B, b])
    
    def dexpinv(self, V): 
        TOL = 1e-3
        A, a = self.get_mx(), self.get_vec()
        B, b = V.get_mx(), V.get_vec()
        alpha = np.linalg.norm(A)
        rho = np.transpose(A).dot(a)
        
        if alpha > TOL: 
            g2 = (1 - (alpha/2) * (1/tan(alpha/2))) / alpha**2
            num = alpha**2 * (1/sin(alpha/2))**2 + 2*alpha*(1/tan(alpha/2)) - 8
            denom = 4*alpha**4
            g2tilde = num/denom
        
        else:
            g2 = 1/12 + alpha**2/720 + alpha**4/30240 
            g2tilde = 1/360 + alpha**2/7560 + alpha**4/201600    
        
        AB = np.cross(A, B)
        AAB = np.cross(A, AB)
        aB = np.cross(a, B)
        Ab = np.cross(A, b)
        aAB = np.cross(a, AB)
        AaB = np.cross(A, aB)
        AAb = np.cross(A, Ab)
        
        C = B - 1/2*AB + g2*AAB
        c = b - 1/2 * (aB + Ab) + rho * g2tilde * AAB + g2 * (aAB + AaB + AAb)  
        return se3(np.concatenate((C, c)))
    
    def cay(self):
        A, a = self.get_components()
        c = 1 + (np.linalg.norm(A))**2 / 4
        A = to_hat_matrix(A)
        I = np.eye(3)
        
        B = I + 1/c * (A + A.dot(A)/2)

        b = I - A/2
        b = np.linalg.inv(b) #######!!!!
        b = b.dot(a)
        return SE3([B, b])
    
    def dcayinv(self, v):
        x, y = self.get_components()
        u, v = v.get_components()
        
        I = np.eye(3) 
        X = to_hat_matrix(x)
        U = to_hat_matrix(u)
        
        A = I - X/2
        B = I + X/2
        C = A.dot(U.dot(B))
        c = from_hat_matrix(C)
        d = A.dot( v + U.dot(y)/2 )
        return se3(np.concatenate((c, d)))
    
    def ccsk(self):
        # Rekkefølge: 4-5-6-1-2-3
        A, a = self.get_components()
        
        # so(3)-part
        A = so3(A)
        B = A.ccsk()
        E1 = B.get_mx()
        
        # R^3-part
        E2 = a
        return SE3([E1,E2])

    def dccskinv(self, v):
        # Rekkefølge: 4-5-6-1-2-3
        x1, x2, x3 = self.get_mx()
        c1, c2, c3 = cos(x1), cos(x2), cos(x3)
        s1, s2, s3 = sin(x1), sin(x2), sin(x3)
        #t2 = tan(x2)
        y = v.get()
        
        A = np.eye(3)
        A[1,1] = c1
        A[2,2] = c1/c2
        A[0,1] = s1*s2/c2 #s1*t2
        A[0,2] = -s2*c1/c2#-c1*t2
        A[1,2] = s1
        A[2,1] = -s1/c2

        b = self.get_vec()
        B = -to_hat_matrix(b)

        C = np.block([[A,               np.zeros((3, 3))],
                      [B, np.eye(3)]])

        D = C.dot(y) 
        return se3(D)
    
    def Phi(self, name):
        if name in ["exp", "Exp"]:
            return self.exp()
        elif name == 'cay':
            return self.cay()
        elif name == "ccsk":
            return self.ccsk()
        else:
            print('Error in coordinate map')
        
    def dPhiinv(self, v, name):
        if name in ["exp", "Exp"]:
            return self.dexpinv(v)
        elif name == 'cay':
            return self.dcayinv(v)
        elif name == "ccsk":
            return self.dccskinv(v)
        else:
            print('Error in differential of coordinate map')
            

class UDQ:
    name = "Unit dual quaternions"
        
    a0 = 1
    a = np.zeros(3)
    A = [a0, a]
    b0 = 1
    b = np.zeros(3)
    B = [b0, b]
    
    def __init__(self, a0, a, b0, b): 
        self.a0 = a0
        self.a = a
        self.A = [a0, a]
        self.b0 = b0
        self.b = b
        self.B = [b0, b]
        
    def __mul__(self, other):  
        a0, a, b0, b = self.get_components()
        c0, c, d0, d = other.get_components()
        A = Sp1(a0, a)
        A_eps = Sp1(b0, b)
        B = Sp1(c0, c)
        B_eps = Sp1(d0, d)
        
        C = A*B
        D = A*B_eps + A_eps*B
        c0, c = C.get_components()
        d0, d = D.get_components()
        return UDQ(c0, c, d0, d) 
        
    def get_components(self):
        return self.a0, self.a, self.b0, self.b

    def get_quaternions(self):  
        return self.A, self.B
        
    def print(self):
        print(self.a0, self.a, self.b0, self.b)
    
    def conjugate(self):
        return UDQ(self.a0, -self.a, self.b0, -self.b)
    
    def identity(self):
        return UDQ(1, np.zeros(3), 0, np.zeros(3))


class udq:
    q = np.zeros(6)
    
    def __init__(self, vec):
        self.q = vec
        
    def set_vec(self, vec):
        self.q = vec
        
    def get_vec(self):
        return self.q
    
    def get_components(self):
        return self.q[0:3], self.q[3:6]
    
    def get_Lie_group_id(self):
        return UDQ(1, np.zeros(3), 0, np.zeros(3))
        
    def print(self):
        print(0, "+", self.q[0],"i + ", self.q[1],"j + ", self.q[2],"k")
        print("+ eps(", 0, "+", self.q[3],"i + ", self.q[4],"j + ", 
              self.q[5],"k )")
        
    def __add__(self, other):
        vec = self.get_vec() + other.get_vec()
        return udq(vec)
    
    def mult_w_scalar(self, a):
        return udq(a * self.get_vec())
    
    def exp(self): 
        x = self.get_vec()  
        x, y = np.split(x, 2)
        alpha = np.linalg.norm(x)  
        TOL = 1e-3
        if alpha > TOL:
            s = sin(alpha) / alpha
        else:
            s = 1 - alpha**2/6 + alpha**4/5*4*3*2
                  
        n = normalize(x)
        a0 = cos(alpha)
        a = s * x
        b0 = -x.dot(y) * s
        b = n.dot(y) * cos(alpha) * n - s * np.cross(n, np.cross(n, y))
        return UDQ(a0, a, b0, b)
    
    def dexpinv(self, V):  
        TOL = 1e-3
        u = 2*self.get_vec()
        v = V.get_vec()
        A, a = np.split(u, 2) 
        B, b = np.split(v, 2) 
        alpha = np.linalg.norm(A)
        rho = np.transpose(A).dot(a)
        
        if alpha > TOL: 
            g2 = (1 - (alpha/2) * (1/tan(alpha/2))) / alpha**2
            num = alpha**2 * (1/sin(alpha/2))**2 + 2*alpha*(1/tan(alpha/2)) - 8
            denom = 4*alpha**4
            g2tilde = num/denom
        
        else:
            g2 = 1/12 + alpha**2/720 + alpha**4/30240 
            g2tilde = 1/360 + alpha**2/7560 + alpha**4/201600    
        
        AB = np.cross(A, B)
        AAB = np.cross(A, AB)
        aB = np.cross(a, B)
        Ab = np.cross(A, b)
        aAB = np.cross(a, AB)
        AaB = np.cross(A, aB)
        AAb = np.cross(A, Ab)
        
        C = B - 1/2*AB + g2*AAB
        c = b - 1/2 * (aB + Ab) + rho * g2tilde * AAB + g2 * (aAB + AaB + AAb)  
        return udq(np.concatenate((C, c)))
    
    def ccsk(self):
        a2,a3,a4,a6,a7,a8 = self.get_vec()
        c2, c3, c4 = cos(a2), cos(a3), cos(a4)
        s2, s3, s4 = sin(a2), sin(a3), sin(a4)
    
        p0 = c2*c3*c4 - s2*s3*s4
        p1 = s2*c3*c4 + c2*s3*s4
        p2 = c2*s3*c4 - s2*c3*s4
        p3 = s2*s3*c4 + c2*c3*s4
        
        s2c3a6 = s2*c3*a6
        c2a7 = c2*a7
        s2a8 = s2*a8
        c2s3a6 = c2*s3*a6
        s2a7 = s2*a7
        c2a8 = c2*a8
        c2c3a6 = c2*c3*a6
        s2s3a6 = s2*s3*a6
        
        q0 = (-s2c3a6 - (c2a7 + s2a8)*s3)*c4 
        q0 += (-c2s3a6 + (s2a7 - c2a8)*c3)*s4
        q1 = (c2c3a6 + (s2a7 - c2a8)*s3)*c4
        q1 += (-s2s3a6 + (c2a7 + s2a8)*c3)*s4
        q2 = (-s2s3a6 + (c2a7 + s2a8)*c3)*c4
        q2 += (-c2c3a6 + (-s2a7 + c2a8)*s3)*s4
        q3 = (c2s3a6 + (-s2a7 + c2a8)*c3)*c4
        q3 += (-s2c3a6 + (-c2a7 - s2a8)*s3)*s4
        return UDQ(p0,np.asarray([p1,p2,p3]),q0,np.asarray([q1,q2,q3]))
    
    def dccskinv(self, v): 
        x = se3(2*self.get_vec())
        y = x.dccskinv(se3(v.get_vec()))
        return udq(y.get())
    
    def Phi(self, name):
        if name in ["exp", "Exp"]:
            return self.exp()
        elif name == "ccsk":
            return self.ccsk()
        else:
            print('Error in coordinate map')
        
    def dPhiinv(self, v, name):
        if name in ["exp", "Exp"]:
            return self.dexpinv(v)
        elif name == "ccsk":
            return self.dccskinv(v)
        else:
            print('Error in differential of coordinate map')


class SE3_Squared:
    name = "SE(3)^2"
    
    A = SE3([np.eye(3), np.zeros(3)])
    B = SE3([np.eye(3), np.zeros(3)]) 
    C = np.asarray([A, B])
    
    def __init__(self, elem):
        self.A = elem[0]
        self.B = elem[1]
        self.C = elem
        
    def get_elements(self):
        return self.A, self.B

    def get(self):
        return self.C

    def print(self):
        self.A.print()
        self.B.print()
        
    def identity(self):
        A = SE3([0,0])
        return SE3_Squared([A.identity(), A.identity()])
        

class se3_Squared:
    u = np.zeros(12)
    
    def __init__(self, x):
        self.u = x
    
    def get_vec(self):
        return self.u
    
    def get_se3_elements(self):
        return np.split(self.u, 2)
        
    def get_Lie_group_id(self):
        A = se3(np.zeros(6))
        return SE3_Squared([A.get_Lie_group_id(), A.get_Lie_group_id()])
        
    def print(self):
        print(self.u)
        
    def __add__(self, other): 
        return se3_Squared(self.get_vec() + other.get_vec())
    
    def mult_w_scalar(self, a):   
        return se3_Squared(a * self.get_vec())
    
    def exp(self):  
        A, B = self.get_se3_elements()
        A, B = se3(A), se3(B)
        E1, E2 = A.exp(), B.exp()
        return SE3_Squared([E1, E2])
    
    def dexpinv(self, V): 
        A, B = self.get_se3_elements()
        X, Y = V.get_se3_elements()
        A, B, X, Y = se3(A), se3(B), se3(X), se3(Y)
        C, D = A.dexpinv(X), B.dexpinv(Y)
        return se3_Squared(np.concatenate((C.get(), D.get())))
    
    def cay(self):
        A, B = self.get_se3_elements()
        A, B = se3(A), se3(B)
        E1, E2 = A.cay(), B.cay()
        return SE3_Squared([E1, E2])

    def dcayinv(self, V):
        A, B = self.get_se3_elements()
        X, Y = V.get_se3_elements()
        A, B, X, Y = se3(A), se3(B), se3(X), se3(Y)
        C, D = A.dcayinv(X), B.dcayinv(Y)
        return se3_Squared(np.concatenate((C.get(), D.get())))
    
    def ccsk(self):
        A, B = self.get_se3_elements()
        A, B = se3(A), se3(B)
        E1, E2 = A.ccsk(), B.ccsk()
        return SE3_Squared([E1, E2])

    def dccskinv(self, V):
        A, B = self.get_se3_elements()
        X, Y = V.get_se3_elements()
        A, B, X, Y = se3(A), se3(B), se3(X), se3(Y)
        C, D = A.dccskinv(X), B.dccskinv(Y)
        return se3_Squared(np.concatenate((C.get(), D.get())))
    
    def Phi(self, name):
        if name in ["exp", "Exp"]:
            return self.exp()
        elif name == 'cay':
            return self.cay()
        elif name == "ccsk":
            return self.ccsk()
        else:
            print('Error in coordinate map')
        
    def dPhiinv(self, v, name):
        if name in ["exp", "Exp"]:
            return self.dexpinv(v)
        elif name == 'cay':
            return self.dcayinv(v)
        elif name == "ccsk":
            return self.dccskinv(v)
        else:
            print('Error in differential of coordinate map')


class SE3N:
    name = "SE(3)^N"
    
    def __init__(self, elem):
        self.N = len(elem)
        self.SE3_elements = elem
    
    def get_elements(self):
        return self.SE3_elements
    
    def get_N(self):
        return self.N

    def print(self):
        for elem in self.SE3_elements:
            elem.print()
        

class se3N:
    def __init__(self, x):
        self.N = int(len(x)/6)
        self.se3_elements = x
    
    def get_se3_elements(self):
        return np.split(self.se3_elements, self.N)
    
    def get_se3_elem_i(self, i):
        return self.se3_elements[i*6:(i+1)*6]
        
    def get_vec(self):
        return self.se3_elements
        
    def get_Lie_group_id(self): 
        A = se3(np.zeros(6))
        B = [A.get_Lie_group_id() for i in range(self.N)]
        return SE3N(B)
        
    def print(self):
        elements = self.se3_elements
        for i in range(self.N):
            u = elements[i*6:(i+1)*6]
            print(u)
        
    def __add__(self, other): 
        return se3N(self.get_vec() + other.get_vec())
    
    def mult_w_scalar(self, a):  
        return se3N(a * self.get_vec())
    
    def exp(self):  
        E = []
        for i in range(self.N):
            A = self.get_se3_elem_i(i)
            A = se3(A)
            A = A.exp()
            E.append(A)
        return SE3N(E)
    
    def dexpinv(self, V):
        D = np.zeros(6*self.N)
        for i in range(self.N):
            A = self.get_se3_elem_i(i)
            B = V.get_se3_elem_i(i)
            A, B = se3(A), se3(B)
            C = A.dexpinv(B)
            D[i*6:(i+1)*6] = C.get()
        return se3N(D)
    
    def cay(self):
        C = []
        for i in range(self.N):
            A = self.get_se3_elem_i(i)
            A = se3(A)
            A = A.cay()
            C.append(A)
        return SE3N(C)

    def dcayinv(self, V): 
        D = np.zeros(6*self.N)
        for i in range(self.N):
            A = self.get_se3_elem_i(i)
            B = V.get_se3_elem_i(i)
            A, B = se3(A), se3(B)
            C = A.dcayinv(B)
            D[i*6:(i+1)*6] = C.get()
        return se3N(D)
    
    def ccsk(self):
        C = []
        for i in range(self.N):
            A = self.get_se3_elem_i(i)
            A = se3(A)
            A = A.ccsk()
            C.append(A)
        return SE3N(C)

    def dccskinv(self, V): 
        D = np.zeros(6*self.N)
        for i in range(self.N):
            A = self.get_se3_elem_i(i)
            B = V.get_se3_elem_i(i)
            A, B = se3(A), se3(B)
            C = A.dccskinv(B)
            D[i*6:(i+1)*6] = C.get()
        return se3N(D)
    
    def Phi(self, name):
        if name in ["exp", "Exp"]:
            return self.exp()
        elif name == 'cay':
            return self.cay()
        elif name == "ccsk":
            return self.ccsk()
        else:
            print('Error in coordinate map')
        
    def dPhiinv(self, v, name):
        if name in ["exp", "Exp"]:
            return self.dexpinv(v)
        elif name == 'cay':
            return self.dcayinv(v)
        elif name == "ccsk":
            return self.dccskinv(v)
        else:
            print('Error in differential of coordinate map')
            

class UDQ_2:
    name = "(UDQ)^2"
        
    A = UDQ(1, np.zeros(3), 0, np.zeros(3))
    B = UDQ(1, np.zeros(3), 0, np.zeros(3))
    C = np.asarray([A, B])
    
    def __init__(self, elem):
        self.A = elem[0]
        self.B = elem[1]
        self.C = elem
        
    def get_UDQ2(self):
        return self.C

    def get_UDQs(self):  
        return self.A, self.B
        
    def print(self):
        self.A.print()
        self.B.print()
    
    def identity(self):
        return UDQ_2([self.A.identity(), self.A.identity()])
    
    
class udq_2:
    q = np.zeros(12)
    
    def __init__(self, vec):
        self.q = vec
        
    def set_vec(self, vec):
        self.q = vec
        
    def get_vec(self):
        return self.q
    
    def get_udq_elements(self):
        return np.split(self.q, 2)
    
    def get_Lie_group_id(self):
        A = udq(np.zeros(6))
        return UDQ_2([A.get_Lie_group_id(), A.get_Lie_group_id()])
    
    def print(self): #!!!
        print("fix:", self.q)
        
    def __add__(self, other):
        vec = self.get_vec() + other.get_vec()
        return udq_2(vec)
    
    def mult_w_scalar(self, a):
        return udq_2(a * self.get_vec())
    
    def exp(self):  
        A, B = self.get_udq_elements()
        A, B = udq(A), udq(B)
        E1, E2 = A.exp(), B.exp()
        return UDQ_2([E1, E2])
    
    def dexpinv(self, V): 
        A, B = self.get_udq_elements()
        X, Y = V.get_udq_elements()
        A, B, X, Y = udq(A), udq(B), udq(X), udq(Y)
        C, D = A.dexpinv(X), B.dexpinv(Y)
        return udq_2(np.concatenate((C.get_vec(), D.get_vec())))
    
    def ccsk(self):  
        A, B = self.get_udq_elements()
        A, B = udq(A), udq(B)
        E1, E2 = A.ccsk(), B.ccsk()
        return UDQ_2([E1, E2])
    
    def dccskinv(self, V): 
        A, B = self.get_udq_elements()
        X, Y = V.get_udq_elements()
        A, B, X, Y = udq(A), udq(B), udq(X), udq(Y)
        C, D = A.dccskinv(X), B.dccskinv(Y)
        return udq_2(np.concatenate((C.get_vec(), D.get_vec())))
    
    def Phi(self, name):
        if name in ["exp", "Exp"]:
            return self.exp()
        elif name == "ccsk":
            return self.ccsk()
        else:
            print('Error in coordinate map')
        
    def dPhiinv(self, v, name):
        if name in ["exp", "Exp"]:
            return self.dexpinv(v)
        elif name == "ccsk":
            return self.dccskinv(v)
        else:
            print('Error in differential of coordinate map')

class UDQ_N:
    name = "(UDQ)^N"
    
    def __init__(self, elem):
        self.N = len(elem)
        self.UDQ_elements = elem
    
    def get_elements(self):
        return self.UDQ_elements
    
    def get_N(self):
        return self.N
    
    def print(self):
        for elem in self.UDQ_elements:
            elem.print()

class udq_N:
    def __init__(self, x):
        self.N = int(len(x)/6)
        self.udq_elements = x
    
    def get_udq_elements(self):
        return np.split(self.udq_elements, self.N)
    
    def get_udq_elem_i(self, i):
        return self.udq_elements[i*6:(i+1)*6]
        
    def get_vec(self):
        return self.udq_elements
        
    def get_Lie_group_id(self): 
        A = udq(np.zeros(6))
        B = [A.get_Lie_group_id() for i in range(self.N)]
        return UDQ_N(B)
        
    def print(self):
        elements = self.udq_elements
        for i in range(self.N):
            u = elements[i*6:(i+1)*6]
            print(u)
        
    def __add__(self, other): 
        return udq_N(self.get_vec() + other.get_vec())
    
    def mult_w_scalar(self, a):  
        return udq_N(a * self.get_vec())
    
    def exp(self):  
        E = []
        for i in range(self.N):
            A = self.get_udq_elem_i(i)
            A = udq(A)
            A = A.exp()
            E.append(A)
        return UDQ_N(E)
    
    def dexpinv(self, V):
        D = np.zeros(6*self.N)
        for i in range(self.N):
            A = self.get_udq_elem_i(i)
            B = V.get_udq_elem_i(i)
            A, B = udq(A), udq(B)
            C = A.dexpinv(B)
            D[i*6:(i+1)*6] = C.get_vec()
        return udq_N(D)
    
    def ccsk(self):  
        E = []
        for i in range(self.N):
            A = self.get_udq_elem_i(i)
            A = udq(A)
            A = A.ccsk()
            E.append(A)
        return UDQ_N(E)
    
    def dccskinv(self, V):
        D = np.zeros(6*self.N)
        for i in range(self.N):
            A = self.get_udq_elem_i(i)
            B = V.get_udq_elem_i(i)
            A, B = udq(A), udq(B)
            C = A.dccskinv(B)
            D[i*6:(i+1)*6] = C.get_vec()
        return udq_N(D)
    
    def Phi(self, name):
        if name in ["exp", "Exp"]:
            return self.exp()
        elif name == "ccsk":
            return self.ccsk()
        else:
            print('Error in coordinate map')
        
    def dPhiinv(self, v, name):
        if name in ["exp", "Exp"]:
            return self.dexpinv(v)
        elif name == "ccsk":
            return self.dccskinv(v)
        else:
            print('Error in differential of coordinate map')
