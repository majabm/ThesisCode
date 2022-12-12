# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as sl
from math import sin, cos, tan, sqrt
from mpmath import cot

from Master.Liegrupper import SO3, so3, Sp1, sp1, SE3, se3, UDQ, udq, SE3_Squared, se3_Squared, UDQ_2, udq_2, SE3N, se3N, UDQ_N, udq_N


def identify_lie_group(name):
    if name in ["SO(3)", "SO3", "so3", "so(3)", "So3", "So(3)"]:
        return SO3, so3
    if name in ["Sp(1)", "Sp1", "sp(1)", "sp1", "Quaternion", "quaternion",
                "Quaternions", "quaternions", "Quat", "quat"]:
        return Sp1, sp1
    if name in ["SE(3)", "SE3", "se(3)", "se3", "Se(3)", "Se3"]:
        return SE3, se3
    if name in ["Sp(1)^", "Sp1^", "sp(1)^", "sp1^", "Dual quaternion", 
                "dual quaternion", "dual Quaternion", "Dual Duaternion",
                "Dual Quaternions", "dual quaternions", "dual Quaternions", 
                "Dual quaternions", "Dual", "dual", "dq", "DQ", "UDQ", "udq"
                "Udq", "DUQ", "duq", "Duq", "udq", "UDQ", "Udq"]:
        return UDQ, udq
    if name in ["SE(3)^2", "SE3^2", "se(3)^2", "se3^2", "Se(3)^2", "Se3^2",
                "SE(3)2", "SE32", "se(3)2", "se32", "Se(3)2", "Se32",
                "SE(3)-2", "SE3-2", "se(3)-2", "se3-2", "Se(3)-2", "Se3-2"]:
        return SE3_Squared, se3_Squared
    if name in ["SE(3)^N", "SE3^N", "se(3)^N", "se3^N", "Se(3)^N", "Se3^N",
                "SE(3)N", "SE3N", "se(3)N", "se3N", "Se(3)N", "Se3N",
                "SE(3)-N", "SE3-N", "se(3)-N", "se3-N", "Se(3)-N", "Se3-N"]:
        return SE3N, se3N
    if name in ["UDQ2", "udq2", "Udq2", "DUQ2", "duq2", "Duq2", "udq2", "UDQ2", 
                "Udq2", "UDQ^2", "udq^2", "Udq^2", "UDQ_2", "udq_2"]:
        return UDQ_2, udq_2
    if name in ["UDQN", "udqN", "UdqN", "DUQN", "duqN", "DuqN", "udqN", "UDQN", 
                "UdqN", "UDQ^N", "udq^N", "Udq^N", "UDQ_N", "udq_N"]:
        return UDQ_N, udq_N
    else: print("Error: Cannot identyfy Lie group")
    
def identify_order(name):
    if name in ["lie_euler", "Lie_Euler", "Lie_euler", "lie_Euler",
                "lie-euler", "Lie-Euler", "Lie-euler", "lie-Euler",
                "Lie Euler", "Lie euler", "lie Euler", "lie euler",
                "LieEuler", "Lieeuler", "lieEuler", "lieeuler",
                "le", "LE", "Le", "lE"]:
        return 1
    if name in ["improved lie euler", "improved Lie Euler",
                "impr lie euler", "impr Lie Euler",
                "impr", "Impr", "ile", "iLE", "Heun Euler"]:
        return 2
    if name in ["rkmk4", "mkrk4", "MKRK4", "RKMK4", "mk4", "MK4, rkmk, mkrk"]:
        return 4
    else:
        print("Error: Method identification (order)")
        
def find_folder(name):
    if name == "Euler's free rigid body":
        return "FRB/"
    elif name == "Heavy top":
        return "HeavyTop/"
    elif name == "1-fold 3D pendulum":
        return "Pendulum1fold/"
    elif name == "2-fold 3D pendulum":
        return "Pendulum2fold/"
    elif name == "N-fold 3D pendulum":
        return "PendulumNfold/"

def to_hat_matrix(v):
    U = np.zeros((3,3))
    U[0,1] = -v[2]
    U[0,2] = v[1]
    U[1,2] = -v[0]
    A = U - np.transpose(U)
    return(A)

def from_hat_matrix(A):  
    v = np.zeros(3)
    v[0] = -A[1,2]
    v[1] = A[0,2]
    v[2] = -A[0,1]
    return(v)

def to_sp1_matrix(u):
    u1, u2, u3 = u
    A = np.zeros((4,4))
    A[0,1:] = -u
    A[1,2] = u3
    A[1,3] = -u2
    A[2,3] = u1
    A = A - np.transpose(A)
    return A

def normalize(v):
    norm = np.linalg.norm(v)
    TOL = 1e-9
    if norm < TOL: 
       return v
    return v / norm