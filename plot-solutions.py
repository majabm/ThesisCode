# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from physics-problems import FRB, Pendulum1Fold, Pendulum2Fold, PendulumNFold
from RKMK-methods import RKMK
from various-function import find_folder

directory = "C:/Users/Maja/OneDrive - NTNU/Documents/NTNU/V2022/Masteroppgave/Figurer//"


# plots the numerical solution of FRB
def plot_frb(frb, save=True):
    m0 = frb.Problem.get_init()
    r0 = np.linalg.norm(m0)
    
    fig = plt.figure()
    
    # draw sphere
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = r0*np.cos(u)*np.sin(v)
    y = r0*np.sin(u)*np.sin(v)
    z = r0*np.cos(v)
    
    # for wire frame
    ax.plot_wireframe(x, y, z, color="gray", alpha = 0.5)
    ax.set_title("Numerical solution")
    
    # for surface
    ax.plot_surface(x,y,z, color="gray", alpha = 0.2)
    
    # initial solution
    ax.scatter(m0[0], m0[1], m0[2])
    
    # numerical solution
    m = frb.get_solution()
    x,y,z = np.transpose(m)
    ax.plot(x,y,z)
    
    x = [m0 for i in range(2)]
    ax.set_box_aspect((np.ptp(x), np.ptp(x), np.ptp(x)))
    limit = 5/4*r0
    ax.set_xlim3d(-limit, limit)
    ax.set_ylim3d(-limit, limit)
    ax.set_zlim3d(-limit, limit)
    
    plotname = frb.make_plot_name("Numerical solution")
    #print(plotname)
    plt.savefig(directory + "FRB/" + plotname + ".pdf")
    plt.show()


# plots the numerical solution of the 2-fold pendulum
def plot_2F_pendulum(save = True):
    
    # solve numerical problem
    N = 500
    Prob = Pendulum2Fold("SE3^2")
    #Prob.set_init(np.asarray([1/np.sqrt(2),0,1/np.sqrt(2),0,1,0]))
    #Prob.set_init(np.asarray([1/np.sqrt(2),0,1/np.sqrt(2),0.1,0,-0.1]))
    Sol = RKMK(Prob, "exp", N, 5)
    Sol.numerical_approximation("rkmk4")
    
    # initial solution
    y0 = Prob.get_init()
    #print(y0)
    y01 = y0[0:3]
    #print(y01)
    y02 = y0[6:9] + y01
    #print(y02)
    
    # numerical solution
    numerical_solution = Sol.get_solution()
    q1, w1, q2, w2 = np.hsplit(numerical_solution, 4)
    
    # end points
    yf1 = q1[-1]
    yf2 = q2[-1] + yf1
    
    # making the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # initial solution
    ax.scatter(y01[0], y01[1], -y01[2])
    ax.scatter(y02[0], y02[1], -y02[2])
    
    # numerical solution
    x,y,z = np.transpose(q1)
    ax.plot(x,y,-z) # label???
    x,y,z = np.transpose(q1+q2)
    ax.plot(x,y,-z)
    
    # final solution
    ax.scatter(yf1[0], yf1[1], -yf1[2], color = "black")
    ax.scatter(yf2[0], yf2[1], -yf2[2], color = "black")
    
    # line to final positioning?
    ax.plot([0,yf1[0]], [0,yf1[1]], [0,-yf1[2]], color = "black")
    ax.plot([yf1[0],yf2[0]], [yf1[1],yf2[1]], [-yf1[2],-yf2[2]], color = "black")
    
    # Adding Figure Labels
    ax.set_title('Numerical solution of 2-fold pendulum')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.legend()
    
    if save:
        plotname = Sol.make_plot_name("Numerical solution")
        print(plotname)
        plt.savefig(directory + "Pendulum2fold/" + plotname + ".pdf")
    plt.show()
