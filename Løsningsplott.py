# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from Master.Problemer import FRB, Pendulum1Fold, Pendulum2Fold, PendulumNFold
from Master.RKMK import RKMK
from Master.Hjelpefunksjoner import find_folder

directory = "C:/Users/Maja/OneDrive - NTNU/Documents/NTNU/V2022/Masteroppgave/Figurer//"

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



def plot_1_fold_pendulum(Sol, plot_name = None):
    y0 = Sol.Problem.get_init()
    q0, w0 = np.split(y0, 2)
    r0 = np.linalg.norm(q0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # initial solution
    ax.scatter(q0[0], q0[1], -q0[2])
    
    # numerical solution
    y = Sol.get_solution()
    q, w = np.hsplit(y, 2)
    x,y,z = np.transpose(q)
    ax.plot(x,y,-z)
    
    # limit the axes (in case of diverging solution)
    x = [q0 for i in range(2)]
    ax.set_box_aspect((np.ptp(x), np.ptp(x), np.ptp(x)))
    limit = 5/4*r0
    ax.set_xlim3d(-limit, limit)
    ax.set_ylim3d(-limit, limit)
    ax.set_zlim3d(-limit, limit)
    
    plotname = Sol.make_plot_name("Numerical solution")
    #print(plotname)
    plt.savefig(directory + "Pendulum1fold/" + plotname)
    plt.show()
    

def plot_2_fold_pendulum(Sol, plot_name = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial solution
    y0 = Sol.Problem.get_init()
    q01, w01, q02, w02 = np.split(y0, 4)

    # Numerical solution
    y = Sol.get_solution()
    q1, w1, q2, w2 = np.hsplit(y, 4)
    
    # Find positions wrt the same point
    x01 = q01
    x02 = x01 + q02
    x1 = q1
    x2 = x1 + q2
    
    # Plot initial solutions
    ax.scatter(x01[0], x01[1], -x01[2])
    ax.scatter(x02[0], x02[1], -x02[2])
    
    # Plot numerical solution
    x,y,z = np.transpose(x1)
    ax.plot(x,y,-z)
    x,y,z = np.transpose(x2)
    ax.plot(x,y,-z)
    
    plotname = Sol.make_plot_name("Numerical solution")
    #print(plotname)
    plt.savefig(directory + "Pendulum2fold/" + plotname)
    plt.show()
    plt.show()


def plot_N_fold_pendulum(Sol, plot_name = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    N = Sol.Problem.N
    #print(N)

    # Initial solution
    y0 = Sol.Problem.get_init()
    #print(y0)
    x0 = np.zeros(3)
    for i in range(N):
        q0i = y0[i*6:i*6+3]
        x0 += q0i
        ax.scatter(x0[0], x0[1], -x0[2])
        
    # Numerical solution
    y = Sol.get_solution()
    #print(y)
    x = np.zeros((len(y),3))
    #print(x)
    for i in range(N): 
        #print(i)
        q = y[:,6*i:6*i+3]
        #print(q)
        x = x + q
        a, b, c = np.transpose(x)
        ax.plot(a, b, -c)
    
    plotname = Sol.make_plot_name("Numerical solution")
    #print(plotname)
    plt.savefig(directory + "PendulumNfold/" + plotname)
    plt.show()
    



########### THE NICE WAY TO PLOT THE PENDULUM SOLUTION ##########
def plot_2F_pendulum(save = True):
    ### 2-FOLD PENDULUM ###
    
    # SOLVE NUMERICAL PROBLEM
    N = 500
    Prob = Pendulum2Fold("SE3^2")
    #Prob.set_init(np.asarray([1/np.sqrt(2),0,1/np.sqrt(2),0,1,0]))
    #Prob.set_init(np.asarray([1/np.sqrt(2),0,1/np.sqrt(2),0.1,0,-0.1]))
    Sol = RKMK(Prob, "exp", N, 5)
    Sol.numerical_approximation("rkmk4")
    
    # INITIAL SOLUTION
    y0 = Prob.get_init()
    #r0 = np.linalg.norm(y0)
    #print(y0)
    y01 = y0[0:3]
    #print(y01)
    y02 = y0[6:9] + y01
    #print(y02)
    
    # NUMERICAL SOLUTION
    numerical_solution = Sol.get_solution()
    
    # THE DATA POINTS
    q1, w1, q2, w2 = np.hsplit(numerical_solution, 4)
    
    # End points 
    yf1 = q1[-1]
    yf2 = q2[-1] + yf1
    
    # Plotting the figure
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


def plot_3F_pendulum(save = True):
    ### 3-FOLD PENDULUM ###
    
    # SOLVE NUMERICAL PROBLEM
    N = 1000
    Prob = PendulumNFold("SE3^N", 3)
    #Prob.set_init_1([1,0,0,0,0,0])
    #Prob.set_init_2([1,0,0,0,0,0])
    Sol = RKMK(Prob, "exp", N, 5)
    Sol.numerical_approximation("rkmk4")
    
    # INITIAL SOLUTION
    y0 = Prob.get_init()
    y01 = y0[0:3]
    y02 = y0[6:9] + y01
    y02 = y0[6:9] + y01
    
    # NUMERICAL SOLUTION
    numerical_solution = Sol.get_solution()
    t = Sol.get_time()

    # THE DATA POINTS
    q1, w1, q2, w2 = np.hsplit(numerical_solution, 4)
    
    # End points 
    yf1 = q1[-1]
    yf2 = q2[-1] + yf1
    
    # Plotting the figure
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
    ax.legend()
    
    if save:
        plotname = Sol.make_plot_name("Numerical solution")
        print(plotname)
        plt.savefig(directory + "Pendulum2fold/" + plotname + ".pdf")
    plt.show()



