# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from physics-problems import FRB, Pendulum1Fold, Pendulum2Fold, PendulumNFold
from RKMK-methods import RKMK
from various-functions import find_folder

directory = "C:/Users/Maja/OneDrive - NTNU/Documents/NTNU/V2022/Masteroppgave/Figurer//"


# Plots angular momentum for FRB as a function of number of steps
def plot_frb_momentum(method, N, tf, save=True): 
    Prob = FRB("SO3")
    m0 = Prob.get_init()
    l0 = np.linalg.norm(m0) 
    plt.figure(figsize=(8,3))
    for lie_group in ["SO3", "Sp1"]:
        Prob.set_lie_group(lie_group) 
        for coord_map in ["exp", "ccsk", "cay"]: 
            Sol = RKMK(Prob, coord_map, N, tf)
            Sol.numerical_approximation(method)     
            m = Sol.get_solution()
            norm = np.multiply(m,m) 
            norm = norm.sum(axis = 1)
            norm = np.sqrt(norm)
            error = l0 - norm 
            plt.plot(error, label = Sol.coordinate_map 
                       + " " + Prob.Lie_group_name)

    probName = Prob.name
    
    plt.title(Sol.method)
    plt.xlabel("Number of steps")
    plt.ylabel("Error")
    #plt.legend(loc = "upper left")
    plt.legend()    
    plotname = Sol.make_plot_name("Vector length")
    print(plotname)
    folder = find_folder(probName)
    plt.savefig(directory + folder + plotname + ".pdf", bbox_inches = "tight")
    
    plt.show()


# plots q1^Tq1 for the 1-fold pendulum as a function of number of steps
def plot_norm_of_q_1_fold_pendulum(Sol):
    y0 = Sol.Problem.get_init()
    q0, w0 = np.split(y0, 2)
    
    # numerical solution
    y = Sol.get_solution()
    q, w = np.hsplit(y, 2)
    norm = np.multiply(q, q) 
    norm = norm.sum(axis = 1)
    plt.plot(1 - norm)
    
    plotname = Sol.make_plot_name("qTq")
    #print(plotname)
    plt.savefig(directory + "Pendulum1fold/" + plotname)
    plt.show()
    

def plot_q_w_1_fold_pendulum(Sol):
    y0 = Sol.Problem.get_init()
    q0, w0 = np.split(y0, 2)
    
    # numerical solution
    y = Sol.get_solution()
    q, w = np.hsplit(y, 2)
    qTw = np.multiply(q, w) 
    qTw = qTw.sum(axis = 1)
    plt.plot(qTw)
    
    plotname = Sol.make_plot_name("qTw")
    #print(plotname)
    plt.savefig(directory + "Pendulum1fold/" + plotname)
    plt.show()
    plt.show()
   

# plots qi^Tqi for the 2-fold pendulum as a function of number of steps
def plot_qTq_2fold(method, N, i, save = True):
    Prob = Pendulum2Fold("se3^2")
    Prob.set_init_1(np.asarray([0,1,0,1,0,1]))
    Prob.set_init_2(np.asarray([0,1,0,1,0,1]))
    
    j = i-1

    plt.figure(figsize=(8,3))
    for coord_map in ["exp", "ccsk", "cay"]:
        Sol = RKMK(Prob, coord_map, N)
        Sol.numerical_approximation(method)  
        y = Sol.get_solution()
        
        q = y[:,j*6:j*6+3]
        #print(q)
        norm = np.multiply(q, q) 
        norm = norm.sum(axis = 1)
        plt.plot(1-norm, label = Sol.coordinate_map + " " + Prob.Lie_group_name)
        
    
    Prob.set_lie_group("UDQ^2")
    Name = "SP(1)^2"
    
    for coord_map in ["exp", "ccsk"]:
        Sol = RKMK(Prob, coord_map, N)
        Sol.numerical_approximation(method)  
        y = Sol.get_solution()
        
        q = y[:,j*6:j*6+3]
        norm = np.multiply(q, q) 
        norm = norm.sum(axis = 1)
        plt.plot(1-norm, label = Sol.coordinate_map + " " + Name)

    plt.title(Sol.method)
    plt.xlabel("Number of steps")
    plt.ylabel("Error")
    #plt.legend()
    plt.legend(loc = "upper left")

    if save:
        plotname = Sol.make_plot_name("qTq (alt y0)")
        print(plotname)
        plt.savefig(directory + "Pendulum2fold/" + plotname + 
                    ", i=" + str(i) + ".pdf", bbox_inches = "tight")
    plt.show()

    
# plots qi^Tii for the 2-fold pendulum as a function of number of steps 
def plot_qTw_2fold(method, N, i, save = True):
    Prob = Pendulum2Fold("se3^2")
    Prob.set_init_1(np.asarray([0,1,0,1,0,1]))
    Prob.set_init_2(np.asarray([0,1,0,1,0,1]))
    
    j = i-1
    
    plt.figure(figsize=(8,3))
    for coord_map in ["exp", "ccsk", "cay"]:
        Sol = RKMK(Prob, coord_map, N)
        Sol.numerical_approximation(method)  
        y = Sol.get_solution()
        
        q = y[:,j*6:j*6+3]
        w = y[:,j*6+3:i*6]
        #print(q)
        norm = np.multiply(q, w) 
        norm = norm.sum(axis = 1)
        plt.plot(norm, label = Sol.coordinate_map + Prob.Lie_group_name)
    
    Prob.set_lie_group("udq^2")
    print("over halvveis!")
    Name = "SP(1)^2"
    
    for coord_map in ["exp", "ccsk"]:
        Sol = RKMK(Prob, coord_map, N)
        Sol.numerical_approximation(method)  
        y = Sol.get_solution()
        
        q = y[:,j*6:j*6+3]
        w = y[:,j*6+3:i*6]
        #print(q)
        norm = np.multiply(q, w) 
        norm = norm.sum(axis = 1)
        plt.plot(norm, label = Sol.coordinate_map + Name)

    plt.title(Sol.method)
    plt.xlabel("Number of steps")
    plt.ylabel("Error")
    #plt.legend()
    plt.legend(loc = "lower left")
    
    if save:    
        plotname = Sol.make_plot_name("qTw")
        #print(plotname)
        plt.savefig(directory + "Pendulum2fold/" + plotname + 
                    ", i=" + str(i) + ".pdf", bbox_inches = "tight")
    plt.show()



