# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from Master.RKMK import RKMK
from Master.Problemer import FRB, Pendulum1Fold, Pendulum2Fold, PendulumNFold
from Master.Hjelpefunksjoner import identify_order, find_folder

directory = "C:/Users/Maja/OneDrive - NTNU/Documents/NTNU/V2022/Masteroppgave/Figurer//"



def test_step_length(Prob, coord_map, method, k0, k, save = True):
    ### TODO ###
    M = k + 1 - k0
    error = np.zeros(M)
    h_list = np.zeros(M)
    
    Nref = 10* (2**k)
    Ref = RKMK(Prob, coord_map, Nref, 5)
    Ref.numerical_approximation(method)    
    y = Ref.get_solution()
    y_ref = y[-1,:]
    
    klist = np.arange(k0, k+1)
    #print(klist)
    
    for i in range(M):
        print(klist[i])
        N = 2**klist[i]
        Sol = RKMK(Prob, coord_map, N, 5)
        Sol.numerical_approximation(method)     
        y = Sol.get_solution()
        end_point = y[-1,:]
        error[i] = np.linalg.norm(y_ref - end_point)
        h_list[i] = Sol.tf/N
    
    x = np.linspace(h_list[0], h_list[-1], k)
    
    plt.loglog(h_list, error, label = "Error")
    #x = np.asarray([i for i in range(10)])
    n = identify_order(method)
    #plt.loglog(x, x**2, linestyle = "dotted", label = "y = x^2")
    #plt.loglog(x, x**3, linestyle = "dotted", label = "y = x^3")
    plt.loglog(x, x**n, linestyle = "dotted", label = "y = x^%i" % n)
    plt.title("Global error with respect to step size")
    plt.xlabel("Time step h")
    plt.ylabel("Error")
    plt.legend()

    if save:    
        plotname = Sol.make_order_plot_name(2**k0, 2**k, Nref)
    
        probName = Prob.name
        folder = find_folder(probName)
        print(plotname)
        plt.savefig(directory + folder + plotname + ".pdf")
    plt.show()


def compare_coordinate_maps_and_lie_groups(Prob, lie_groups, coord_maps, method, k0, k, save=True):
    # note that cay does not work for dual kvaternions 
    
    M = k + 1 - k0
    error = np.zeros(M)
    h_list = np.zeros(M)
    
    Nref = 10* (2**k)
    Ref = RKMK(Prob, "exp", Nref, 5) # which for reference?
    Ref.numerical_approximation("rkmk4")     
    y = Ref.get_solution()
    y_ref = y[-1,:]
    
    klist = np.arange(k0, k+1)
    #print(klist)
    
    plt.figure(figsize=(8,4))
    for lie_group in lie_groups:
        #print(lie_group)
        Prob.set_lie_group(lie_group) 
        for coord_map in coord_maps: 
            for i in range(M):
                N = 2**klist[i]
                Sol = RKMK(Prob, coord_map, N, 5)
                Sol.numerical_approximation(method)     
                y = Sol.get_solution()
                end_point = y[-1,:]
                error[i] = np.linalg.norm(y_ref - end_point)
                h_list[i] = Sol.tf/N
            plt.loglog(h_list, error, label = Sol.coordinate_map 
                       + " " + Prob.Lie_group_name)
        
    n = identify_order(method)
    x = np.linspace(h_list[0], h_list[-1], k)
    plt.loglog(x, x**n, linestyle = "dotted", label = "y = x^%i" % n)
    probName = Prob.name
    
    plt.title(Sol.method)
    plt.xlabel("Time step")
    plt.ylabel("Global error")
    plt.legend(loc = "lower right")
    
    if save:
        plotname = Sol.make_comparison_plot_name("Compare Phi and G", 2**k0, 2**k, Nref)
        print(plotname)
        folder = find_folder(probName)
        plt.savefig(directory + folder + plotname + ".pdf", bbox_inches = "tight")
    
    plt.show()
    return 0

#Prob = FRB("SO3")
#compare_coordinate_maps_and_lie_groups(Prob, ["SO3", "Sp1"], ["exp", "ccsk", "cay"], 
#                                       "rkmk4", 2, 14, True)


def compare_some_things_pendulum(N, method, k0, k, save = True):
    # note that cay does not work for dual kvaternions 
    
    if N == 1:
        Prob = Pendulum1Fold("SE3")
    elif N == 2:
        Prob = Pendulum2Fold("SE3^2")
    else:
        Prob = PendulumNFold("SE3^N", N)  # OBS: høy N betyr høyere k0 !!!
    
    M = k + 1 - k0
    error = np.zeros(M)
    h_list = np.zeros(M)
    
    Nref = 10* (2**k)
    Ref = RKMK(Prob, "exp", Nref, 5) # which for reference?
    Ref.numerical_approximation("rkmk4")     
    y = Ref.get_solution()
    y_ref = y[-1,:]
    
    klist = np.arange(k0, k+1)
    plt.figure(figsize=(8,4))
    for coord_map in ["exp", "ccsk", "cay"]: 
        print(coord_map)
        for i in range(M):
            #print(i)
            Ni = 2**klist[i]
            Sol = RKMK(Prob, coord_map, Ni, 5)
            Sol.numerical_approximation(method)     
            y = Sol.get_solution()
            #print(1)
            end_point = y[-1,:]
            error[i] = np.linalg.norm(y_ref - end_point)
            h_list[i] = Sol.tf/Ni
        #x = np.linspace(h_list[0], h_list[-1], k)
        plt.loglog(h_list, error, label = Sol.coordinate_map 
                   + " " + Prob.Lie_group_name)
    if N == 1:
        Prob = Pendulum1Fold("UDQ")
    elif N == 2:
        Prob = Pendulum2Fold("UDQ^2")
    else:
        Prob = PendulumNFold("UDQ^N", N) 
        
    Name = "SP(1)^2"
    for coord_map in ["exp", "ccsk"]: 
        print(coord_map)
        for i in range(M):
            #print(i)
            Ni = 2**klist[i]
            Sol = RKMK(Prob, coord_map, Ni, 5)
            Sol.numerical_approximation(method)     
            y = Sol.get_solution()
            #print(2)
            end_point = y[-1,:]
            error[i] = np.linalg.norm(y_ref - end_point)
            h_list[i] = Sol.tf/Ni
        #x = np.linspace(h_list[0], h_list[-1], k)
        plt.loglog(h_list, error, label = Sol.coordinate_map 
                    + " " + Name)
    
    #x = np.asarray([i for i in range(10)])
    x = np.linspace(h_list[0], h_list[-1], k) 
    n = identify_order(method)
    plt.loglog(x, x**n, linestyle = "dotted", label = "y = x^%i" % n)
    probName = Prob.name
    
    #plt.title("Global error for " + probName + " with different \ncoordinate " +
    #          "mappings and Lie groups, using " + Sol.method)
    plt.title(Sol.method)
    plt.xlabel("Time step")
    plt.ylabel("Global error")
    plt.legend()

    if save:
        plotname = Sol.make_comparison_plot_name("Compare Phi and G some", 2**k0, 2**k, Nref)
        print(plotname)
        folder = find_folder(probName)
        plt.savefig(directory + folder + plotname + ".pdf", bbox_inches = "tight")
    
    plt.show()

# compare_some_things_pendulum(2, "le", 5, 14, True)
























def test_step_length_SYM(Prob, method, k0, k, save = True):
    ### TODO ###
    M = k + 1 - k0
    error = np.zeros(M)
    h_list = np.zeros(M)
    
    Nref = 10* (2**k)
    Ref = SYM(Prob, Nref, 5)
    Ref.numerical_approximation(method)    
    y = Ref.get_solution()
    y_ref = y[-1,:]
    
    klist = np.arange(k0, k+1)
    #print(klist)
    
    for i in range(M):
        print(klist[i])
        N = 2**klist[i]
        Sol = SYM(Prob, N, 5)
        Sol.numerical_approximation(method)     
        y = Sol.get_solution()
        end_point = y[-1,:]
        error[i] = np.linalg.norm(y_ref - end_point)
        h_list[i] = Sol.tf/N
    
    x = np.linspace(h_list[0], h_list[-1], k)
    
    plt.loglog(h_list, error, label = "Error")
    #x = np.asarray([i for i in range(10)])
    n = identify_order(method)
    #plt.loglog(x, x**2, linestyle = "dotted", label = "y = x^2")
    #plt.loglog(x, x**3, linestyle = "dotted", label = "y = x^3")
    plt.loglog(x, x**n, linestyle = "dotted", label = "y = x^%i" % n)
    plt.title("Global error with respect to step size")
    plt.xlabel("Time step h")
    plt.ylabel("Error")
    plt.legend()

    if save:    
        plotname = Sol.make_order_plot_name(2**k0, 2**k, Nref)
    
        probName = Prob.name
        folder = find_folder(probName)
        print(plotname)
        plt.savefig(directory + folder + plotname + ".pdf")
    plt.show()
    
#Prob = FRB("SO3")
#Prob = FRB("Sp1")
#Prob = HeavyTop("SE3")
#Prob = Pendulum1Fold("SE3")
#Prob = Pendulum1Fold("UDQ")
#Prob = Pendulum2Fold("SE3^2")
#Prob = Pendulum2Fold("UDQ^2")
#Prob = PendulumNFold("SE3^N", 4)  # OBS: høy N betyr høyere k0 !!!
#Prob = PendulumNFold("UDQ^N", 4)
#test_step_length_SYM(Prob, "rkmk4", 2, 12, False)




# what does thid plot even mean? Why did I make it?
def compare_SYM(Prob, method, k0, k, save = True):
    #Assumes lie_groups contain two elements
    
    M = k + 1 - k0
    error = np.zeros(M)
    h_list = np.zeros(M)
    
    klist = np.arange(k0, k+1)
    
    for i in range(M):
        N = 2**klist[i]
        Sol1 = RKMK(Prob, "ccsk", N, 5)
        Sol1.numerical_approximation(method)     
        y1 = Sol1.get_solution()
        end_point1 = y1[-1,:]
        Sol2 = SYM(Prob, N, 5)
        Sol2.numerical_approximation(method)     
        y2 = Sol2.get_solution()
        end_point2 = y2[-1,:]
        error[i] = np.linalg.norm(end_point1 - end_point2)
        h_list[i] = Sol1.tf/N
    plt.plot(h_list, error, label = "Error")
    plt.xscale("log")
    
    probName = Prob.name
    plt.title("Difference between STUFF " + probName + "\n")
    plt.xlabel("Time step h")
    plt.ylabel("Error")
    plt.legend()
    
    if save:
        plotname = Sol1.make_comparison_plot_name("Compare SYM", 2**k0, 2**k, "None")
        print(plotname)
        folder = find_folder(probName)
        plt.savefig(directory + folder + plotname + ".pdf")
    plt.show()
#Prob = FRB("SO3")
#compare_SYM(Prob, "rkmk4", 6, 12)


def compare_order_SYM(Prob, method, k0, k, save = True):
    ### TODO ###
    M = k + 1 - k0
    error = np.zeros(M)
    h_list = np.zeros(M)
    
    Nref = 10* (2**k)
    Ref = SYM(Prob, Nref, 5)
    #Ref = RKMK(Prob, "ccsk", Nref, 5)
    Ref.numerical_approximation(method)    
    y = Ref.get_solution()
    y_ref = y[-1,:]
    
    klist = np.arange(k0, k+1)
    #print(klist)
    
    for i in range(M):
        print(klist[i])
        N = 2**klist[i]
        Sol = RKMK(Prob, "ccsk", N, 5)
        Sol.numerical_approximation(method)     
        y = Sol.get_solution()
        end_point = y[-1,:]
        error[i] = np.linalg.norm(y_ref - end_point)
        h_list[i] = Sol.tf/N
    
    x = np.linspace(h_list[0], h_list[-1], k)
    plt.loglog(h_list, error, label = "Error")
    
    for i in range(M):
        print(klist[i])
        N = 2**klist[i]
        Sol = SYM(Prob, N, 5)
        Sol.numerical_approximation(method)     
        y = Sol.get_solution()
        end_point = y[-1,:]
        error[i] = np.linalg.norm(y_ref - end_point)
        h_list[i] = Sol.tf/N
    
    x = np.linspace(h_list[0], h_list[-1], k)
    
    plt.loglog(h_list, error, label = "Error SYM")
    #x = np.asarray([i for i in range(10)])
    n = identify_order(method)
    #plt.loglog(x, x**2, linestyle = "dotted", label = "y = x^2")
    #plt.loglog(x, x**3, linestyle = "dotted", label = "y = x^3")
    plt.loglog(x, x**n, linestyle = "dotted", label = "y = x^%i" % n)
    plt.title("Global error with respect to step size")
    plt.xlabel("Time step h")
    plt.ylabel("Error")
    plt.legend()

    if save:    
        plotname = Sol.make_order_plot_name(2**k0, 2**k, Nref)
    
        probName = Prob.name
        folder = find_folder(probName)
        print(plotname)
        plt.savefig(directory + folder + "SYM (alt)" + plotname + ".pdf")
    plt.show()
    
#Prob = FRB("SO3")
#compare_order_SYM(Prob, "rkmk4", 2, 14)# -*- coding: utf-8 -*-

