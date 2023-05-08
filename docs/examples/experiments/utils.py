import matplotlib.pyplot as plt
import numpy as np


def plot_tradeoff(df_standard,df_reshape,title,ind_1 = (0,100), ind_2 = (0,100)):
    plt.rcParams.update({
    "text.usetex":True,

    "font.size":22,
    "font.family": "serif"
})
    beg1,end1 = ind_1
    beg2,end2 = ind_2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 3))
    ax1.plot(np.mean(np.vstack(df_standard['Violation_val']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Test_val']),axis = 1)[beg1:end1], color="tab:blue", label=r"Standard set")
    ax1.fill(np.append(np.quantile(np.vstack(df_standard['Violation_val']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Violation_val']),0.9,axis = 1)[beg1:end1][::-1]), np.append(np.quantile(np.vstack(df_standard['Test_val']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Test_val']),0.90,axis = 1)[beg1:end1][::-1]), color="tab:blue", alpha=0.2)

    ax1.plot(np.mean(np.vstack(df_reshape['Violation_val']),axis = 1)[beg2:end2],np.mean(np.vstack(df_reshape['Test_val']),axis = 1)[beg2:end2], color = "tab:orange",label=r"Reshaped set")
    ax1.fill(np.append(np.quantile(np.vstack(df_reshape['Violation_val']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Violation_val']),0.9,axis = 1)[beg2:end2][::-1]), np.append(np.quantile(np.vstack(df_reshape['Test_val']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Test_val']),0.90,axis = 1)[beg2:end2][::-1]), color="tab:orange", alpha=0.2)

    ax1.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    ax1.set_xlabel("Empirical $\mathbf{CVaR}$")
    ax1.set_ylabel("Objective value")
    ax1.legend()

    ax2.plot(np.mean(np.vstack(df_standard['Violations']),axis = 1)[beg1:end1], np.mean(np.vstack(df_standard['Test_val']),axis = 1)[beg1:end1], color="tab:blue", label=r"Standard set")
    ax2.fill(np.append(np.quantile(np.vstack(df_standard['Violations']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Violations']),0.9,axis = 1)[beg1:end1][::-1]), np.append(np.quantile(np.vstack(df_standard['Test_val']),0.1,axis = 1)[beg1:end1],np.quantile(np.vstack(df_standard['Test_val']),0.90,axis = 1)[beg1:end1][::-1]), color="tab:blue", alpha=0.2)

    ax2.plot(np.mean(np.vstack(df_reshape['Violations']),axis = 1)[beg2:end2], np.mean(np.vstack(df_reshape['Test_val']),axis = 1)[beg2:end2], color="tab:orange", label=r"Reshaped set")
    ax2.fill(np.append(np.quantile(np.vstack(df_reshape['Violations']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Violations']),0.9,axis = 1)[beg2:end2][::-1]), np.append(np.quantile(np.vstack(df_reshape['Test_val']),0.1,axis = 1)[beg2:end2],np.quantile(np.vstack(df_reshape['Test_val']),0.90,axis = 1)[beg2:end2][::-1]), color="tab:orange", alpha=0.2)
    ax2.set_xlabel("Probability of constraint violation")
    ax2.set_ylabel("Objective value")
    ax2.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    ax2.legend()
    plt.savefig(title+"_curve.pdf",bbox_inches='tight')
    plt.show()

def plot_iters(dftrain, title, steps = 2000, logscale = True):
    plt.rcParams.update({
    "text.usetex":True,

    "font.size":22,
    "font.family": "serif"
})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 3))

    ax1.plot(dftrain["Violation_val"][:steps], label = "Out-of-sample empirical CVaR")
    ax1.plot(dftrain["Violation_train"][:steps], label = "In-sample empirical CVaR",linestyle = "--")

    ax1.set_xlabel("Iterations")
    ax1.hlines(xmin = 0, xmax = dftrain["Violation_val"][:steps].shape[0], y = -0.0, linestyles = "--",color = "black", label = "Target threshold: 0")
    ax1.legend()
    ax2.plot(dftrain["Test_val"][:steps], label = "Objective value")
    ax2.set_xlabel("Iterations")
    ax2.ticklabel_format(style="sci",axis='y',scilimits = (0,0), useMathText=True)
    ax2.legend()
    if logscale:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    plt.savefig(title+"_iters.pdf",bbox_inches='tight')


def pareto_frontier(Xs, Ys, maxX = False, maxY = False):
    Xs = np.array(Xs)
    Ys = np.array(Ys)
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

def plot_contours(x,y,set, g_level,eps_list, inds, num_scenarios,train, title, lower = -10, upper = 25, diff = 5, standard = True):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout= True)
    ax_lst = [ax1, ax2, ax3, ax4]

    cur_ind = 0
    contours = []
    for axis in ax_lst:
        axis.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[cur_ind]],2)))
        axis.set_xlabel(r"$u_1$")
        axis.set_ylabel(r"$u_2$")
        for k_ind in range(1):
            axis.contour(x,y,set[cur_ind][k_ind], [1], colors = ["red"],linewidths = [2])

        a1 = axis.contourf(x,y,np.zeros((50,50)),np.arange(lower, upper, diff),extend='both',alpha = 1)
        contours.append(a1)
        for scene in range(num_scenarios):
            a1 = axis.contourf(x,y,g_level[cur_ind][scene],np.arange(lower, upper, diff),extend='both',alpha = 0.4)
            contours.append(a1)
        axis.scatter(train[:,0],train[:,1], color = "white" ,edgecolors= "black")
        axis.scatter(np.mean(train,axis= 0)[0], np.mean(train,axis= 0)[1],color = ["tab:orange"])
        cur_ind +=1
        fig.colorbar(contours[0], ax = axis)

    if standard:
        post = "Standard"
    else:
        post = "Reshaped"
    fig.suptitle(post+" set", fontsize = 30)
    plt.savefig(title+"_"+ post + ".pdf", bbox_inches='tight')


def plot_contours2(x,y,set, g_level,eps_list, inds, num_scenarios,train, title, lower = -10, upper = 25, diff = 5, standard = True):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout= True)
    ax_lst = [ax1, ax2, ax3, ax4]

    cur_ind = 0
    contours = []
    for axis in ax_lst:
        axis.set_title(r"$\epsilon$ = {}".format(round(eps_list[inds[cur_ind]],2)))
        axis.set_xlabel(r"$u_1$")
        axis.set_ylabel(r"$u_2$")
        for k_ind in range(1):
            axis.contour(x,y,set[cur_ind][k_ind], [1], colors = ["red"],linewidths = [2])

        a1 = axis.contourf(x,y,np.zeros((50,50)),np.arange(lower, upper, diff),extend='both',alpha = 1)
        contours.append(a1)
        for scene in range(num_scenarios):
            a1 = axis.contourf(x,y,g_level[cur_ind][scene],np.arange(lower, upper, diff),extend='both',alpha = 0.4)
            contours.append(a1)
        axis.scatter(train[:,0],train[:,1], color = "white" ,edgecolors= "black")
        axis.scatter(np.mean(train,axis= 0)[0], np.mean(train,axis= 0)[1],color = ["tab:orange"])
        cur_ind +=1
        fig.colorbar(contours[0], ax = axis)

    if standard:
        post = "Standard"
    else:
        post = "Reshaped"
    fig.suptitle(post+" set", fontsize = 30)
    plt.savefig(title+"_"+ post + ".pdf", bbox_inches='tight')

def plot_contours_line(x,y,set, g_level,prob_list, num_scenarios,train, title, standard = True):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout= True)
    ax_lst = [ax1, ax2, ax3, ax4]

    cur_ind = 0
    for axis in ax_lst:
        axis.set_title(r'$\hat{\eta}$' + ' = {}'.format(prob_list[cur_ind]))
        axis.set_xlabel(r"$u_1$")
        axis.set_ylabel(r"$u_2$")
        for scene in range(num_scenarios):
            axis.contour(x,y,g_level[cur_ind][scene],[0], colors = ["tab:purple"],alpha = 1,linestyles = ["-"])
        axis.scatter(train[:,0],train[:,1], color = "white",edgecolor = "black")
        axis.scatter(np.mean(train,axis= 0)[0], np.mean(train,axis= 0)[1],color = ["tab:orange"])
        for k_ind in range(1):
            axis.contour(x,y,set[cur_ind][k_ind], [1], colors = ["red"],linewidths = [2])
        cur_ind +=1
    if standard:
        post = "Standard"
    else:
        post = "Reshaped"
    fig.suptitle(post+" set", fontsize = 30)
    plt.savefig(title+"_"+ post + ".pdf", bbox_inches='tight')
