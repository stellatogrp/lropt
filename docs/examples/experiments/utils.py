import numpy as np
import matplotlib.pyplot as plt

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

def plot_iters(dftrain, title, steps = 2000):
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


