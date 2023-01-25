def scatterGazeInSteps(gazeFile, step):
    """
    Take 'gazeFile' sequence, and cut into windows of 'step' length across the range, and
    then scatter them sequentially on the same plot to visualise them.
    """
    rangelist = list(range(0, gazeFile.shape[0], step)) 
    #generate start and stop points for the length of the file

    starts = rangelist[:-2] #index to stop it being out of bounds
    stops = rangelist[1:] # index to stop it being out of bounds
    
    for start, stop in zip(starts, stops): # zip start and stop together and scatter
        print(start, stop)
        plt.scatter(gazeFile[:,0][start:stop], gazeFile[:,1][start:stop])     

def writeHistoryVariable(name):
    tempmodelHistory[name] = history[name]
    return tempmodelHistory[name]