## These are general .py programs written to be used across all programs ##
import numpy as np

def nameAsString(string):
    for k, v in list(locals().iteritems()):
         if v is string:
            name_as_str = k
    return name_as_str
def splitUpString(string,delimiter=''):
    if delimiter is '':
        split_array = np.array(list(map(float,string.split())))
    else:
        split_array = np.array(list(map(float,string.split(delimiter))))
    return split_array

def plotHistoBar(values,start,end,bin_size,tick_range=[],normalize=False):
    # properly plot a histogram (with the values shown in the current bins!)
    # input: values: what to create histogram of
    #        start, end, and bin_size: numbers to define an np.arange
    #        normalize: True if you want proportion out of 1
    #        tick_range: np.arange where you wanted labeled ticks
    import matplotlib.pyplot as plt
    bins = np.arange(start,end,bin_size)
    hist = np.histogram(values,bins)
    if normalize == True:
        yvalues = hist[0]/sum(hist[0])
    else:
        yvalues = hist[0]
    xr = (bins[1:]+bins[:-1])/2
    ax = plt.bar(xr,yvalues,width=0.8*bin_size)
    
    if tick_range == []: 
        tick_range = bins
    
    # get ticks
    ticks = []
    wanted_ticks = np.around(tick_range,3) # for some reason arange loses precision sometimes
    for tick in bins:
        if tick in wanted_ticks:
            ticks.append(tick)
    tick_mask = np.arange(start,len(bins),len(tick_range)+1) # is len(tick_range)+1 universal?
    plt.xticks(xr[tick_mask]-bin_size/2,ticks) # might have to check 0.5 being universal?

def tscorePlot(mdl,yrange=7.5,names=None): 
    # plot the tscores with 2 SEs shaded for model results
    import matplotlib.pyplot as plt
    if names is None: # need to add names for regression results without column names like MixedLMresults
        try:
            names = np.array(mdl.normalized_cov_params.columns)
        except:
            raise('You need some names for your xticks')
    plt.subplots(1,1, figsize=(4,3))
    xr = np.linspace(-.5,len(mdl.tvalues)-0.5,100)
    twoSD = 1.96*np.ones(100)
    plt.fill_between(xr,twoSD,-twoSD,alpha=0.1,color=[0,0,0])
    xticks = np.arange(0,len(mdl.tvalues)-0.5)
    plt.bar(xticks,mdl.tvalues)
    ax = plt.gca(); ax.set_ylim((-yrange,yrange))
    plt.ylabel('t-values')
    plt.xticks(xticks,names,rotation=90)
    plt.show()
    
def csvWriter(lists,filename):
    # write lists of info into separate lines of comma delmited format for opening in excel
    import csv
    with open(str(filename)+'.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', # comma separate
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for t in lists:
            csvwriter.writerow(t)

def seFromProp(num_correct,trials):
    # calculate standard error for proportions
    stderr = np.sqrt(num_correct*(1-num_correct/trials)/(trials-1)) / np.sqrt(trials)
    return stderr

def findInd(idx): # note: np.where does this, but it returns a tuple and this returns a list...actually now an array
    # get the indices when given boolean vector (like find function in matlab)
    idxs = [i for i,val in enumerate(idx) if val]
    return np.array(idxs)

def findAinB(A,B): # find where A is in B. this version works for lists and np arrays
    temp = set(A)
    inds = [i for i, val in enumerate(B) if val in temp] 
    return inds

def findAinBlists(A,B):
    # gets indicies for A in B where B is a list of lists
    inds = []
    for first in A:
        temp_ind = findInd(first in sublist for sublist in B)
        inds.extend(temp_ind)
    inds = np.unique(inds)
    return inds

def superVstack(a,b):
    # make it so you can vstack onto empty row
    if len(a)==0:
        stack = b
    else:
        stack = np.vstack([a,b])
    return stack

def findUniquePairs(a):
    # take a list of number pairs and return the unique pairs
    a = np.array(a)
    a.sort(axis=1)
    b = np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
        )
    _,d,cts = np.unique(b,return_index=True,return_counts=True)
    uniquePairs = a[d] 
    num_of_each = cts
    return uniquePairs,num_of_each

def addOnesColumn(X):
    # add a row of ones at the beginning of a np array for a regression
    if np.ndim(X)==1: # if a single column
        X = np.vstack((np.ones(len(X)),X)).T
    else:
        X = np.hstack((np.ones(len(X))[:,np.newaxis],X))
    return X

def fileDeleter(path,partial_name):
    import os,glob
    for filename in glob.glob(path+partial_name+"*"):        
        os.remove(filename)
'''
use these to delete the files made by ClusterRun:
fileDeleter("/home1/john/thetaBurst/code/","sge_engine")
fileDeleter("/home1/john/thetaBurst/code/","sge_controller")
fileDeleter("/home1/john/thetaBurst/code/","bcbio-e")
'''

def getLogicalChunks(array):
    # find start and end indices for chunks of values >0 in an array
    foundstart = False
    foundend = False
    startindex = 0
    endindex = 0
    starts = []
    ends = []    
    for i in range(0, len(array)):
        if array[i] != 0:
            if not foundstart:
                foundstart = True
                startindex = i
        else:
            if foundstart:
                foundend = True
                endindex = i - 1
        # for last point
        if i==(len(array)-1):
            if foundstart:
                starts.append(startindex)
                if array[i] == 1:
                    ends.append(i)
                else:
                    ends.append(i-1)
        if foundend:
            #print(startindex, endindex)
            starts.append(startindex)
            ends.append(endindex)
            foundstart = False
            foundend = False
            startindex = 0
            endindex = 0  
    return starts,ends

def bootstrap(data, bootnum=100, samples=None, bootfunc=None):
    """Performs bootstrap resampling on numpy arrays.
    Bootstrap resampling is used to understand confidence intervals of sample
    estimates. This function returns versions of the dataset resampled with
    replacement ("case bootstrapping"). These can all be run through a function
    or statistic to produce a distribution of values which can then be used to
    find the confidence intervals.
    Parameters
    ----------
    data : numpy.ndarray
        N-D array. The bootstrap resampling will be performed on the first
        index, so the first index should access the relevant information
        to be bootstrapped.
    bootnum : int, optional
        Number of bootstrap resamples
    samples : int, optional
        Number of samples in each resample. The default `None` sets samples to
        the number of datapoints
    bootfunc : function, optional
        Function to reduce the resampled data. Each bootstrap resample will
        be put through this function and the results returned. If `None`, the
        bootstrapped data will be returned
    Returns
    -------
    boot : numpy.ndarray
        If bootfunc is None, then each row is a bootstrap resample of the data.
        If bootfunc is specified, then the columns will correspond to the
        outputs of bootfunc.
    Examples
    --------
    Obtain a twice resampled array:
    >>> from astropy.stats import bootstrap
    >>> import numpy as np
    >>> from astropy.utils import NumpyRNGContext
    >>> bootarr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 2)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([[6., 9., 0., 6., 1., 1., 2., 8., 7., 0.],
           [3., 5., 6., 3., 5., 3., 5., 8., 8., 0.]])
    >>> bootresult.shape
    (2, 10)
    Obtain a statistic on the array
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 2, bootfunc=np.mean)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([4. , 4.6])
    Obtain a statistic with two outputs on the array
    >>> test_statistic = lambda x: (np.sum(x), np.mean(x))
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 3, bootfunc=test_statistic)
    >>> bootresult  # doctest: +FLOAT_CMP
    array([[40. ,  4. ],
           [46. ,  4.6],
           [35. ,  3.5]])
    >>> bootresult.shape
    (3, 2)
    Obtain a statistic with two outputs on the array, keeping only the first
    output
    >>> bootfunc = lambda x:test_statistic(x)[0]
    >>> with NumpyRNGContext(1):
    ...     bootresult = bootstrap(bootarr, 3, bootfunc=bootfunc)
    ...
    >>> bootresult  # doctest: +FLOAT_CMP
    array([40., 46., 35.])
    >>> bootresult.shape
    (3,)
    """
    if samples is None:
        samples = data.shape[0]

    # make sure the input is sane
    if samples < 1 or bootnum < 1:
        raise ValueError("neither 'samples' nor 'bootnum' can be less than 1.")

    if bootfunc is None:
        resultdims = (bootnum,) + (samples,) + data.shape[1:]
    else:
        # test number of outputs from bootfunc, avoid single outputs which are
        # array-like
        try:
            resultdims = (bootnum, len(bootfunc(data)))
        except TypeError:
            resultdims = (bootnum,)

    # create empty boot array
    boot = np.empty(resultdims)
    
    for i in range(bootnum):
        bootarr = np.random.randint(low=0, high=data.shape[0], size=samples)
        if bootfunc is None:
            boot[i] = data[bootarr]
        else:
            boot[i] = bootfunc(data[bootarr])

    return boot