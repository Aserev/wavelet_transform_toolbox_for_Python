import numpy as np # standard library
from numpy import exp,log,log2,sqrt,abs,cos,sin,pi # standard functions
from scipy.integrate import simps # for numerical integration by Simpson rule
from matplotlib import pyplot as plt # for plotting
from matplotlib.colors import ListedColormap # for colormaps
import csv # for writing csv files

def Cantor(x,n):
  """
  function: Cantor(x,n)

  parameters:
  -----------
  x = function argument (real number between 0 and 1)
  n = number of recursion steps

  return values:
  -------------
  value of the Cantor function at position x after n recursion steps. 

  details:
  -------
  Implementation of the Cantor function, defined as the distribution function 
  s(x) = mu([0,x]) of the uniform measure lying on the tradic Cantor set. See 
  figure 1(a) in [1] for a plot. 

  The function can be implemented recursively, see for instance [2].

  references: 
  ----------
  [1] Arneodo, A., Bacry, E., & Muzy, J. F. (1995). The thermodynamics of fractals 
      revisited with wavelets. Physica A: Statistical Mechanics and Its Applications
      213(1-2) 232-275
  [2] http://en.wikipedia.org/wiki/Cantor_function
  """
  if((n==0) or (x==0) or (x==1)): return x
  elif((1./3 <= x) and (x <= 2./3)): return 0.5
  elif((0 < x) and (x < 1./3)): return 0.5*Cantor(3*x,n-1)
  elif((2./3 < x) and (x < 1)): return 0.5+0.5*Cantor(3*x-2,n-1)
  else: return 0

def is_local_extremum(a,extremum_type='extremum',n=1,threshold=0):
  """
  function: is_local_extremum(a,type='extremum',n=1,threshold=0). 
  This function finds local extrema (or just maxima/minima) in an array. The definition of `local' is with 
  respect to the n next left and right neighbors. Furthermore, `threshold' (>0) qualifies values only as local
  extrema, if they differ from their neighbors by at least `threshold' (this helps to avoid classifing numerical noise/rounding errors
  as extrema). 

  parameters:
  ----------
  a: 1d numpy array of which we want to know the structure of local extrema
  type: either 'extremum','maximum' or 'minimum' (default: 'extremum')
  n: positive integer (default: n=1)
  threshold: positive value (default: 0)

  return value:
  -------------
  Array of `size(a)', with 1 where array a has a local extremum (or maximum/minimum) respect to the n left and right neighbors, 0 elsewhere.

  note:
  ----
  The left- and right-most value of the array are not considered local extrema.
  """  
  if(extremum_type=='maximum'):
    N = len(a) # size of arrary
    loc_max = np.zeros(N) # list of local maxima: 1, where local maxima, 0 else
    for i in range(1,N-1): 
    # define n left and right neighbors: 
      if(i < n):
        ln = a[:i] # all the left neighbors
        rn = a[i+1:i+1+n] # n right neighbors
      elif(i > N-n-1):
        ln = a[i-n:i] # n left neighbors
        rn = a[i+1:] # all the right neighbors
      else:
        ln = a[i-n:i] # n left neighbors
        rn = a[i+1:i+1+n] # n right neighbors
      ln_max = (a[i]>ln+threshold).astype(int) # 1, where element is larger than left neighbors (+threshold), 0 else
      rn_max = (a[i]>rn+threshold).astype(int) # 1, where element is larger than right neighbors (+threshold), 0 else
      if(sum(ln_max*rn_max) == n): loc_max[i] = 1 # if true, i is a local maximum
    return loc_max

  elif(extremum_type=='minimum'):
    N = len(a) # size of arrary
    loc_min = np.zeros(N) # list of local minima: 1, where local minima, 0 else
    for i in range(1,N-1): 
      # define n left and right neighbors: 
      if(i < n):
        ln = a[:i] # all the left neighbors
        rn = a[i+1:i+1+n] # n right neighbors
      elif(i > N-n-1):
        ln = a[i-n:i] # n left neighbors
        rn = a[i+1:] # all the right neighbors
      else:
        ln = a[i-n:i] # n left neighbors
        rn = a[i+1:i+1+n] # n right neighbors
      ln_min = (a[i]<ln-threshold).astype(int) # 1, where element is smaller than left neighbors (-threshold), 0 else
      rn_min = (a[i]<rn-threshold).astype(int) # 1, where element is smaller than right neighbors (-threshold), 0 else
      if(sum(ln_min*rn_min) == n): loc_min[i] = 1 # if true, is is a local minimum
    return loc_min

  elif(extremum_type=='extremum'):
    loc_max = is_local_extremum(a,extremum_type='maximum',n=n,threshold=threshold) # find local maxima
    loc_min = is_local_extremum(a,extremum_type='minimum',n=n,threshold=threshold) # find local minima
    loc_ext = loc_max+loc_min # add list of local maxima and minima
    return loc_ext

  else:
    raise 'error: extremum_type must be "extremum","maximum" or "minimum".'
    return a

def Gaussian(x,a):
	"""
	function: Gaussian(x,a), centered at x = 0 and with standard deviation a. 

	definition:
	----------
	Gaussian(x,a) = \frac{1}{2 \sqrt{\pi} a} exp((x/a)^2)

  	return value:
  	-------------
  	value of Gaussian wavelet (centered at x =0) at position x and with standard deviation a. 
	"""
	return exp(-0.5*(x/float(a))**2)/(sqrt(2*pi)*a)

def Gaussian_1st_derivative(x,a):
	"""
	function: Gaussian_1st_derivative(x,a), centered at x = 0 and with standard deviation a. 

	definition:
	----------
	Gaussian_1st_derivative(x,a) = x exp(-(x/a)^2/2)/(sqrt(2 pi) a^3).

  	return value:
  	-------------
  	value of negative of derivative of Gaussian mother wavelet (centered at x =0) at position x and standard deviation a. 
	"""
	return x*exp(-0.5*(x/float(a))**2)/(sqrt(2*pi)*a**3)

def Ricker(x,a):
	"""
	function: Ricker(x,a), centered at x = 0 and width parameter a.

	definition:
	----------
	Ricker(x,a) = (1-(x/a)^2) exp(-(x/a)^2/2) / (sqrt(2*pi) a^3)

  	return value:
  	-------------
  	value of Ricker wavelet (centered at x =0, width parameter a) at position x. 
  	This wavelet is also known as the Mexican hat wavelet. 
	"""
	return (1-(x/float(a))**2)*exp(-0.5*(x/float(a))**2)/(sqrt(2*pi)*a**3)


class WT:
    """
    The WT class provides an implementation of the wavelet transform (WT). We define
    the wavelet transform of a signal X(t) at scale `a' and position `b' with analyzing 
    wavelet psi(t) as follows:

    W_{\psi}(a,b) := \int_{-\infty}^\infty dt X(t) psi((x-b)/a). 

    Note that the overall normalisation must be incorporated into the wavelet psi. 
    For further details on the WT, see for instance [1,2,3]. 

    initialisation: 
    --------------
    X = WT(signal,arguments,scale,N,wavelet,finite_size_cutoff)

    arguments:
    ---------
    - signal:	signal to be analysed (a 1d-array, default:  devil's staircase function, see [1])
    - arguments:the points at which `signal' is sampled (1d-array, default: [0,1,2,...size(signal)])
    - scale:	[min_scale,max_scale] = minimum and maximum value of considered scale (default: [0.5,(value range)/4])
    - N:		number of (log)-equidistant points of scales between min_scale and max_scale (default: 100)
    - wavelet:	analysing wavelet (default: Gaussian wavelet). You may either use Gaussian, Gaussian_1st_derivative or Ricker, which
             	 are already implemented, or you may pass your own function (with two arguments: position x and width parameter a).
    - finite_size_cutoff: 	consider a signal X(t) sampled at times t_1,....t_n. We want to calculate its wavelet coefficient at a time `b' and at 
    						scale `a'. In order to avoid finite size effects, it is recommended to ignore wavelet coefficients at a time `b'
    						with a distance less than some multiple of `a' from the boundaries 't_1' and 't_n'. This multiple
    						is here set equal to `2' by default, but any other positive number may be passed. Setting finite_size_cutoff to 0
    						means there is no finite size cut-off considered. 

    methods:
    -------
    - X.get_WT(a,b) returns the wavelettransform at scale a and position b
    - X.get_WT_matrix() returns the entire wavelet transform as an (N x len(arguments)) matrix
    - X.output() writes the wavelet transform (i.e. the matrix from X.get_WT_matrix()) to a csv file
    - X.input() reads in a wavelet transform (in form of a matrix) from a csv file
    - X.heatmap() plots a heatmap of the wavelet transform of the signal in the (log(scale),arguments)-plane
    - X.skeleton() plots the skeleton (local extrema in the b-argument) of the wavelet transform of the signal in the (log(scale),arguments)-plane
    - X.test() shows the wavelet transform of the Cantor function, i.e. it reproduces figure 1(a) from [1].  

    references:
    ----------
    [1] Arneodo et al.The thermodynamics of fractals revisited with wavelets. Phys. A, 213 (1995), 232-275
    [2] Arneodo et al. Wavelet Based Multifractal Formalism: Applications to DNA Sequences, 
        Satellite Images of the Cloud Structure, and Stock Market Data, in Science of disasters, Springer (2002)
    [3] Yiou, P., Sornette, D. and Ghil, M., Data-adaptive wavelets and multi-scale singular-spectrum analysis. 
        Phys. D 142, (2002) 254-290.

    author:
    ------
    Sandro Claudio Lera
    Singapore-ETH Centre, ETH Zurich, 1 CREATE Way, #06-01 CREATE Tower, 138602 Singapore
    email: slera@ethz.ch

    notes:
    ----
    - You may use this code for your own applications. Note that this code is written for personal use and may contain bugs or inefficiencies. 
    - Report bugs to slera@ethz.ch
    """
    def __init__(	self,
    				signal = np.array([Cantor(i,2**8) for i in np.linspace(0,1,2**8)]),
    				arguments = None,
    				scale = None, 
    				N=100,
    				wavelet=Gaussian,
    				finite_size_cutoff=2.):

        self.__signal = signal # signal to be analyzed (1d-array)
        if(arguments is None): self.__arguments = np.r_[:np.size(signal)]
        else: self.__arguments = arguments # the points at which `signal' is sampled

        if(scale is None): 
          if(arguments is None):
            self.__sclim = np.array([0.5,0.25*np.size(signal)]) # default scale limits
          else:
            self.__sclim = np.array([0.5*(self.__arguments[1]-self.__arguments[0]),0.25*(self.__arguments[-1]-self.__arguments[0])])
      	else: 
          self.__sclim = scale # [sc_min,sc_max] = minimum and maximum value of considered scale

        self.__finite_size_cutoff = finite_size_cutoff # ignore coefficients with distance less than 'fsc*a' from the boundaries
      	self.__N = N # number of discrete scales
        self.__scales = np.logspace(log2(self.__sclim[0]),log2(self.__sclim[1]),num=self.__N,base=2.0) # considered scales

        self.__wavelet = wavelet # analysing wavelet, function of one argument
        self.__WTvals = np.zeros((self.__N,len(self.__arguments))) # wavelet transform at scale a and position b
        self.__WT_analysis() # calculate the wavelet transform

    # wavelet transform at scale a and position b: 
    def __WT(self,a,b):
    	wvlt_vals = self.__wavelet(self.__arguments-b,a) # wavelet (must be properly normalized!)
      	integrand = self.__signal*wvlt_vals # integrand
      	return simps(integrand,self.__arguments) # apply Simpson integration formula
    	
    # wavelet transform analysis at different scales a and positions b: 
    # we only consider the transform at positions b with at least a 
    # distance 'finite_size_cutoff*a' from the boundaries, in order to omit finite size effects
    def __WT_analysis(self):
    	left_lim = self.__arguments[0] # left most value
    	right_lim = self.__arguments[-1] # right most value
        for i,a in enumerate(self.__scales): # iterate through scales
        	for j,b in enumerate(self.__arguments): # iterate through scales
        		if( (b- self.__finite_size_cutoff*a < left_lim) or (b+self.__finite_size_cutoff*a > right_lim) ): # check if close to boundary
          			self.__WTvals[i,j] = np.nan # boundary values are ignored
          			continue
        		self.__WTvals[i,j] = self.__WT(a,b)  # calculate wavelet transform at scale a and position b
     
    # return the wavelet transform at scale a and position b
    def get_WT(self,a,b): 
        """
        Returns the wavelet transform at scale a and position b.
        See documentation of class "WT" for details. 

       	initialisation: 
      	-------------
      	X.get_WT(a,b)

        arguments:
        ---------
       	X: an instance of class WT
       	a: scale
       	b: position
       	"""
        return self.__WT(a,b)

    # return the wavelet transform matrix of dimension (N x len(arguments))
    def get_WT_matrix(self): 
      	"""
      	Returns the entire wavelet transform as an (N x len(arguments)) matrix. 
      	See documentation of class "WT" for details. 

        initialisation: 
        --------------
        X.get_WT_matrix()

        arguments:
        ---------
        X: an instance of class WT
        """
        return self.__WTvals

    def output(self,filename='wavelet_matrix.csv'):
        """
        Writes the wavelet transform (i.e. the matrix from X.get_WT_matrix()) to a csv file. 
        See documentation of class "WT" for details. 

        initialisation:
        --------------
        X.output(filename='wavelet_matrix.csv')

        arguments:
        ---------
        X: an instance of class WT
        filename: name of csv file (default: 'wavelet_matrix')
        """
        ow = open(filename,'w') # output writer
        oo = csv.writer(ow,delimiter=',') # output object
        oo.writerows(self.__WTvals) # write f to output 
        ow.close() # close

    def input(self,filename,signal=None,args=None,scales=None):
        """
        Read a matrix form of a csv. This matrix is interpreted as WT.
        See documentation of class "WT" for details. 

        initialisation:
        --------------
        X.input(filename,
                signal=None,
                args=None,
                scales=None)

        arguments:
        ---------
        X: an instance of class WT
        filename: name of csv file to read in
        signal: signal that was analyzed (1d-array, optional)
        args: points at which the signal was sampled (1d-array, optional)
        scales: scales at which the signal was sampled (1d-array, optional)
        """
        self.__WTvals = np.genfromtxt(filename,delimiter=',') # load the csv
        self.__N = np.size(self.__WTvals[:,0])
        if(signal is None): self.__signal = np.r_[:np.size(self.__WTvals[0,:])]
        else: self.__signal = signal
        if(args is None): self.__arguments = np.r_[:np.size(self.__WTvals[0,:])]
        else: self.__arguments = args
        if(scales is None): self.__scales = np.r_[:N]
        else: self.__scales = scales
        self.__sclim = [self.__scales[0],self.__scales[-1]]


    def heatmap(self,logscale=False,lower_threshold=None,upper_threshold=None,show_signal=False,argument_name='position',signal_name='signal',filename='WT_analysis.png'):
        """
        Returns a heat map plot of the wavelet transform. See documentation of class "WT" for details.  

        initialisation:
        --------------
        X.heatmap(logscale=False,
                  lower_threshold=None,
                  upper_threshold=None,
                  show_signal=False,
                  argument_name='position',
                  signal_name='signal',
                  filename='WTMM_analysis.png')

        arguments:
        ---------
        X: an instance of class WT
        logscale: if true, the wavelet coefficients are rescaled to the interval [0.0001,1] and then the
                  logarithm is taken. This may help resolve fine contours at small scales. (default: False)
        lower_threshold:  if a numerical value is passed, all wavelet coefficients below the threshold are 
                          set to NaN. This may help improve the resolution by ignoring large finite size effects. (default: None)
        upper_threshold:  if a numerical value is passed, all wavelet coefficients above the threshold are 
                          set to NaN. This may help improve the resolution by ignoring large finite size effects. (default: None)
        show_signal: if True, the analyzed signal is plotted over the heatmap 
        argument_name: label for the x-axis (default: 'position')
        signal_name: if show_signal is True, you can pass the y-label information for the signal (default: 'signal')            
        filename: name of output figure (default: 'WT_analysis.png')
        """
        plt.clf()
        fig = plt.figure(0,figsize=(10,5))
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)

        # set axes: 
        ax = plt.gca()
        ax.set_xlabel(argument_name)
        ax.set_ylabel(r'$\log_2$(scale)')
        ax.set_yscale('log',basey=2)
        ax.set_ylim(self.__sclim[0],self.__sclim[1])

        # prepare the data for plotting:
        vals = self.__WTvals.copy()
        vals = np.ma.masked_array(vals,np.isnan(vals)) # mask the NaN (from finite size cut-offs)
        if (logscale==True): # rescale  to[0.001,1] and take logarithm, if requested
          vals = vals + np.ma.abs(np.nanmin(vals))+0.001 
          vals = vals/np.nanmax(vals)
          vals = np.ma.log(vals)
        if (lower_threshold is not None): # set values below threshold to NaN
          vals = np.ma.masked_where(vals<lower_threshold,vals)
        if (upper_threshold is not None): # set values above threshold to NaN
          vals = np.ma.masked_where(vals>upper_threshold,vals)  

        # make the plot:
        B,A = np.meshgrid(self.__arguments,self.__scales) # arguments for the plot
        plt.pcolormesh(B,A,vals)
        plt.colorbar(orientation='horizontal')

        # make plot of signal:
        if(show_signal==True):
          ax2 = plt.twinx()
          ax2.plot(self.__arguments,self.__signal,color='Brown')
          ax2.tick_params(axis='y', colors='brown')
          ax2.spines['right'].set_color('brown')
          ax2.set_ylabel(signal_name,color='Brown')

        # write to output:
        plt.tight_layout()
        plt.savefig(filename,format='png',dpi=1000)
        plt.close()

    def skeleton(self,extremum_type='extremum',show_signal=True,lower_threshold=None,upper_threshold=None,argument_name='position',signal_name='signal',filename='WT_skeleton.png'):
        """
        Returns skeleton of a wavelet transform. The skeleton is defined as the structure of the local
        minima/maxima/extrema in the position argument. See documentation of class "WT" and method "heatmap" for details.  

        initialisation:
        --------------
        X.skeleton(extremum_type='extremum',
                    show_signal=True,
                    lower_threshold=None,
                    upper_threshold=None,
                    argument_name='position',
                    signal_name='signal',
                    filename='WTMM_skeleton.png')

        arguments:
        ---------
        X: an instance of class WT
        extremum_type: type of local extremum that defines skeleton structure, either 'maximum','mininum', or 'extremum'
        show_signal: if True, the analyzed signal is plotted over the skeleton
        lower_threshold:  if a numerical value is passed, all wavelet coefficients below the threshold are 
                          set to NaN. This may help getting rid of finite size effects. (default: None)
        upper_threshold:  if a numerical value is passed, all wavelet coefficients above the threshold are 
                          set to NaN. This may help getting rid of finite size effects. (default: None) 
        argument_name: label for the x-axis (default: 'position')
        signal_name: if show_signal is True, you can pass the y-label information for the signal (default: 'signal')
        filename: name of output figure (default: 'WT_skeleton.png')
        """
        skel = np.ma.array(self.__WTvals.copy(),mask=True) # initialise skeleton, nothing masked yet
        for i in range(self.__N): # find local extrema at each scale
          extremum_structure = is_local_extremum(self.__WTvals[i,:],extremum_type=extremum_type,threshold=10.**-10) # 1 where local extremum, 0 else
          skel[i,:].mask =  1-extremum_structure # mask value as NaN, where no extremum
       
      	if (lower_threshold is not None): # set values below threshold to NaN
          skel = np.ma.masked_where(self.__WTvals < lower_threshold,skel) 
        if (upper_threshold is not None): # set values above threshold to NaN
          skel = np.ma.masked_where(self.__WTvals > upper_threshold,skel) 

        # create plot window: 
        fig = plt.figure(0,figsize=(10,5))
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)

        # set axes: 
        ax = plt.gca()
        ax.set_xlabel(argument_name)
        ax.set_ylabel(r'$\log_2$(scale)')
        ax.set_yscale('log',basey=2)
        ax.set_ylim(self.__sclim[0],self.__sclim[1])

        # make the plot:
        B,A = np.meshgrid(self.__arguments,self.__scales)
        plt.pcolormesh(B,A,skel)
        plt.colorbar(orientation='horizontal')

        # make plot of signal:
        if(show_signal==True):
          ax2 = plt.twinx()
          ax2.plot(self.__arguments,self.__signal,color='Brown',linewidth=0.5)
          ax2.set_ylabel(signal_name,color='brown')
          ax2.tick_params(axis='y', colors='brown')
          ax2.spines['right'].set_color('brown')
          ax2.spines['left'].set_color('SteelBlue')
          ax.tick_params(axis='y', colors='SteelBlue')
          ax.set_ylabel(r'$\log_2$(scale)',color='SteelBlue')
          
        # write to output:
        plt.tight_layout()
        plt.savefig(filename,format='png',dpi=1500) # high resolution to see small scales
        plt.close()

    def test(self):
        """
        Reproduces figure 1(a) from [1]. See documentation of class "WT" for details. 

        initialisation: 
        --------------
        X = WT()
        X.test() where X is an instance of the class WT. 

        note:
        ----
        Do not call X.test() if you pass your own arguments to the instance X! They will
        be overwritten by X.test().

        references: 
        ----------
        [1] Arneodo, a., Bacry, E., & Muzy, J. F. (1995). The thermodynamics of fractals 
            revisited with wavelets. Physica A: Statistical Mechanics and Its Applications
            213(1-2) 232-275
        """
        n = 2**12 # number of sample points
        # n = 2**15 # (high resolution down to really fine scales, takes several hours to execute)
        self.__signal = np.array([Cantor(i,n) for i in np.linspace(0,1,n)]) # Cantor function 
        self.__arguments = np.linspace(0,1,n) # arguments
        self.__N = 300 # number of scales
        self.__sclim = np.array([2.**(-12),2.**(-2)]) # scale limits 
        self.__scales = np.logspace(log2(self.__sclim[0]),log2(self.__sclim[1]),num=self.__N,base=2.0) # scales
        self.__WTvals = np.zeros((self.__N,len(self.__arguments))) # wavelet transform at scale a and position b
        self.__wavelet = Gaussian_1st_derivative # use first derivative to analyse structure of slopes
        self.__WT_analysis() # perform wavelet transform at all scales and positions
        self.heatmap(lower_threshold=-1,
                      show_signal=True,
                      signal_name='Cantor',
                      filename='Cantor_test_heatmap.png') # plot heat map
        self.skeleton(signal_name='Cantor',
                      filename='Cantor_test_skeleton.png') # plot skeleton structure