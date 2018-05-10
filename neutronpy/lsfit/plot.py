import matplotlib.pyplot as plt
import numpy as np


class PlotFit(object):
    """class containing fit plotting methods

    Methods
    -------
     build_param_table
     plot
    """

    def build_param_table(self,function_str=None):
       r"""Builds a table of parameters for including in a plot or for use with print

       Parameters
       ==========
       function_str : string
           A string that describes the fitting function

       """

       parms=np.array(self.params)
       d_parms=np.array(self.xerror)
       #determine number of decimal places
       # use the differnce in the number of decimal places from the parameter and the error
       # to determine how many places to keep  add one more decimal place for good measure.
       # if the difference is less than 1 make it 2.
       print(parms)
       print ("\n")
       print(d_parms)
       n_dec=np.floor(np.log10(np.abs(parms)))-np.floor(np.log10(np.abs(d_parms)))+1
       n_dec[n_dec<1]=2
       n_dec=n_dec.astype('int')
       #table is generated below here
       str_out=''
       if function_str is not None:
           str_out=str_out+function_str+'\n'
       #flabel columns
       str_out=str_out+'param\t\terr\n'
       # for each parameter append the parameter and the error keep 2 decimal places for the error to the string
       for idx in range(len(parms)):
           fmtstr=r'"%1.'+str(n_dec[idx])+'e\t\t'+r'%1.1e\n" %(parms[idx],d_parms[idx])'
           #print(fmtstr)
           str_out=str_out+eval(fmtstr)
       #append the chisqruared value
       str_out=str_out+"$\chi^2=$%f\n"%self.chi2_min
       return str_out

    def plot(self,fit_function,function_str=None,plot_residuals=True):
           r""" Plots the data and the results of a fit

           Parameters
           ----------
           fit_function :function
               the function used in the fit
           function_str :string
                A string that describes the fit function.
           residuals :boolean
                If this is True it will give a plot of the residuals

           """
           xin=self.data[1]
           yin=self.data[2]
           errin=self.data[3]
           plt.figure()
           if plot_residuals:
               ax1=plt.subplot2grid((2,3),(0,0),colspan=2)
               ax2=plt.subplot2grid((2,3),(1,0),colspan=2)
               ax2.plot(xin,self.residuals(self.params,self.data),'b-')
           else:
               ax1=plt.subplot2grid((2,3),(0,0),colspan=2,rowspan=2)
           #plot fit
           ax1.plot(xin,fit_function(self.params,xin),'b-')
           plt.hold(True)
           #plot data
           ax1.errorbar(xin,yin,yerr=errin,fmt='ro')
           # add text table
           parmtext=self.build_param_table(function_str)
           ax3=plt.subplot2grid((2,3),(0,2),rowspan=2)
           ax3.text(0.1,0.5,parmtext,size=16)
           ax3.get_xaxis().set_visible(False)
           ax3.get_yaxis().set_visible(False)
           plt.show()
