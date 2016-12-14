# -*- coding: utf-8 -*-
r"""
A class for handeling a collection of scans

"""
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_c
import numpy as np


class Scans(object):
    r"""
    A class for a collection of scans

    Attributes
    ----------
    scans: dict
        A dictionary of scan objects
    num_scans: int
        The number of scans

    Methods
    -------
    waterfall(self,x='e',y='detector',label_column='h',offset=5,fmt='b-',legend=False)
          Makes a waterfall plot of the scan collection

    pcolor(self,x=None,z='detector',y=None,clims=None,color_norm=None,cmap='jet')
          Makes a false color contour plot of the scan collection


    """
    def __init__(self,scans_dict=None):
        self.scans=scans_dict
        if scans_dict!=None:
          self.num_scans=len(scans_dict)

    def update(self,scans_dict):
        r"""
        update the scans_dict to include the dictionary scans_dict
        This will update any scans that are already in the class and will append those that are not

        Parameters
        ----------
        scans_dict: dict
              A dictionary of multiple scans to add to the collection
        """
        self.scans.update(scans_dict)
        self.num_scans=len(scans_dict)

    def scans_check(self):
        r"""
        Check to see if their are scans in the object

        This should be used for other methods that work on individual scans in the collection

        Raises
        ------
        RuntimeError
             if there are no scans
        """
        if self.scans==None:
            raise RuntimeError('There must be at lest one scan')
    def waterfall(self,x='e',y='detector',label_column='h',offset=5,fmt='b-',legend=False):
        r"""
        Create a waterfall plot of all the scans in the collection

        Parameters
        ----------
          x: string
             one of the columns that is in each scan to plot on the x axis
          y: string
             one of the columns that is in each scan to plot on the y axis
             default is 'detector'
          label_column: string
             one of the columns that is in each scan to label each seperate scan in the plot
          offest: float
             how much to offset, in y, each successive curve from the previous one
          fmt: string
             a matplotlib format string for the plot.  The default is a blue line 'b-'
          legend: bool
             a flag to plot a legend default is False which will not plot a legend
        """
        self.scans_check()
        fh=plt.figure()
        plt.hold(True)
        for idx,scan_num in enumerate(self.scans.keys()):
            xin=self.scans[scan_num].data[x]
            yin=self.scans[scan_num].data[y]
            avg_label_val=self.scans[scan_num].data[label_column].mean()
            label_str="%s =%1.3f"%(label_column,avg_label_val)
            plt.plot(xin,yin+offset*idx,fmt,label=label_str)
            plt.xlabel(x)
            plt.ylabel(y)
        if legend:
            plt.legend()
        plt.show(block=False)

    def mean_col(self,col=None):
        r"""
        Take the mean of a given column in every scan of the collection

        Parameters
        ----------
        col: string
           The name of the column for the mean

        Returns
        -------
        array_like
            an array where each element is the average of the column of a specific
            scan in the collection
        """
        col_mean=np.zeros(self.num_scans)
        for idx, scan_num in enumerate(self.scans.keys()):
            col_mean[idx]=self.scans[scan_num].data[col].mean()
        return col_mean

    def pcolor(self,x=None,z='detector',y=None,clims=None,color_norm=None,cmap='jet'):
        r"""
        create a false colormap for a coloction of scans.

        The y direction is always what varies between scans.

        Parameters
        ----------
           x: string
             one of the columns that is in each scan to plot on the x axis
           z: string
             one of the columns that is in each scan to plot on the z axis
             default is 'detector'
           y: one of the columns that is in each scan to plot on the y axis.
              this is the parameter that varies between scans, but is cosntant over a given scan.
           clims: array_like
              this is an array of two floats.  The first is the minimum color scale,
              the second is the maximum of the color scale.  By default the maximum and minimum are chosen
           color_norm: string
                if the color scale is on a log or linear scale. Default is linear.  'log' is logscale
                any other string is linear
           cmap: string
              one of the matplot lib colormaps.  The default is jet.

        """
        self.scans_check()
        fh=plt.figure()
        #calculate y spacing
        meany=self.mean_col(col=y)
        biny=np.zeros(len(meany)+1)  # generate an array for bin boundaries of the y axis
        biny[1:-1]=(meany[:-1]+meany[1:])/2  #generate the bin boundaries internal to the array
        biny[0]=2*meany[0]-biny[1] # generate the first bin boundary to be the same disatance from the mean as the second bin boundary
        biny[-1]=2*meany[-1]-biny[-2] # generate the last bin boundary to be the same diastance from the mean as the next to last.
        if clims==None:
          #calcualte intensity range
          intens_max=0.
          intens_min=1.
          for idx, scan_num in enumerate(self.scans.keys()):
              maxz=self.scans[scan_num].data[z].max()
              minz=self.scans[scan_num].data[z].min()
              if maxz>intens_max:
                  intens_max=maxz
              if (minz<intens_min)&(minz>0):
                  intens_min=minz
          #print(intens_min,intens_max)
        else:
            intens_max=clims[1]
            intens_min=clims[0]
        for idx, scan_num in enumerate(self.scans.keys()):
            meansx=self.scans[scan_num].data[x]
            zvals=self.scans[scan_num].data[z]
            yvals=np.array([biny[idx],meany[idx],biny[idx+1]])
            xvals=np.zeros(len(meansx)+1)
            xvals[1:-1]=(meansx[:-1]+meansx[1:])/2.
            xvals[0]=2*meansx[0]-xvals[1]
            xvals[-1]=2*meansx[-1]-xvals[-2]
            xmat=np.vstack((xvals,xvals,xvals))
            ymat=np.tile(yvals,(len(xvals),1)).T
            zmat=np.vstack((zvals,zvals))
            if color_norm=='log':
                plt.pcolor(xmat,ymat,zmat,norm=mpl_c.LogNorm(vmin=intens_min,vmax=intens_max),cmap=cmap)
            else:
                plt.pcolor(xmat,ymat,zmat,vmin=intens_min,vmax=intens_max,cmap=cmap)


        plt.xlabel(x)
        plt.ylabel(y)
        plt.colorbar()
        plt.show(block=False)

        #return intens_min, intens_max
