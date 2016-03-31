#import matplotlib
#matplotlib.use('svg')
#matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import pylab
import numpy as np
#import pyodbc
#import MySQLdb
import math
#from scipy import interpolate
#from pylab import *
#from random import randint
import random
dm_list=[20]#[10,20,50,100]
size_list=[7]
cow_list=[5, 10, 15, 20]#[5,6,7,8,9,10,11,12,13,14,15]

class single_nanoparticle():
    def __init__(self,d,x,y):
        self.x_pos = x
        self.y_pos = y
        self.diameter = d
        self.area = 3.14/4.0 *d**2
        
    def distance(self,other):
        return ((self.pos_x-other.pos_x)**2 + (self.pos_y-other.pos_y)**2)**0.5
        
def sum_area():
    return sum_area

def generate_random_NP(no_NP,X_size,Y_size,d,dm):
    NP_pos=np.zeros([no_NP,5])
    for n in range(no_NP):
        NP_pos[n,0] = n
        NP_pos[n,1] = random.random()*float(X_size)
        NP_pos[n,2] = random.random()*float(Y_size)
        NP_pos[n,3] = random.gauss(d, float(d)/float(dm))
        NP_pos[n,4] = 1
        n+=1
    return NP_pos

def area_of_agglomarations(NP_pos):
    x_min = min(NP_pos[:,1])
    x_max = max(NP_pos[:,1])
    y_min = min(NP_pos[:,2])
    y_max = max(NP_pos[:,2])
    X_aray = np.zeros(2,2)
    
for dm in dm_list:
    print 'dm: ' + str(dm)
    for size in size_list:#range(3,11,1):
        d = size
        plot_cow_x = []
        plot_cow_y = []
        print 'size: ' + str(d)
        for c in cow_list:
            cov=float(c)/100.0
            X_size = 1000.0
            Y_size = 1000.0
            no_NP = int(float(cov) * (float(X_size)*float(Y_size)) / (math.pi*0.25*float(d)**2.0))
            coverage = no_NP*(math.pi*0.25*float(d)**2.0)/(float(X_size)*float(Y_size))

            print 'number of particles: ' + str(no_NP)
            print 'coverage = in % ' + str(coverage*100)
            NP_pos=generate_random_NP(no_NP,X_size,Y_size,d,dm)
            
            no_2d = 0
            NP_pos_remove=np.zeros(int(no_NP))
            agg = {}
            for n in range(no_NP):
                #friends_of_n = 1
                for nn in range(n,int(no_NP)):
                    if abs((NP_pos[n,1]-NP_pos[nn,1])**2+abs(NP_pos[n,2]-NP_pos[nn,2])**2) < (0.5*(NP_pos[n,3]+NP_pos[nn,3]))**2 and n != nn:
                        #if (n in ):
                        #print 'particle no: '+str(NP_pos[n,0]) + ' and '+ str(NP_pos[nn,0]) +' belongs together'
                        #print NP_pos[n,:]
                        #print NP_pos[nn,:]
                        NP_pos[n,4] += 1
                        NP_pos[nn,4] += 1
                        NP_pos_remove[n] = 1
                        NP_pos_remove[nn] = 1
                        no_2d +=1
                        #friends_of_n += 1
            NP_Single=NP_pos[NP_pos_remove == 0, :]
            #print 'Number of single particles : ' + str(len(NP_Single))
            #print 'fraction of single particles : ' + str(float(len(NP_Single))/float(no_NP))
            NP_new=NP_pos[NP_pos_remove == 1, :]
            plot_cow_x.append(coverage)
            plot_cow_y.append(float(no_NP-len(NP_Single))/float(no_NP))
            if len(NP_new) > 0:
                agg_array=np.zeros((len(NP_new),len(NP_new)))
                #agg_array[:,:]=0
                for n in range(len(NP_new)):
                    #print n
                    #print agg_array
                    for nn in range(n,len(NP_new)):
                        if ((NP_new[n,1]-NP_new[nn,1])**2+(NP_new[n,2]-NP_new[nn,2])**2) < (0.5*(NP_pos[n,3]+NP_pos[nn,3]))**2 and n != nn:
                            #print str(n) + ' and ' + str(nn)
                            if (NP_new[n,0] in agg_array) or (NP_new[nn,0] in agg_array):
                                #print str(n) + ' or '+ str(nn) + ' is in agg'
                                for i in range(len(NP_new)):
                                    if (agg_array[n,i] == NP_new[n,0]) or (agg_array[nn,i] == NP_new[nn,0]):
                                        agg_array[n,i] = NP_new[n,0]
                                        agg_array[nn,i] = NP_new[nn,0]
                                        break
                            else:
                                for i in range(len(NP_new)):
                                    if sum(agg_array[:,i])  == 0:
                                        agg_array[n,i] = NP_new[n,0]
                                        agg_array[nn,i] = NP_new[nn,0]
                                        break
                #print agg_array
                agg_array_new=agg_array[:,sum(agg_array) != 0]
                NP_agg_V = []
                
                for i in range(len(agg_array_new[1,:])):
                    temp = []
                    area_temp = []
                    for ii in range(len(agg_array_new[:,1])):
                        if agg_array_new[ii,i] != 0:
                            temp.append((NP_pos[agg_array_new[ii,i],3])**3.0)
                            #area_temp.append(NP_pos[agg_array_new[ii,i],:])
                    #print temp
                    NP_agg_V.append((sum(temp))**(1.0/3.0))
            list_of_size = np.append(NP_Single[:,3],np.array(NP_agg_V)[:])
            
            #print 'fraction of toughing in % '+str(float(no_2d) / float(no_NP)*100)
            if True:
                fig = plt.figure()
                fig.subplots_adjust(bottom=0.2) # Make room for x-label
                fig.subplots_adjust(left=0.2) # Make room for x-label
                fig.subplots_adjust(right=0.8) # Make room for x-label

                ratio = 0.61803398             # Golden mean
                #ratio =0.4                   # This figure should be very wide to span two columns
                fig_width = 14
                fig_width = fig_width /2.54     # width in cm converted to inches
                fig_height = fig_width*ratio
                fig.set_size_inches(fig_width,fig_height)
                axis = fig.add_subplot(211)
                axis2 = fig.add_subplot(212)
                #x = np.random.random(100)*3+5
                normed_value = 2

                hist, bins = np.histogram(list_of_size, bins=200,range=(0,20), density=True)
                hist2, bins2 = np.histogram(NP_Single[:,3], bins=200,range=(0,20), density=True)
                widths = np.diff(bins)
                hist *= len(list_of_size)*widths
                hist2 *= len(NP_Single[:,3])*widths

                
                axis.bar(bins2[:-1], hist2, widths,color='b' )
                axis2.bar(bins[:-1], hist, widths,color='r' )
                
                #axis2.set_xticklabels([0,5,10,15])
                #axis.plot(bins[:-1],hist, 'b-',label='agg')
                #axis.plot(bins2[:-1],hist2, 'r-',label='agg')
                axis.set_xlim(0,15)
                axis2.set_xlim(0,15)
                axis.set_xticklabels([])
                #axis.set_ylim(0,1)
                #axis2.set_ylim(0,1)
                axis.annotate(str('No of NP used: '+str(no_NP)),xy=(0.1, 0.8), xycoords='axes fraction', fontsize=8)
                axis.tick_params(direction='in', length=6, width=1, colors='k',labelsize=8,axis='both',pad=3)
                axis2.tick_params(direction='in', length=6, width=1, colors='k',labelsize=8,axis='both',pad=3)
                axis2.set_xlabel('Size / [nm]', fontsize=8)
                axis.set_ylabel('Counts', fontsize=8)
                axis2.set_ylabel('Counts', fontsize=8)
                directory = 'fig-'+str(dm)+'/size_'+str(d)
                file_path = directory +'/Size_distribution_cov_'+str(c)+'_size_'+str(d)+'.png'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                print file_path
                plt.savefig(file_path,dpi=600)
                plt.close()
            if True:
                fig = plt.figure()
                fig.subplots_adjust(bottom=0.2) # Make room for x-label
                #fig.subplots_adjust(left=0.2) # Make room for x-label
                #fig.subplots_adjust(right=0.8) # Make room for x-label

                ratio = 0.61803398             # Golden mean
                #ratio = 0.4                     # This figure should be very wide to span two columns
                fig_width = 10
                fig_width = fig_width /2.54     # width in cm converted to inches
                fig_height = fig_width
                fig.set_size_inches(fig_width,fig_height)

                axis = fig.add_subplot(111,aspect='equal')
                #dx=NP_Single[:,3]
                #dx_in_points = np.diff(axix.transData.transform(zip([0]*len(dx), dx)))
                #print dx_in_points
                #ax.scatter(5,5,s=9*d**2, facecolors='blue',edgecolors='none')
                if len(NP_Single) > 0:
                    for i in range(len(NP_Single)):
                        circ=plt.Circle((NP_Single[i,1],NP_Single[i,2]),radius=0.5*float(NP_Single[i,3]),alpha=0.5,color='g')
                        axis.add_patch(circ)
                if len(NP_new) > 0:
                    for i in range(len(NP_new)):
                        circ=plt.Circle((NP_new[i,1],NP_new[i,2]),radius=0.5*float(NP_new[i,3]),alpha=0.5,color='r')
                        axis.add_patch(circ)
                #axis.set_aspect('equal')
                
                axis.set_xlim(0,X_size)
                axis.set_ylim(0,Y_size)
                axis.tick_params(direction='in', length=6, width=1, colors='k',labelsize=8,axis='both',pad=3)
                axis.set_xlabel('X axis / [nm]', fontsize=8)
                axis.set_ylabel('Y axis / [nm]', fontsize=8)
                #ax.scatter(NP_Single[:,1],NP_Single[:,2],s=1dx_in_points**2,marker='s', edgecolors='none')
                directory = 'fig-'+str(dm)+'/size_'+str(d)
                file_path = directory +'/cow_'+str(c)+'_size_'+str(d)+'_area.png'
                if not os.path.exists(directory):
                    os.makedirs(directory)
                print file_path
                plt.savefig(file_path,dpi=600)
                plt.close()

        if len(plot_cow_x) > 9:
            fig = plt.figure()
            fig.subplots_adjust(bottom=0.2) # Make room for x-label
            fig.subplots_adjust(left=0.2) # Make room for x-label
            fig.subplots_adjust(right=0.8) # Make room for x-label

            ratio = 0.61803398             # Golden mean
            ratio = 0.4                     # This figure should be very wide to span two columns
            fig_width = 14
            fig_width = fig_width /2.54     # width in cm converted to inches
            fig_height = fig_width*ratio
            fig.set_size_inches(fig_width,fig_height)

            axis = fig.add_subplot(1,1,1)
            #axis.semilogy()
            axis.plot(plot_cow_x,plot_cow_y, 'b*',label='agg')

            #plt.xlim(0,50000)
            plt.ylim(0,max(plot_cow_y)*1.2)
            axis.tick_params(direction='in', length=6, width=1, colors='k',labelsize=8,axis='both',pad=3)
            axis.set_xlabel('Coverage', fontsize=8)
            axis.set_ylabel('Fraction of agglomarates', fontsize=8)
            #plt.tight_layout()
            #plt.show()
            directory = 'fig-'+str(dm)+'/size_'+str(d)
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = directory+'/Coverage_size_'+str(d)+'.png'
            print file_path
            plt.savefig(file_path,dpi=600)
            plt.close()
