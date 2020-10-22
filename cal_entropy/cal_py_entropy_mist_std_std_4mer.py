### pay attention to the index
import numpy as np
from collections import defaultdict
import multiprocessing
import os, sys, errno
import re
import argparse
import time
import gc
from scipy.stats import sem
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
import itertools
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn import preprocessing
import random
import discrete_random_variable as drv

def get_hist_auoBIN_mindistan(data_list):
    hist, edges_01 = np.histogram(data_list, bins="auto", range=[0,4], normed=False, weights=None, density=None)
    hist, edges_02 = np.histogram(data_list, bins="doane",range=[0,4], normed=False, weights=None, density=None)
    return max(len(edges_01),len(edges_02),75)

def get_hist_auoBIN_tilts(data_list):
    hist, edges_01 = np.histogram(data_list, bins="auto", range=[0,60], normed=False, weights=None, density=None)
    hist, edges_02 = np.histogram(data_list, bins="doane",range=[0,60], normed=False, weights=None, density=None)
    return max(len(edges_01),len(edges_02),75)

def get_hist_auoBIN_relrots(data_list):
    hist, edges_01 = np.histogram(data_list, bins="auto", range=[0,180], normed=False, weights=None, density=None)
    hist, edges_02 = np.histogram(data_list, bins="doane",range=[0,180], normed=False, weights=None, density=None)
    return max(len(edges_01),len(edges_02),75)

def get_hist_auoBIN_btcross(data_list):
    hist, edges_01 = np.histogram(data_list, bins="auto", range=[-50,50], normed=False, weights=None, density=None)
    hist, edges_02 = np.histogram(data_list, bins="doane",range=[-50,50], normed=False, weights=None, density=None)
    return max(len(edges_01),len(edges_02),75)


def shan_entropy(x,mat01,y=None,mat02=None,binNum=None,base=2):
    '''
    matrix specify which structure features: mindist,tilts,relrots, btcross
    if y is not None, it outputs joint entropy for x and y
    '''

    if mat01 == "dist":
        opbins01 = np.mean(np.array([ get_hist_auoBIN_mindistan(xxx) for xxx in x ])); RAN01 = [0,4]
        #opbins01 = get_hist_auoBIN_mindistan(x); RAN01 = [0,4]
    elif mat01 == "tilt":
        opbins01 = np.mean(np.array([ get_hist_auoBIN_tilts(xxx) for xxx in x ])); RAN01 = [0,60]
        #opbins01 = get_hist_auoBIN_tilts(x); RAN01 = [0,60]
    elif mat01 == "relt":
        opbins01 = np.mean(np.array([ get_hist_auoBIN_relrots(xxx) for xxx in x ])); RAN01 = [0,180]
        #opbins01 = get_hist_auoBIN_relrots(x); RAN01 = [0,180]
    elif mat01 == "cros":
        opbins01 = np.mean(np.array([ get_hist_auoBIN_btcross(xxx) for xxx in x ])); RAN01 = [-50,50]
        #opbins01 = get_hist_auoBIN_btcross(x); RAN01 = [-50,50]
    else:
        #print(mat01)
        print "error unidentified mat01"
        return 0

    if binNum is not None:
        opbins = binNum
    else:
        opbins = opbins01

    #hist_ = np.histogram(x, bins=opbins, range=RAN01, normed=None)[0]
    hist_list = [ np.histogram(xxx, bins=opbins, range=RAN01, normed=None)[0] for xxx in x ]
    hist_ = np.array([ np.sum(np.array([a,b,c,d]))*sem([a,b,c,d])  for a, b, c, d in zip(hist_list[0], hist_list[1], hist_list[2], hist_list[3]) ])
    hist_1d =  hist_.ravel()

    if y is not None:
        if mat02 == "dist":
            opbins02 = np.mean(np.array([ get_hist_auoBIN_mindistan(yyy) for yyy in y ])); RAN02 = [0,4]
            #opbins02 = get_hist_auoBIN_mindistan(y); RAN02 = [0,4]
        elif mat02 == "tilt":
            opbins02 = np.mean(np.array([ get_hist_auoBIN_tilts(yyy) for yyy in y ])); RAN02 = [0,60]
            #opbins02 = get_hist_auoBIN_tilts(y); RAN02 = [0,60]
        elif mat02 == "relt":
            opbins02 = np.mean(np.array([ get_hist_auoBIN_relrots(yyy) for yyy in y ])); RAN02 = [0,180]
            #opbins02 = get_hist_auoBIN_relrots(y); RAN02 = [0,180]
        elif mat02 == "cros":
            opbins02 = np.mean(np.array([ get_hist_auoBIN_btcross(yyy) for yyy in y ])); RAN02 = [-50,50]
            #opbins02 = get_hist_auoBIN_btcross(y); RAN02 = [-50,50]
        else:
            print "error unidentified mat02"
            return 0

        if binNum is not None:
            opbins = binNum
        else:
            opbins = int(round(np.mean([opbins01,opbins02]),0))

        #hist_ = np.histogram2d(x, y, bins=opbins, range=[RAN01,RAN02],normed=None)[0]
        hist_xy_list = [ np.histogram2d(xxx, yyy, bins=opbins, range=[RAN01,RAN02],normed=None)[0].ravel() for xxx, yyy in zip(x,y) ]
        hist_1d = np.array([ np.sum(np.array([a,b,c,d]))*sem([a,b,c,d]) for a, b, c, d in zip(hist_xy_list[0], hist_xy_list[1], hist_xy_list[2], hist_xy_list[3]) ])


    beta = 0.05
    #hist_1d =  hist_.ravel()
    len_d = hist_1d.shape[0]

    hist_normalized = (hist_1d + beta) / float(np.sum(hist_1d) + len_d*beta)
    S = -sum(hist_normalized* np.log2(hist_normalized))
    return S,opbins


def calc_MI_2D(xx,yy,mm01,mm02,binNum=None):
    H_X = shan_entropy(xx,mm01,binNum=binNum)[0]
    H_Y = shan_entropy(yy,mm02,binNum=binNum)[0]
    H_XY = shan_entropy(xx,mm01,yy,mm02,binNum=binNum)[0]

    MI_2D = H_X + H_Y - H_XY
    return MI_2D


def conditonal_entropy(xx,yy,mm01,mm02):
    '''
    xx is the conditional term
    H(Y|X) = H(X,Y) - H(X)
    '''
    H_X = shan_entropy(xx,mm01)[0]
    H_XY = shan_entropy(xx,mm01,yy,mm02)[0]
    
    return H_XY -  H_X

def cal_entropy_MIST_RAM_2D(a,b,c,d,factor,binNum=None):
    '''
    a,b,c,d must be in order: dist, tilt,relt,cros; 
    '''
    oneclus_dist_real_list = a
    one_hlx_tilts_list = b
    one_hlx_relrots_list = c
    oneclus_cross_list = d
   
    print ("check_datasize:", len(oneclus_dist_real_list[0]))
    ndx_list = []
    #for i in range(200):
    #    if factor == 3:
    #        ndx_rand = random.sample(range(0, factor*4), factor*4)
    #        ndx_list.append(ndx_rand)
    #    else:
    #        ndx_rand = random.sample(range(0, factor*4), 3*4) ### for comparing with 5mer
    #        ndx_list.append(ndx_rand)
    ndx_list = list(itertools.permutations([0,1,2,3], 4))

    dofs = [a, b, c, d]
    
    ent_mist = []
    for ndx_run in ndx_list:
        '''
        ### track matrix for diff oligomer
        ### different oligomer has different distributions per figure
        ### 3, 4, 5 for 3mer, 4mer, 5mer
        '''
        mask_list = []
        ### track matrix for diff oligomer
        ### different oligomer has different distributions per figure
        ### 3, 4, 5 for 3mer, 4mer, 5mer
        for ndx in ndx_run:
            if ndx == 0:
                mask_list.append("dist")
            if ndx == 1:
                mask_list.append("tilt")
            if ndx == 2:
                mask_list.append("relt")
            if ndx == 3:
                mask_list.append("cros")

        dofs_run = [dofs[i] for i in ndx_run]

        ent_1D = []
        for ds,mask in zip(dofs_run,mask_list):
            ent_ds = shan_entropy(ds,mask,binNum=binNum)[0]
            ent_1D.append(ent_ds)

        ent_1D_sum = np.sum(np.asarray(ent_1D))    
        
        ent_MI_2D = []
        ent_MI_2D.append(calc_MI_2D(dofs_run[0],dofs_run[1],
                                    mask_list[0],mask_list[1],binNum=binNum)) ## be careful about the order
        for ndx, a in enumerate(dofs_run):
            MI_temp_2D = []
            if ndx > 1:
                for nnn in range(ndx):
                    MI_temp_2D.append(calc_MI_2D(a,dofs_run[nnn],
                                                 mask_list[ndx],mask_list[nnn],binNum=binNum)) ## be careful about the order

                ent_MI_2D.append(np.max(np.asarray(MI_temp_2D)))
 
        #ent_MI_3D = []
        #ent_MI_3D.append(calc_MI_3D(dofs_run[0],dofs_run[1],dofs_run[2],
        #                                 mask_list[0],mask_list[1],mask_list[2]))
        #for ndx, k in enumerate(dofs_run):
        #    MI_temp_3D = []
        #    if ndx > 2:
        #        for i in range(ndx):
        #            for j in range(i):
        #                MI_temp_3D.append(calc_MI_3D(k,dofs_run[i],dofs_run[j],
        #                                                  mask_list[ndx],mask_list[i],mask_list[j]))

        #        ent_MI_3D.append(np.max(np.asarray(MI_temp_3D)))

        ent_mist.append(ent_1D_sum - np.sum(np.asarray(ent_MI_2D))) #- np.sum(np.asarray(ent_MI_3D)))
        print ("check:",ent_1D_sum," | ", np.sum(np.asarray(ent_MI_2D))," = ",ent_1D_sum - np.sum(np.asarray(ent_MI_2D)))

    return np.asarray(ent_mist).min()


def run_get_population(files):
    len_list = np.array([ len(open(f).readlines()) for f in files ])
   
    return len_list / float(sum(len_list))

def get_max_index(files):
    len_list = np.array([ len(open(f).readlines()) for f in files ])
    
    return list(len_list).index(len_list.max())

if __name__ == '__main__':
    #outputDir, group, repeat)

    parser = argparse.ArgumentParser(description='Plot random data in parallel')
    parser.add_argument('-of', '--output_filename', required=True,
                        help='The output file name')

    parser.add_argument('-o', '--outputDir', required=True,
                        help='The directory to which plot files should be saved')

    parser.add_argument('-i', '--data_fileset', required=True, default=1, nargs='+',
                        help='The list of input data file')

    parser.add_argument('-ns', '--num_scale', required=True, default=1)

    parser.add_argument('-nr', '--num_repeat', required=True, default=1)

    parser.add_argument('-nc', '--num_clust', required=True, default=1)

    args = parser.parse_args()

    
    input_dataset = args.data_fileset

    output_file = args.output_filename
    output_dir = args.outputDir
    
    output_scale = args.num_scale
    output_repeat = args.num_repeat
    output_clust = args.num_clust

    tar_ndx = get_max_index(input_dataset)

    #### minimum distance ####
    real_distan_list=[]
    if True:    
        f=open('./cal_real_distances_betw_helix_ordered/distan_bet_helix_cluser_'+str(tar_ndx)+'.dat')
        lines=f.readlines()
        #print ("len",len(lines))
        one_clust_total=[]
        one_clust_distan_a_b_list=[]
        one_clust_distan_b_c_list=[]
        one_clust_distan_c_d_list=[]
        one_clust_distan_d_a_list=[]
        for line in lines:
            one_clust_distan_a_b_list.append(float(line.strip('\n').split(" ")[5]))
            one_clust_distan_b_c_list.append(float(line.strip('\n').split(" ")[7]))
            one_clust_distan_c_d_list.append(float(line.strip('\n').split(" ")[9]))
            one_clust_distan_d_a_list.append(float(line.strip('\n').split(" ")[11]))
        one_clust_total.append(one_clust_distan_a_b_list)
        one_clust_total.append(one_clust_distan_b_c_list)
        one_clust_total.append(one_clust_distan_c_d_list)
        one_clust_total.append(one_clust_distan_d_a_list)
 
        real_distan_list.append(one_clust_total)
    #### Tilts angle ####
    tilts_of_clusters=[]
    if True:
        f=open('./cal_tilts_rots_relrots_simply_hisring/cal_rots_tilts_relrots_clus_'+str(tar_ndx)+'.dat')
        lines=f.readlines()
        one_hlx_tilts_list=[]
        one_hlx_tilts_dict=defaultdict(list)
        for line in lines:
            hlxid=line.split(" ")[5]
            #one_hlx_tilts_dict[str(hlxid)].append(line)
            #one_hlx_tilts_dict[str(hlxid)].append(line)
            one_hlx_tilts_dict[str(hlxid)].append(float(line.strip('\n').split(" ")[7]))
        one_hlx_tilts_list.append(one_hlx_tilts_dict['a'])
        one_hlx_tilts_list.append(one_hlx_tilts_dict['b'])
        one_hlx_tilts_list.append(one_hlx_tilts_dict['c'])
        one_hlx_tilts_list.append(one_hlx_tilts_dict['d'])

        tilts_of_clusters.append(one_hlx_tilts_list)
        
    #### relative rotation angle ####
    relrots_of_clusters=[]
    if True: 
        f=open('./cal_tilts_rots_relrots_simply_hisring/cal_rots_tilts_relrots_clus_'+str(tar_ndx)+'.dat')
        lines=f.readlines()
        one_hlx_relrots_list=[]
        one_hlx_relrots_dict=defaultdict(list)
        for line in lines:
            hlxid=line.split(" ")[5]
            #one_hlx_relrots_dict[str(hlxid)].append(line)
            one_hlx_relrots_dict[str(hlxid)].append(float(line.strip('\n').split(" ")[11]))
        one_hlx_relrots_list.append(one_hlx_relrots_dict['a'])
        one_hlx_relrots_list.append(one_hlx_relrots_dict['b'])
        one_hlx_relrots_list.append(one_hlx_relrots_dict['c'])
        one_hlx_relrots_list.append(one_hlx_relrots_dict['d'])
    
        relrots_of_clusters.append(one_hlx_relrots_list)
    
    #### crossing angle #####
    btcross_of_clusters=[]
    if True: 
        f=open('./crossing_angle_refine_hpa_ordered/cal_cross_angle_cluser_'+str(tar_ndx)+'.dat')
        lines=f.readlines()
        one_clust_total=[]
        one_clust_cross_a_b_list=[]
        one_clust_cross_b_c_list=[]
        one_clust_cross_c_d_list=[]
        one_clust_cross_d_a_list=[]
        for line in lines:
            one_clust_cross_a_b_list.append(float(line.strip('\n').split(" ")[4]))
            one_clust_cross_b_c_list.append(float(line.strip('\n').split(" ")[6]))
            one_clust_cross_c_d_list.append(float(line.strip('\n').split(" ")[8]))
            one_clust_cross_d_a_list.append(float(line.strip('\n').split(" ")[10]))
        one_clust_total.append(one_clust_cross_a_b_list)
        one_clust_total.append(one_clust_cross_b_c_list)
        one_clust_total.append(one_clust_cross_c_d_list)
        one_clust_total.append(one_clust_cross_d_a_list)
        btcross_of_clusters.append(one_clust_total)

    ### 4 is for 4mer
    entropy_4mer = cal_entropy_MIST_RAM_2D(real_distan_list[0], tilts_of_clusters[0],relrots_of_clusters[0],btcross_of_clusters[0],4, binNum=200)


    outputfile = open(str(output_dir)+output_file,"w")

    outputfile.write(str(output_scale)+" "+str(output_repeat)+" "+str(output_clust)+" entropy-- "+str(entropy_4mer)+"\n")
      
    outputfile.close()            
 
