#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:36:54 2022

@author: a849882640
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 20:11:29 2022

@author: a849882640
"""



import networkx as nx #导入建网络模型包，命名nx
import matplotlib.pyplot as mp #导入科学绘图包，命名mp
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


T1=100     #可能会出现错误因为下面的t与该t不一致
num=100
N1=200
num1=list(np.zeros(T1))*num
num2=list(np.zeros((3,T1)))*num
x_per_all_=list(np.zeros(T1))
x_per_all_a=list(np.zeros((3,T1)))
x_per_all_a1=list(np.zeros((3,T1)))
x_per_all_a2=list(np.zeros((3,T1)))
x_per_all_y=list(np.zeros((3,T1)))
xt_per_all_=list(np.zeros(T1))
xt_per_all_a=list(np.zeros((3,T1)))
xt_per_all_a1=list(np.zeros((3,T1)))
xt_per_all_a2=list(np.zeros((3,T1)))
xt_per_all_y=list(np.zeros((3,T1)))
for l in range(num):
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-

    N=200
    T=100
    k=0.05
    p0=0.3    #策略2的初始概率
    theta=0.3  #转型升级策略的比例
    fda=50000
    b=5000
    pda=600000
    cda=400000
    beta=0.7
    q=5000
    theta_=0.3
    theta_a=[0.3,0.3,0.3]
    theta_a1=[0.3,0.3,0.3]
    theta_a2=[0.3,0.3,0.3]
    theta_y=[0.3,0.3,0.3]
    #aaerfa=[0.2,-0.4*theta_a[1]+0.4,-1.6*theta_a[2]*theta_a[2]+1.6*theta_a[2]]      #0.2
    #aaerfa1=[0.05,-0.1*theta_a1[1]+0.1,-0.4*theta_a1[2]*theta_a1[2]+0.4*theta_a1[2]]      #0.05
    #aaerfa2=[0.25,-0.5*theta_a2[1]+0.5,-2*theta_a2[2]*theta_a2[2]+2*theta_a2[2]]        #0.25
    #yyita=[1,-2*theta_y[1]+2,-8*theta_y[2]*theta_y[2]+8*theta_y[2]]        #0.25
    tz=500000
    '''
    cdi=cda*(1-aerfa2)
    fdi=fda*(1+aerfa)
    pdi=pda*(1+aerfa1)
    '''
    def get_0_1_array(array,rate=0.2):
        '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
        zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
        new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
        new_array[:zeros_num] = 0 #将一部分换为0
        np.random.shuffle(new_array)#将0和1的顺序打乱
        re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
        return re_array

    #得到一个全1矩阵，按照rate=0.7的比率生成新矩阵
    arr = np.ones(N)
    arr2 = arr.reshape(1,N)
    start_c = get_0_1_array(arr2,rate=0.7)
    start_c=start_c.reshape(N)

    pr_all=np.zeros((3,T,N,N))  # 收益
    game_theory=np.zeros((5,T,N))
    for q in range(3):
        game_theory[q,0,:]=start_c
    game_theory_=np.zeros((T,N))
    game_theory_[0,:]=start_c
    game_theory_a=game_theory.copy()
    game_theory_a1=game_theory.copy()
    game_theory_a2=game_theory.copy()
    game_theory_y=game_theory.copy()
    ctpr_all=pr_all.copy()
    pr_all_=np.zeros((T,N,N))
    pr_all_a=pr_all.copy()
    pr_all_a1=pr_all.copy()
    pr_all_a2=pr_all.copy()
    pr_all_y=pr_all.copy()
    ctpr_all_a=ctpr_all.copy()
    ctpr_all_a1=ctpr_all.copy()
    ctpr_all_a2=ctpr_all.copy()
    ctpr_all_y=ctpr_all.copy()

    for t in range(T): 
        if t==0:
            G=nx.watts_strogatz_graph(N,3,0.5)
            A=nx.to_numpy_array(G)
            x_per_=[p0]
            xt_per_=[]
            x_per_a=[[p0],[p0],[p0]]   #6个选择α
            xt_per_a=[[],[],[]]
            x_per_a1=[[p0],[p0],[p0]]   #6个选择α
            xt_per_a1=[[],[],[]]
            x_per_a2=[[p0],[p0],[p0]]   #6个选择α
            xt_per_a2=[[],[],[]]
            x_per_y=[[p0],[p0],[p0]]   #6个选择α
            xt_per_y=[[],[],[]]
            #更新前第一次
            for i in range(N):
                B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                aerfa=0.2
                aerfa1=0.05
                aerfa2=0.25
                yita=1
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                for j in B:
                    if start_c[i]==0:
                        if start_c[j]==0:
                            prr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                        else:
                            prr0=(fda*q/(1-theta_)+pda-cda)*(1-beta)   #策略为【0，1】
                    else:
                        if start_c[j]==1:
                            prr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)     #都为1时策略
                        else:
                            prr0=((fdi+b*yita)*q/(theta_)+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                    pr_all_[0,i,j]=prr0
                count_0=0
                o_sum=0
            for i in range(N):
                if start_c[i]==1:
                    count_0+=1
                    o_sum+=sum(pr_all_[0,i,:])
            xt_per_.append(o_sum/count_0)     #求初始时刻策略1的总收益
                

            #更新后第一次
            for a in range(3):          #a
                aaerfa=[0.2,-0.4*theta_a[1]+0.4,-1.2*theta_a[2]*theta_a[2]+1.2*theta_a[2]]
                aerfa=aaerfa[a]
                aerfa1=0.05
                aerfa2=0.25
                yita=1
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if start_c[i]==0:
                            if start_c[j]==0:
                                prr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                            else:
                                prr0=(fda*q/(1-theta_a[a])+pda-cda)*(1-beta)   #策略为【0，1】
                        else:
                            if start_c[j]==1:
                                prr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)     #都为1时策略
                            else:
                                prr0=((fdi+b*yita)*q/(theta_a[a])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                        pr_all_a[a,0,i,j]=prr0
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if start_c[i]==0:
                            if start_c[j]==0:
                                prr1=((fdi+b*yita)*q/(theta_a[a])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #【1，0】时策略
                            else:
                                prr1=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，1】
                        else:
                            if start_c[j]==1:
                                prr1=(fda*q/(1-theta_a[a])+pda-cda)*(1-beta)    #为【0，1】时策略
                            else:
                                prr1=(fda*q+pda-cda)*(1-beta)   #策略为【0，0】
                        ctpr_all_a[a,0,i,j]=prr1
                count_0=0
                o_sum=0
                for i in range(N):
                    if start_c[i]==1:
                        count_0+=1
                        o_sum+=sum(pr_all_a[a,0,i,:])
                xt_per_a[a].append(o_sum/count_0)     #求初始时刻策略1的总收益
                theta_a[a]=0.3
            
            for a1 in range(3):         #a1
                aaerfa1=[0.05,-0.1*theta_a1[1]+0.1,-0.3*theta_a1[2]*theta_a1[2]+0.3*theta_a1[2]]
                aerfa=0.2
                aerfa1=aaerfa[a1]
                aerfa2=0.25
                yita=1
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if start_c[i]==0:
                            if start_c[j]==0:
                                prr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                            else:
                                prr0=(fda*q/(1-theta_a1[a1])+pda-cda)*(1-beta)   #策略为【0，1】
                        else:
                            if start_c[j]==1:
                                prr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)     #都为1时策略
                            else:
                                prr0=((fdi+b*yita)*q/(theta_a1[a1])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                        pr_all_a1[a1,0,i,j]=prr0
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if start_c[i]==0:
                            if start_c[j]==0:
                                prr1=((fdi+b*yita)*q/(theta_a1[a1])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #【1，0】时策略
                            else:
                                prr1=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，1】
                        else:
                            if start_c[j]==1:
                                prr1=(fda*q/(1-theta_a1[a1])+pda-cda)*(1-beta)    #为【0，1】时策略
                            else:
                                prr1=(fda*q+pda-cda)*(1-beta)   #策略为【0，0】
                        ctpr_all_a1[a1,0,i,j]=prr1
                count_0=0
                o_sum=0
                for i in range(N):
                    if start_c[i]==1:
                        count_0+=1
                        o_sum+=sum(pr_all_a1[a1,0,i,:])
                xt_per_a1[a1].append(o_sum/count_0)     #求初始时刻策略1的总收益
                theta_a1[a1]=0.3
            
            for a2 in range(3):   #a2
                aaerfa2=[0.25,-0.5*theta_a2[1]+0.5,-1.5*theta_a2[2]*theta_a2[2]+1.5*theta_a2[2]] 
                aerfa=0.2
                aerfa1=0.05
                aerfa2=aaerfa2[a2]
                yita=1
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if start_c[i]==0:
                            if start_c[j]==0:
                                prr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                            else:
                                prr0=(fda*q/(1-theta_a2[a2])+pda-cda)*(1-beta)   #策略为【0，1】
                        else:
                            if start_c[j]==1:
                                prr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)     #都为1时策略
                            else:
                                prr0=((fdi+b*yita)*q/(theta_a2[a2])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                        pr_all_a2[a2,0,i,j]=prr0
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if start_c[i]==0:
                            if start_c[j]==0:
                                prr1=((fdi+b*yita)*q/(theta_a2[a2])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #【1，0】时策略
                            else:
                                prr1=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，1】
                        else:
                            if start_c[j]==1:
                                prr1=(fda*q/(1-theta_a2[a2])+pda-cda)*(1-beta)    #为【0，1】时策略
                            else:
                                prr1=(fda*q+pda-cda)*(1-beta)   #策略为【0，0】
                        ctpr_all_a2[a2,0,i,j]=prr1
                count_0=0
                o_sum=0     
                for i in range(N):
                    if start_c[i]==1:
                        count_0+=1
                        o_sum+=sum(pr_all_a2[a2,0,i,:])
                xt_per_a2[a2].append(o_sum/count_0)      #求初始时刻策略1的总收益
                theta_a2[a2]=0.3
        
            for y in range(3):   #yita
                yyita=[1,-2*theta_y[1]+2,-6*theta_y[2]*theta_y[2]+6*theta_y[2]]
                aerfa=0.2
                aerfa1=0.05
                aerfa2=0.25
                yita=yyita[y]
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if start_c[i]==0:
                            if start_c[j]==0:
                                prr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                            else:
                                prr0=(fda*q/(1-theta_y[y])+pda-cda)*(1-beta)   #策略为【0，1】
                        else:
                            if start_c[j]==1:
                                prr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)     #都为1时策略
                            else:
                                prr0=((fdi+b*yita)*q/(theta_y[y])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                        pr_all_y[y,0,i,j]=prr0
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if start_c[i]==0:
                            if start_c[j]==0:
                                prr1=((fdi+b*yita)*q/(theta_y[y])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #【1，0】时策略
                            else:
                                prr1=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，1】
                        else:
                            if start_c[j]==1:
                                prr1=(fda*q/(1-theta_y[y])+pda-cda)*(1-beta)    #为【0，1】时策略
                            else:
                                prr1=(fda*q+pda-cda)*(1-beta)   #策略为【0，0】
                        ctpr_all_y[y,0,i,j]=prr1
                count_0=0
                o_sum=0     
                for i in range(N):
                    if start_c[i]==1:
                        count_0+=1
                        o_sum+=sum(pr_all_y[y,0,i,:])
                xt_per_y[y].append(o_sum/count_0)      #求初始时刻策略1的总收益
                theta_y[y]=0.3
        else:
            G=nx.watts_strogatz_graph(N,3,0.5)
            A=nx.to_numpy_array(G)
            B_fit_a=[[],[],[]]
            B_fit_a1=[[],[],[]]
            B_fit_a2=[[],[],[]]
            B_fit_y=[[],[],[]]
            #更新前
            for i in range(N):
                aerfa=0.2
                aerfa1=0.05
                aerfa2=0.25
                yita=1
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                for j in B:
                    if game_theory_[t-1,i]==0:
                        if game_theory_[t-1,j]==0:
                            pr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                        else:
                            pr0=(fda*q/(1-theta_)+pda-cda)*(1-beta)   #策略为【0，1】
                    else:
                        if game_theory_[t-1,i]==1:
                            pr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)      #都为1时策略
                        else:
                            pr0=((fdi+b*yita)*q/(theta_)+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                    pr_all_[t,i,j]=pr0      #该时刻下所有点与相应邻居节点的自身收益矩阵
            for i in range(N):
                i_total_=sum(pr_all_[t,i,:])
                PP=random.choice(B)
                fit_total=sum(pr_all_[t,PP,:])
                proba_lv=1/(1+np.exp(i_total_-fit_total)/k)
                if np.random.random() < proba_lv:
                    game_theory_[t,i]=game_theory_[t-1,PP]
                else:
                    game_theory_[t,i]=game_theory_[t-1,i]
            x_per_.append(sum(game_theory_[t,:])/len(game_theory_[t,:]))   #策略2占比
            t_sum=0
            count_t=0
            theta_=x_per_[t]
            for i in range(N):
                if game_theory_[t,i]==1:
                    count_t+=1
                    t_sum+=sum(pr_all_[t,i,:])
            xt_per_.append(t_sum/count_0)     #平均收益
            
            #更新后
            for a in range(3):
                aaerfa=[0.2,-0.4*theta_a[1]+0.4,-1.6*theta_a[2]*theta_a[2]+1.6*theta_a[2]]
                aerfa=aaerfa[a]
                aerfa1=0.05
                aerfa2=0.25
                yita=1
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if game_theory_a[a,t-1,i]==0:
                            if game_theory_a[a,t-1,j]==0:
                                pr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                            else:
                                pr0=(fda*q/(1-theta_a[a])+pda-cda)*(1-beta)   #策略为【0，1】
                        else:
                            if game_theory_a[a,t-1,i]==1:
                                pr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)      #都为1时策略
                            else:
                                pr0=((fdi+b*yita)*q/(theta_a[a])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                        pr_all_a[a,t,i,j]=pr0      #该时刻下所有点与相应邻居节点的自身收益矩阵
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1]) 
                    ctpr_all_a=pr_all_a
                    for j in B:
                        if game_theory_a[a,t-1,i]==0:
                            if game_theory_a[a,t-1,j]==0:
                                cst_prr=((fdi+b*yita)*q/(theta_a[a])+pdi-cdi-tz)*beta   #[1,0]，该节点的相反策略的收益
                                cst_prc=(fda*q/(1-theta_a[a])+pda-cda)*(1-beta)   #该节点相反策略下的邻居节点的收益
                            else:
                                cst_prr=((fdi+b*yita)*q+pdi-cdi-tz)*beta  #[1,1]
                                cst_prc=((fdi+b*yita)*q+pdi-cdi-tz)*beta     #该节点相反策略下的邻居节点的收益
                        else:
                            if game_theory_a[a,t-1,i]==1:
                                cst_prr=(fda*q/(1-theta_a[a])+pda-cda)*(1-beta)  #[0,1]时策略
                                cst_prc=((fdi+b*yita)*q/(theta_a[a])+pdi-cdi-tz)*beta     #该节点相反策略下的邻居节点的收益
                            else:
                                cst_prr=(fda*q+pda-cda)*(1-beta)   #[0,0]时策略
                                cst_prc=(fda*q+pda-cda)*(1-beta)   #该节点相反策略下的邻居节点的收益
                        ctpr_all_a[a,t,i,j]=cst_prr      #该时刻下所有点与相应邻居节点的自身收益矩阵
                        ctpr_all_a[a,t,j,i]=cst_prc
                    i_total_a=sum(pr_all_a[a,t,i,:])
                    cti_total_a=sum(ctpr_all_a[a,t,i,:])
                    B_fit_a=[]
                    for q in B:
                        fit_pr_a=sum(pr_all_a[a,t,q,:])-i_total_a    #收益差
                        fit_ctpr_a=sum(ctpr_all_a[a,t,q,:])-cti_total_a
                        if fit_pr_a>=fit_ctpr_a:
                            B_fit_a.append(fit_pr_a)
                        else:
                            B_fit_a.append(fit_ctpr_a)
                    fit_bB=B[B_fit_a.index(min(B_fit_a))]
                    fit_total=sum(pr_all_a[a,t,fit_bB,:])
                    proba_lv=1/(1+np.exp(min(B_fit_a))/k)
                    RAM=np.random.random()
                    if sum(pr_all_a[a,t,fit_bB,:])-i_total_a >= sum(ctpr_all_a[a,t,fit_bB,:])-cti_total_a:
                        game_theory_a=game_theory_a
                        if RAM < proba_lv:
                            game_theory_a[a,t,i]=game_theory_a[a,t-1,fit_bB]
                        else:
                            game_theory_a[a,t,i]=game_theory_a[a,t-1,i]
                    else:
                        game_theory_a[a,t,i]=1-game_theory_a[a,t-1,i]
                        if np.random.random() < proba_lv:
                            game_theory_a[a,t,i]=game_theory_a[a,t-1,fit_bB]
                        else:
                            game_theory_a[a,t,i]=game_theory_a[a,t,i]
                x_per_a[a].append(sum(game_theory_a[a,t,:])/len(game_theory_a[a,t,:]))   #策略2占比
                t_sum=0
                theta_a[a]=x_per_a[a][t]
                count_t=0
                for i in range(N):
                    if game_theory_a[a,t,i]==1:
                        count_t+=1
                        t_sum+=sum(pr_all_a[a,t,i,:])
                xt_per_a[a].append(t_sum/count_0)     #平均收益
                
            for a1 in range(3):
                aaerfa1=[0.05,-0.1*theta_a1[1]+0.1,-0.4*theta_a1[2]*theta_a1[2]+0.4*theta_a1[2]]
                aerfa1=aaerfa1[a1]
                aerfa=0.2
                aerfa2=0.25
                yita=1
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if game_theory_a1[a1,t-1,i]==0:
                            if game_theory_a1[a1,t-1,j]==0:
                                pr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                            else:
                                pr0=(fda*q/(1-theta_a1[a1])+pda-cda)*(1-beta)   #策略为【0，1】
                        else:
                            if game_theory_a1[a1,t-1,i]==1:
                                pr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)      #都为1时策略
                            else:
                                pr0=((fdi+b*yita)*q/(theta_a1[a1])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                        pr_all_a1[a1,t,i,j]=pr0      #该时刻下所有点与相应邻居节点的自身收益矩阵
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1]) 
                    ctpr_all_a1=pr_all_a1
                    for j in B:
                        if game_theory_a1[a1,t-1,i]==0:
                            if game_theory_a1[a1,t-1,j]==0:
                                cst_prr=((fdi+b*yita)*q/(theta_a1[a1])+pdi-cdi-tz)*beta   #[1,0]，该节点的相反策略的收益
                                cst_prc=(fda*q/(1-theta_a1[a1])+pda-cda)*(1-beta)   #该节点相反策略下的邻居节点的收益
                            else:
                                cst_prr=((fdi+b*yita)*q+pdi-cdi-tz)*beta  #[1,1]
                                cst_prc=((fdi+b*yita)*q+pdi-cdi-tz)*beta     #该节点相反策略下的邻居节点的收益
                        else:
                            if game_theory_a1[a1,t-1,i]==1:
                                cst_prr=(fda*q/(1-theta_a1[a1])+pda-cda)*(1-beta)  #[0,1]时策略
                                cst_prc=((fdi+b*yita)*q/(theta_a1[a1])+pdi-cdi-tz)*beta     #该节点相反策略下的邻居节点的收益
                            else:
                                cst_prr=(fda*q+pda-cda)*(1-beta)   #[0,0]时策略
                                cst_prc=(fda*q+pda-cda)*(1-beta)   #该节点相反策略下的邻居节点的收益
                        ctpr_all_a1[a1,t,i,j]=cst_prr      #该时刻下所有点与相应邻居节点的自身收益矩阵
                        ctpr_all_a1[a1,t,j,i]=cst_prc
                    i_total_a1=sum(pr_all_a1[a1,t,i,:])
                    cti_total_a1=sum(ctpr_all_a1[a1,t,i,:])
                    B_fit_a1=[]
                    for q in B:
                        fit_pr_a1=sum(pr_all_a1[a1,t,q,:])-i_total_a1    #收益差
                        fit_ctpr_a1=sum(ctpr_all_a1[a1,t,q,:])-cti_total_a1
                        if fit_pr_a1>=fit_ctpr_a1:
                            B_fit_a1.append(fit_pr_a1)
                        else:
                            B_fit_a1.append(fit_ctpr_a1)
                    fit_bB=B[B_fit_a1.index(min(B_fit_a1))]
                    if sum(pr_all_a1[a1,t,fit_bB,:])-i_total_a1 >= sum(ctpr_all_a1[a1,t,fit_bB,:])-cti_total_a1:
                        game_theory_a1=game_theory_a1
                    else:
                        game_theory_a1[a1,t-1,i]=1-game_theory_a1[a1,t-1,i]
                    fit_total=sum(pr_all_a1[a1,t,fit_bB,:])
                    fit_total=sum(pr_all_a1[a1,t,fit_bB,:])
                    proba_lv=1/(1+np.exp(min(B_fit_a1))/k)
                    RAM=np.random.random()
                    if sum(pr_all_a1[a1,t,fit_bB,:])-i_total_a1 >= sum(ctpr_all_a1[a1,t,fit_bB,:])-cti_total_a1:
                        game_theory_a1=game_theory_a1
                        if RAM < proba_lv:
                            game_theory_a1[a1,t,i]=game_theory_a1[a1,t-1,fit_bB]
                        else:
                            game_theory_a1[a1,t,i]=game_theory_a1[a1,t-1,i]
                    else:
                        game_theory_a1[a1,t,i]=1-game_theory_a1[a1,t-1,i]
                        if RAM < proba_lv:
                            game_theory_a1[a1,t,i]=game_theory_a1[a1,t-1,fit_bB]
                        else:
                            game_theory_a1[a1,t,i]=game_theory_a1[a1,t,i]
                x_per_a1[a1].append(sum(game_theory_a1[a1,t,:])/len(game_theory_a1[a1,t,:]))   #策略2占比
                t_sum=0
                count_t=0
                theta_a1[a1]=x_per_a1[a1][t]
                for i in range(N):
                    if game_theory_a1[a1,t,i]==1:
                        count_t+=1
                        t_sum+=sum(pr_all_a1[a1,t,i,:])
                xt_per_a1[a1].append(t_sum/count_0)     #平均收益
                
            for a2 in range(3):
                aaerfa2=[0.25,-0.5*theta_a2[1]+0.5,-2*theta_a2[2]*theta_a2[2]+2*theta_a2[2]]
                aerfa2=aaerfa2[a2]
                aerfa=0.2
                aerfa1=0.05
                yita=1
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if game_theory_a2[a2,t-1,i]==0:
                            if game_theory_a2[a2,t-1,j]==0:
                                pr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                            else:
                                pr0=(fda*q/(1-theta_a2[a2])+pda-cda)*(1-beta)   #策略为【0，1】
                        else:
                            if game_theory_a2[a2,t-1,i]==1:
                                pr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)      #都为1时策略
                            else:
                                pr0=((fdi+b*yita)*q/(theta_a2[a2])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                        pr_all_a2[a2,t,i,j]=pr0      #该时刻下所有点与相应邻居节点的自身收益矩阵
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1]) 
                    ctpr_all_a2=pr_all_a2
                    for j in B:
                        if game_theory_a2[a2,t-1,i]==0:
                            if game_theory_a2[a2,t-1,j]==0:
                                cst_prr=((fdi+b*yita)*q/(theta_a2[a2])+pdi-cdi-tz)*beta   #[1,0]，该节点的相反策略的收益
                                cst_prc=(fda*q/(1-theta_a2[a2])+pda-cda)*(1-beta)   #该节点相反策略下的邻居节点的收益
                            else:
                                cst_prr=((fdi+b*yita)*q+pdi-cdi-tz)*beta  #[1,1]
                                cst_prc=((fdi+b*yita)*q+pdi-cdi-tz)*beta     #该节点相反策略下的邻居节点的收益
                        else:
                            if game_theory_a2[a2,t-1,i]==1:
                                cst_prr=(fda*q/(1-theta_a2[a2])+pda-cda)*(1-beta)  #[0,1]时策略
                                cst_prc=((fdi+b*yita)*q/(theta_a2[a2])+pdi-cdi-tz)*beta     #该节点相反策略下的邻居节点的收益
                            else:
                                cst_prr=(fda*q+pda-cda)*(1-beta)   #[0,0]时策略
                                cst_prc=(fda*q+pda-cda)*(1-beta)   #该节点相反策略下的邻居节点的收益
                        ctpr_all_a2[a2,t,i,j]=cst_prr      #该时刻下所有点与相应邻居节点的自身收益矩阵
                        ctpr_all_a2[a2,t,j,i]=cst_prc
                    i_total_a2=sum(pr_all_a2[a2,t,i,:])
                    cti_total_a2=sum(ctpr_all_a2[a2,t,i,:])
                    B_fit_a2=[]
                    for q in B:
                        fit_pr_a2=sum(pr_all_a2[a2,t,q,:])-i_total_a2    #收益差
                        fit_ctpr_a2=sum(ctpr_all_a2[a2,t,q,:])-cti_total_a2
                        if fit_pr_a2>=fit_ctpr_a2:
                            B_fit_a2.append(fit_pr_a2)
                        else:
                            B_fit_a2.append(fit_ctpr_a2)
                    fit_bB=B[B_fit_a2.index(min(B_fit_a2))]
                    fit_total=sum(pr_all_a2[a2,t,fit_bB,:])
                    proba_lv=1/(1+np.exp(min(B_fit_a2))/k)
                    RAM=np.random.random()
                    if sum(pr_all_a2[a2,t,fit_bB,:])-i_total_a2 >= sum(ctpr_all_a2[a2,t,fit_bB,:])-cti_total_a2:
                        game_theory_a2=game_theory_a2
                        if RAM < proba_lv:
                            game_theory_a2[a2,t,i]=game_theory_a2[a2,t-1,fit_bB]
                        else:
                            game_theory_a2[a2,t,i]=game_theory_a2[a2,t-1,i]
                    else:
                        game_theory_a2[a2,t,i]=1-game_theory_a2[a2,t-1,i]
                        if RAM < proba_lv:
                            game_theory_a2[a2,t,i]=game_theory_a2[a2,t-1,fit_bB]
                        else:
                            game_theory_a2[a2,t,i]=game_theory_a2[a2,t,i]
                x_per_a2[a2].append(sum(game_theory_a2[a2,t,:])/len(game_theory_a2[a2,t,:]))   #策略2占比
                t_sum=0
                count_t=0
                theta_a2[a2]=x_per_a2[a2][t]
                for i in range(N):
                    if game_theory_a2[a2,t,i]==1:
                        count_t+=1
                        t_sum+=sum(pr_all_a2[a2,t,i,:])
                xt_per_a2[a2].append(t_sum/count_0)     #平均收益
            
            for y in range(3):
                yyita=[1,-2*theta_y[1]+2,-8*theta_y[2]*theta_y[2]+8*theta_y[2]]
                aerfa2=0.25
                aerfa=0.2
                aerfa1=0.05
                yita=yyita[y]
                cdi=cda*(1-aerfa2)
                fdi=fda*(1+aerfa)
                pdi=pda*(1+aerfa1)
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1])
                    for j in B:
                        if game_theory_y[y,t-1,i]==0:
                            if game_theory_y[y,t-1,j]==0:
                                pr0=(fda*q+pda-cda)*(1-beta)    #都为0时策略
                            else:
                                pr0=(fda*q/(1-theta_y[y])+pda-cda)*(1-beta)   #策略为【0，1】
                        else:
                            if game_theory_y[y,t-1,i]==1:
                                pr0=((fdi+b*yita)*q+pdi-cdi-tz)*beta+(1-beta)*(-tz)      #都为1时策略
                            else:
                                pr0=((fdi+b*yita)*q/(theta_y[y])+pdi-cdi-tz)*beta+(1-beta)*(-tz)    #策略为【1，0】
                        pr_all_y[y,t,i,j]=pr0      #该时刻下所有点与相应邻居节点的自身收益矩阵
                for i in range(N):
                    B=np.array([u for (u, v) in enumerate(A[i,:]) if v == 1]) 
                    ctpr_all_y=pr_all_y
                    for j in B:
                        if game_theory_y[y,t-1,i]==0:
                            if game_theory_y[y,t-1,j]==0:
                                cst_prr=((fdi+b*yita)*q/(theta_y[y])+pdi-cdi-tz)*beta   #[1,0]，该节点的相反策略的收益
                                cst_prc=(fda*q/(1-theta_y[y])+pda-cda)*(1-beta)   #该节点相反策略下的邻居节点的收益
                            else:
                                cst_prr=((fdi+b*yita)*q+pdi-cdi-tz)*beta  #[1,1]
                                cst_prc=((fdi+b*yita)*q+pdi-cdi-tz)*beta     #该节点相反策略下的邻居节点的收益
                        else:
                            if game_theory_y[y,t-1,i]==1:
                                cst_prr=(fda*q/(1-theta_y[y])+pda-cda)*(1-beta)  #[0,1]时策略
                                cst_prc=((fdi+b*yita)*q/(theta_y[y])+pdi-cdi-tz)*beta     #该节点相反策略下的邻居节点的收益
                            else:
                                cst_prr=(fda*q+pda-cda)*(1-beta)   #[0,0]时策略
                                cst_prc=(fda*q+pda-cda)*(1-beta)   #该节点相反策略下的邻居节点的收益
                        ctpr_all_y[y,t,i,j]=cst_prr      #该时刻下所有点与相应邻居节点的自身收益矩阵
                        ctpr_all_y[y,t,j,i]=cst_prc
                    i_total_y=sum(pr_all_y[y,t,i,:])
                    cti_total_y=sum(ctpr_all_y[y,t,i,:])
                    B_fit_y=[]
                    for q in B:
                        fit_pr_y=sum(pr_all_y[y,t,q,:])-i_total_y    #收益差
                        fit_ctpr_y=sum(ctpr_all_y[y,t,q,:])-cti_total_y
                        if fit_pr_y>=fit_ctpr_y:
                            B_fit_y.append(fit_pr_y)
                        else:
                            B_fit_y.append(fit_ctpr_y)
                    fit_bB=B[B_fit_y.index(min(B_fit_y))]
                    fit_total=sum(pr_all_y[y,t,fit_bB,:])
                    proba_lv=1/(1+np.exp(min(B_fit_y))/k)
                    RAM=np.random.random()
                    if sum(pr_all_y[y,t,fit_bB,:])-i_total_y >= sum(ctpr_all_y[y,t,fit_bB,:])-cti_total_y:
                        game_theory_y=game_theory_y
                        if RAM < proba_lv:
                            game_theory_y[y,t,i]=game_theory_y[y,t-1,fit_bB]
                        else:
                            game_theory_y[y,t,i]=game_theory_y[y,t-1,i]
                    else:
                        game_theory_y[y,t,i]=1-game_theory_y[y,t-1,i]
                        if RAM < proba_lv:
                            game_theory_y[y,t,i]=game_theory_y[y,t-1,fit_bB]
                        else:
                            game_theory_y[y,t,i]=game_theory_y[y,t,i]
                x_per_y[y].append(sum(game_theory_y[y,t,:])/len(game_theory_y[y,t,:]))   #策略2占比
                t_sum=0
                count_t=0
                theta_y[y]=x_per_y[y][t]
                for i in range(N):
                    if game_theory_y[y,t,i]==1:
                        count_t+=1
                        t_sum+=sum(pr_all_y[y,t,i,:])
                xt_per_y[y].append(t_sum/count_0)     #平均收益
                
    x_per_all_=np.array(x_per_)+np.array(x_per_all_)
    x_per_all_a=np.array(x_per_a)+np.array(x_per_all_a)
    x_per_all_a1=np.array(x_per_a1)+np.array(x_per_all_a1)
    x_per_all_a2=np.array(x_per_a2)+np.array(x_per_all_a2)
    x_per_all_y=np.array(x_per_y)+np.array(x_per_all_y)
    xt_per_all_=np.array(xt_per_)+np.array(xt_per_all_)
    xt_per_all_a=np.array(xt_per_a)+np.array(xt_per_all_a)
    xt_per_all_a1=np.array(xt_per_a1)+np.array(xt_per_all_a1)
    xt_per_all_a2=np.array(xt_per_a2)+np.array(xt_per_all_a2)
    xt_per_all_y=np.array(xt_per_y)+np.array(xt_per_all_y)
x_per_avg_=x_per_all_/num
x_per_avg_a=x_per_all_a/num
x_per_avg_a1=x_per_all_a1/num
x_per_avg_a2=x_per_all_a2/num
x_per_avg_y=x_per_all_y/num
xt_per_avg_=xt_per_all_/num
xt_per_avg_a=xt_per_all_a/num
xt_per_avg_a1=xt_per_all_a1/num
xt_per_avg_a2=xt_per_all_a2/num
xt_per_avg_y=xt_per_all_y/num
'''
x_per_avg_=list(x_per_avg_)
x_per_avg_a=list(x_per_avg_a)
x_per_avg_a1=list(x_per_avg_a1)
x_per_avg_a2=list(x_per_avg_a2)
x_per_avg_y=list(x_per_avg_y)
xt_per_avg_=list(xt_per_avg_)
xt_per_avg_a=list(xt_per_avg_a)
xt_per_avg_a1=list(xt_per_avg_a1)
xt_per_avg_a2=list(xt_per_avg_a2)
xt_per_avg_y=list(xt_per_avg_y)
'''
print(np.mean(x_per_avg_))
print(np.mean(xt_per_avg_))
for i in range(3):
    print(np.mean(x_per_avg_a[i]),np.std(x_per_avg_a[i]))
    print(np.mean(x_per_avg_a1[i]),np.std(x_per_avg_a1[i]))
    print(np.mean(x_per_avg_a2[i]),np.std(x_per_avg_a2[i]))
    print(np.mean(x_per_avg_y[i]),np.std(x_per_avg_y[i]))
    print(np.mean(xt_per_avg_a[i]),np.std(xt_per_avg_a[i]))
    print(np.mean(xt_per_avg_a1[i]),np.std(xt_per_avg_a[i]))
    print(np.mean(xt_per_avg_a2[i]),np.std(xt_per_avg_a[i]))
    print(np.mean(xt_per_avg_y[i]),np.std(xt_per_avg_a[i]))


X = np.array(list(range(T)))
tu=['k--o','k--s','k--D']
label_a1=['α1=0.2','α1 is linear','α1 is nonlinear']
label_a2=['α2=0.05','α2 is linear','α2 is nonlinear']
label_a3=['α3=0.25','α3 is linear','α3 is nonlinear']
label_y=['η=0.25','η is linear','η is is nonlinear']
#tu=['k-+','k-.','k-x','k-o','k-_','k-|']
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(0)
for i in range(3):
    plt.plot(X,x_per_avg_a[i],tu[i],markerfacecolor='none',label=label_a1[i])
#plt.plot(X,y0,'k-+')         # 策略2占比
plt.legend()
fig = plt.figure(1)
for i in range(3):
    plt.plot(X,xt_per_avg_a[i][:],tu[i],markerfacecolor='none',label=label_a1[i])
#plt.plot(X[1:],y1[1:],'k--v')    #平均收益
plt.legend()
fig = plt.figure(2)
for i in range(3):
    plt.plot(X,x_per_avg_a1[i],tu[i],markerfacecolor='none',label=label_a2[i])
#plt.plot(X,y0,'k-+')         # 策略2占比
plt.legend()
fig = plt.figure(3)
for i in range(3):
    plt.plot(X,xt_per_avg_a1[i][:],tu[i],markerfacecolor='none',label=label_a2[i])
#plt.plot(X[1:],y1[1:],'k--v')    #平均收益
plt.legend()
fig = plt.figure(4)
for i in range(3):
    plt.plot(X,x_per_avg_a2[i],tu[i],markerfacecolor='none',label=label_a3[i])
#plt.plot(X,y0,'k-+')         # 策略2占比
plt.legend()
fig = plt.figure(5)
for i in range(3):
    plt.plot(X,xt_per_avg_a2[i][:],tu[i],markerfacecolor='none',label=label_a3[i])
plt.legend()
fig = plt.figure(6)
for i in range(3):
    plt.plot(X,x_per_avg_y[i],tu[i],markerfacecolor='none',label=label_y[i])
#plt.plot(X,y0,'k-+')         # 策略2占比
plt.legend()
fig = plt.figure(7)
for i in range(3):
    plt.plot(X,xt_per_avg_y[i][:],tu[i],markerfacecolor='none',label=label_y[i])
plt.legend()
fig=plt.figure(8)
#plt.plot(X[1:],xt_per_a1[1][1:],tu[1])
plt.plot(X,xt_per_avg_[:],tu[1],markerfacecolor='none',label='Before updating')
plt.plot(X,xt_per_avg_a[0][:],tu[2],markerfacecolor='none',label='After updating')
plt.legend()
fig=plt.figure(9)
#plt.plot(X[1:],xt_per_a1[1][1:],tu[1])
plt.plot(X,x_per_avg_,tu[1],markerfacecolor='none',label='Before updating')
plt.plot(X,x_per_avg_a[0],tu[2],markerfacecolor='none',label='After updating')
plt.legend()
plt.show()
'''
renwu jieshu
'''       