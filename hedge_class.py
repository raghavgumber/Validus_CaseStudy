# -*- coding: utf-8 -*-
"""
Created on Sat May 22 19:54:42 2021

@author: Raghav Gumber
"""

import scipy
import numpy as np
import pandas as pd
import math
import matplotlib.ticker as mtick

class hedge_pricer():
    '''
    Customized pricing class for the project that handles multi asset modelling
    
    '''
    
    def __init__(self,n_paths,n_assets,r_drifts,sigma_vec,spot,dt,total_time,CF,corr,seed=0):
        '''
        Args:
            n_paths: num paths
            n_assets: integer which is the num. assets to model
            r_drifts: list of floats contining drifts of each asset
            sigma_vec:list of floats contining vols of each asset
            spot:list of floats contining spots of each asset (like GBPUSD)
            dt: float which is delta of time for modelling MC
            total_time: num years to model out
                        corr: matrix of correlations for the multi assets, if only one asset, set =1
            seed: int for RNG, set to 0 default
        Returns:
            object instantiation
        '''
        self.n_paths=n_paths
        self.n_assets=n_assets
        self.r_drifts=r_drifts
        self.sigma_vec=sigma_vec
        self.spot=spot
        self.dt=dt
        self.total_time=total_time
        
        self.corr=corr
        self.seed=seed
        self.path=np.empty(0)
        
        self.time_steps=math.ceil(total_time/dt)
        #self.CF_paths=np.empty(0)
    def ret_paths(self):
        '''
        GBM MC path generator for the assets provided in instatiation
        Returns: array of shape (total_time,n_paths,n_assets), which is also the level of indexing of the paths
            
        '''
        scipy.random.seed(self.seed)
        lss=scipy.zeros((self.n_paths,self.n_assets))
        spots=self.spot*scipy.exp(lss)
        path=[[*spots]]
        for tStep in range(self.time_steps):
            sigma=np.array([[self.sigma_vec[i] for j in range(self.n_paths)] for i in range(self.n_assets)])
            z=scipy.random.normal(0,1,(self.n_paths,self.n_assets))
            z=scipy.dot(z,self.corr)

            for i in range(self.n_assets):
                lss[:,i]+=self.r_drifts[i]-sigma[i]*sigma[i]*self.dt/2+sigma[0]*z[:,0]*self.dt**.5
            spots=self.spot*scipy.exp(lss)
            path.append([*spots])
        self.path=np.array(path)
        return self.path
    def ret_CF(self,CF):
        '''
        Args:
            CF: dictionary where keys are 0...total_time, every value is CF for every asset in foreign ccy. 
            so {0:[100]} where asset = GBPUSD refers to 100 GBP recieved at time 0
        Returns: array of shape (total_time,n_paths,n_assets), which is also the level of indexing of the CFs per asset reported
        in the base currency. for example 100 GBP gets converted to (100*GBPUSD exchange) USD when returned here... i.e CF paths are in base
        currency
        '''
        if len(self.path)==0:
            raise ValueError("run paths first")
        else:
            CF_paths=np.stack((self.path[math.ceil(i/self.dt)]*CF[i] for i in sorted(CF.keys())))
            
            return CF_paths
    def solve_IRR(self,CF_scen):
        '''
        Args:
            CF_scen: list containing CFs indexed by time... example:[-100,10,110]
        Returns:
            Implied IRR for the scenario
        '''
        def NPV(CF_scen,IRR):
            return CF_scen.dot(np.array([scipy.exp(-IRR*t) for t in range(self.total_time+1)]))
        return scipy.optimize.minimize(lambda IRR: NPV(CF_scen,IRR)**2, 0).x[0]
    def option_calc_put(self,r_fwd,K,T,N,index=0):
        '''
        Args:
            r_fwd: float -> forward rate used for discounting, for example in the case of GBPUSD, the r_fwd=r_usd-r_GBP
            K=float-> strike
            T=int-> time length of p=options in years
            N: float -> notional in foreign CCY
            index: int -> index of asset to price in, for example 0 means first asset, in this case 0 refers to GBPUSD with respect to the question
        Returns:
            value of the option per the MC paths of the asset under consideration
            
        '''
        option=0
        for i in range(len(self.path.T[index])):
            scen=self.path.T[index][i]
            option_payoff=[0]*(T)+[max(K-scen[math.ceil(T/self.dt)],0)]
            option_payoff_disc=[math.exp(-r_fwd*i)*option_payoff[i] for i in range(T+1)]
            option+=sum(option_payoff_disc)
        return N*option/len(self.path.T[index])
            
    def plot_IRRs(self,IRRs,ax):
        dist=pd.Series(IRRs)
        dist.plot.kde(ax=ax, legend=False, title='Distribution of unhedged IRRs over the 1000 scenarios',ind=[0,1])
        dist.plot.hist(density=True, ax=ax)
        ax.set_ylabel('Probability')
        ax.grid(axis='y')
        ax.set_xlim([math.floor(min(IRRs)),math.ceil(max(IRRs))])
        v=0
        for percentile in [5,50,95]:
            v+=.01
            pct=np.percentile(IRRs, percentile)
            ax.axvline(x=pct, ymin=0, ymax=1,color='r',linestyle='--')
            ax.xaxis.set_major_formatter(mtick.PercentFormatter())

            ax.text(pct+.25, max(ax.get_yticks())/2+v, '{0} IRR is {1}th percentile'.format("{:.2%}".format(pct/100),percentile), fontsize=12)