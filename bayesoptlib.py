import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib as mpl
from matplotlib.pyplot import figure

import GPy # for Gaussian Process Regression
      
class constrained_bayesian_optimization:
    '''
    Constrained Bayesian Optimization Class:
    
            min_X {f0(X)|fi(X)>=0}
    
    Use the same class for the objective function and the constraints. 
        
    Requires GPy package: https://sheffieldml.github.io/GPy/
    written by: Dinesh Krishnamoorthy, July 2020
    '''
    
    def __init__(self,X_sample,Y_sample,bounds,kernel,
                            mf = None,
                            X_grid = np.linspace(0,1,100),
                            obj_fun = None,
                            t = 10):
        self.X_sample = X_sample
        self.Y_sample = Y_sample
        self.bounds = bounds
        self.kernel = kernel
        self.X_grid = X_grid
        self.grid_size = self.X_grid.shape[0]
        self.nX = self.X_sample.shape[1] 
        self.mf = mf
        self.constraint_method = 'PF'
        self.t = t
        self.epsilon = 0
        
        
    def fit_gp(self):
        # Fit GP model to the data
        if self.mf is not None:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel,mean_function = self.mf)
        else:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel)
        return self

    def optimize_fit(self):
        # Optimize hyperparameters of the GP model 
        return self.m.optimize()
  
    def query_next(self,acquisition='EI',constraint = None):
        # Optimize acquistion function to compute the next point to query
        self.acquisition = acquisition
        self.constraint = constraint
        nX = self.X_sample.shape[1]

        min_val = 500000#-1e-5
        min_x = self.X_sample[-1,:]
        n_restarts=25

        def min_obj(X):
            return self.Objective(X)

        for x0 in np.random.uniform(self.bounds[:, 0].T, self.bounds[:, 1].T,
                                        size=(n_restarts, nX)):
            res = minimize(min_obj, x0=x0.reshape(-1,self.nX),
                                bounds=self.bounds, method='L-BFGS-B')  
            if res.fun < min_val and res.success is True:
                assert res.success is True, "Error: Solver Failed!"      
                min_val = res.fun[0]
                min_x = res.x  
        
        self.res = res
        self.X_next =  min_x.reshape(-1, nX)
        self.min_val = min_val
        return self

    def Objective(self,X):
        # Unconstrained acqusition function
        if self.acquisition=='LCB' or self.acquisition=='UCB' :
            alpha = self.LCB(X)
        elif self.acquisition == 'EI':
            alpha = -self.EI(X)
        elif self.acquisition == 'PI':
            alpha = -self.PI(X)
        elif self.acquisition == 'greedy':
            alpha = self.greedy(X)
        
        if self.constraint is not None:
            assert type(self.constraint) is  list, "Constraint object must be of type list."

            if self.constraint_method is 'PF': # Compute probability of feasibility 
                PF = 1
                for i in range(len(self.constraint)):
                    PF = PF*self.PF(self.constraint[i],np.array(X).reshape(-1,self.nX))

                alpha = PF*alpha # unconstrained acq_fn scaled with the probability of feasibility  

            if self.constraint_method is 'Barrier': # Compute barrier terms
                Br = 0
                Br2 = 0  # For exploration: minimize -[Barrier(LCB) - Barrier(Mean)]
                for i in range(len(self.constraint)):
                    Br = Br + self.LogBarrier(self.constraint[i],np.array(X).reshape(-1,self.nX))
                    Br2 = Br2 + ((self.t*self.LogBarrier(self.constraint[i],np.array(X).reshape(-1,self.nX)) - self.t*self.LogBarrier(self.constraint[i],np.array(X).reshape(-1,self.nX),beta = 0))**2)
                p = np.random.uniform(0, 1, 1)
                if p < self.epsilon:
                    alpha = -Br2 # pure exploration
                else:
                    alpha = alpha + Br # - self.epsilon*Br2
                

            if self.constraint_method is 'Penalty':
                Pn = 0
                for i in range(len(self.constraint)):
                    Pn = Pn + self.Penalty(self.constraint[i],np.array(X).reshape(-1,1))
                alpha = alpha + Pn # unconstrained acq_fn with the barrier terms
        return alpha

    def LCB(self,X):
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        return mu - 2*sigma
    
    def EI(self,X,xi=0.01):
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        mu_sample,si = self.m.predict(self.X_sample)
        f_best = np.max(-mu_sample) # incumbent
        with np.errstate(divide='warn'):
            imp = -mu - f_best - xi
            Z = imp/sigma
            EI = (imp*norm.cdf(Z) + sigma*norm.pdf(Z))
            EI[sigma == 0.0] = 0.0
        return EI

    def PI(self,X,xi=0.01):
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        mu_sample,si = self.m.predict(self.X_sample)
        f_best = np.max(-mu_sample) # incumbent
        with np.errstate(divide='warn'):
            imp = -mu - f_best - xi
            Z = imp/sigma
            PI = norm.cdf(Z)
            PI[sigma == 0.0] = 0.0
        return PI
    
    def greedy(self,X):
        mu, sigma = self.m.predict(X.reshape(-1,self.nX))
        return mu

    def PF(self,constraint,X):
        mu, sigma = constraint.m.predict(X)
        return norm.cdf(0,-mu,sigma) # Probability of feasibility

    def LogBarrier(self,constraint,X,beta = -2):
        mu, sigma = constraint.m.predict(X)
        return -1/self.t*np.log(mu+beta*sigma)

    def Penalty(self,constraint,X,beta = -2):
        mu, sigma = constraint.m.predict(X)
        return -1*np.minimum(0,mu-2*sigma)

    def acq(self):
        if self.acquisition=='LCB' or self.acquisition=='UCB':
            self.acq_fn = self.LCB(self.X_grid)
        if self.acquisition=='EI':
            self.acq_fn = -self.EI(self.X_grid)
        if self.acquisition=='PI':
            self.acq_fn = -self.PI(self.X_grid)
        if self.acquisition == 'TS':
            self.acq_fn = self.posterior_sample
        if self.acquisition == 'greedy':
            self.acq_fn = self.greedy(self.X_grid)
        return self

    def plot(self,plot_acq = False,plot_PF = False,plot_BR = False,
                plot_ideal=False, bounds = None, acq_ylim = None,acq_xlim = None,
                ylim = None,
                ylabel= 'Objective $f(x)$', fig_name=''):
        
        if bounds is None:
            bounds = self.bounds

        assert self.X_sample.shape[1]==1, "X dimension must be 1 for this plot function"
        mu, sigma = self.m.predict(self.X_grid)
        if plot_acq:   
    
            fig = plt.figure(constrained_layout=True)  
            spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
            ax1 = fig.add_subplot(spec[0:2, 0])
            ax2 = fig.add_subplot(spec[2, 0])

            ax1.fill_between(self.X_grid.ravel(), 
                        mu.ravel() + 2 * sigma.ravel(), 
                        mu.ravel() - 2 * sigma.ravel(), 
                        alpha=0.1,label='Confidence') 
            ax1.plot(self.X_grid,mu+2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            ax1.plot(self.X_grid,mu-2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            ax1.plot(self.X_grid,mu,color=(0,0.4,0.7),linewidth=2.4,label='Mean')
            ax1.plot(self.X_sample,self.Y_sample,'kx',label='Data')
            if plot_ideal:
                ax1.plot(self.X_grid,self.obj_fun(self.X_grid),'--',color=(0.6,0.6,0.6),linewidth=1,label='Ground truth')
            ax1.set_xlabel('$x$')
            ax1.set_ylabel(ylabel)
            if ylim is not None:
                ax1.set_ylim(ylim)
            #ax1.legend()
            ax1.grid()

        
            acq_fn = self.Objective(self.X_grid) # get the acquisition function used
            ax2.plot(self.X_grid,acq_fn,color = (0,0.7,0,0.8),linewidth=2,label='constrained')
            if self.constraint is not None:
                self.acq() # get the unconstrained acquisition function
                ax2.plot(self.X_grid,self.acq_fn,'--',color = (0,0.7,0,0.4),linewidth=2,label='unconstrained')
            ax2.plot(self.X_next,self.min_val,'ro',label='$x_{next}$')
            ax2.set_ylabel('Acqusition fn')
            ax2.set_xlabel('$x$')
            if acq_ylim is not None:
                ax2.set_xlim(acq_xlim)
            else:
                ax2.set_xlim(bounds[0][0],bounds[0][1])
            if acq_ylim is not None:
                ax2.set_ylim(acq_ylim)
            ax2.grid()
            plt.show()
            
        if plot_PF or plot_BR is True:
            mu, sigma = self.m.predict(self.X_grid)
            PF_grid = norm.cdf(0,-mu,sigma) # Probability of feasibility
            Br_grid = -1/self.t*np.log(mu-2*sigma) 
            Br_grid0 = -1/self.t*np.log(mu)

            fig = plt.figure(constrained_layout=True,figsize=(4, 2.5))
            spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
            ax1 = fig.add_subplot(spec[0:2, 0])
            ax2 = fig.add_subplot(spec[2, 0])
            ax1.fill_between(self.X_grid.ravel(), 
                        mu.ravel() + 2 * sigma.ravel(), 
                        mu.ravel() - 2 * sigma.ravel(), 
                        alpha=0.1,label='Confidence') 
            ax1.plot(self.X_grid,mu+2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            ax1.plot(self.X_grid,mu-2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            ax1.plot(self.X_grid,mu,color=(0,0.4,0.7),linewidth=2.4,label='Mean')
            ax1.plot(self.X_sample,self.Y_sample,'kx',label='Data')
            ax1.set_xlim(bounds[0][0],bounds[0][1])
            ax1.set_ylabel(ylabel)
            if ylim is not None:
                ax1.set_ylim(ylim)
            if plot_ideal:
                ax1.plot(self.X_grid,self.obj_fun(self.X_grid),'--',color=(0.6,0.6,0.6),linewidth=1,label='Ground truth')
            if plot_PF is True:
                ax2.plot(self.X_grid,PF_grid,color = (0,0.7,0,0.8),linewidth=2)
                ax2.set_xlabel('$x$')
                ax2.set_ylabel('$\Phi(x)$')
                ax2.tick_params(axis='y', labelcolor=(0,0.5,0,0.8))
            if plot_BR is True:
                ax2.plot(self.X_grid,Br_grid,color = (0.7,0.0,0,0.8),linewidth=2)
                ax2.fill_between(self.X_grid.ravel(), 
                        Br_grid.ravel(), 
                        Br_grid0.ravel(), 
                        color = (0.7,0.0,0,0.8),
                        alpha=0.2,label='Confidence') 
                ax2.set_xlim(bounds[0][0],bounds[0][1])
                ax2.set_ylabel('$\mathcal{B}_{\\beta}(x)$')
                ax2.set_yticks([])
            
            plt.show()

        if plot_PF is False and plot_BR is False and plot_acq is False:
            fig = plt.figure(constrained_layout=True,figsize=(4, 2))
            plt.fill_between(self.X_grid.ravel(), 
                        mu.ravel() + 2 * sigma.ravel(), 
                        mu.ravel() - 2 * sigma.ravel(), 
                        color = (0.0,0.5,0,0.8),
                        alpha=0.1,label='Confidence') 
            plt.plot(self.X_grid,mu+2*sigma,color=(0,0.5,0.0,0.1),linewidth=0.3)
            plt.plot(self.X_grid,mu-2*sigma,color=(0,0.5,0.0,0.1),linewidth=0.3)
            plt.plot(self.X_grid,mu,color=(0,0.5,0.0),linewidth=2.4,label='Mean')
            plt.plot(self.X_sample,self.Y_sample,'kx',label='Data')
            if plot_ideal:
                plt.plot(self.X_grid,self.obj_fun(self.X_grid),'--',color=(0.6,0.6,0.6),linewidth=1,label='Ground truth')
            plt.xlabel('$x$')
            plt.ylabel(ylabel)
            plt.ylim(ylim)
            #plt.legend()
            #plt.grid()
            plt.show()
            
        if fig_name:
            fig.savefig(fig_name+'.pdf',bbox_inches='tight')

    def plot_2d(self,x1_range,x2_range,projection = '2d', beta = 0, bounds = None, xlim = None,
                ylim = None, xlabel = '$x_1$',ylabel= ' $x_2$', zlabel = '$f(x_1,x_2)$', fig_name=''):
            
        n1 = x1_range.shape[0]
        n2 = x2_range.shape[0]
        X1,X2 = np.meshgrid(x1_range,x2_range)
        X_grid = np.vstack((X1.flatten(), X2.flatten())).T
        mu, sigma = self.m.predict(X_grid)
        Z = mu.reshape(n1,n2)
        Z2 = mu.reshape(n1,n2) - 2*sigma.reshape(n1,n2)
        fig = plt.figure()
        if projection == '2d':
            plt.plot(self.X_sample[:,0], self.X_sample[:,1],'ko',label='Data')
            plt.contour(X1, X2, Z,cmap=cm.coolwarm)
            #plt.plot(action_space.reshape(self.grid_size,), C_now.reshape(self.grid_size,),'--',color=(0.4,0.4,0.4),label='Current context')
            #plt.plot(self.X_next[:,0], self.context,'rv',label='$x_{next}$')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        elif projection == '3d':
            ax = fig.gca(projection='3d')
            ax.scatter(self.X_sample[:,0], self.X_sample[:,1], self.Y_sample,label='Data')
            
            ax.plot_surface(X1,X2,Z,rstride=8, cstride=8, alpha=0.3,label='Posterior mean')
            ax.plot_surface(X1,X2,Z2,rstride=8, cstride=8, alpha=0.2,label='Posterior mean')
            
            cset = ax.contour(X1,X2,Z, zdir='z', offset=np.min(Z)-1, cmap=cm.coolwarm)
         
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
        
      
class contextual_bayesian_optimization:
    '''
    Contextual Constrained Bayesian Optimization Class:
    
            min_X {f0(X,d)|fi(X,d)>=0} for a given context d
    
    Use the same class for the objective function and the constraints. 
        
    Requires GPy package: https://sheffieldml.github.io/GPy/
    written by: Dinesh Krishnamoorthy, July 2020
    '''
    
    def __init__(self,X_sample,Y_sample,context,bounds,kernel,
                            mf = None,
                            X_grid = np.linspace(0,1,100),
                            obj_fun = None):
        self.X_sample = X_sample
        self.Y_sample = Y_sample
        self.context = context  # New context
        self.bounds = bounds
        self.kernel = kernel
        self.X_grid = X_grid
        self.grid_size = self.X_grid.shape[0]
        self.nX = self.X_sample.shape[1] 
        self.nC = self.context.shape[1]
        self.nU = self.nX - self.nC
        self.mf = mf
        self.constraint_method = 'None'
        self.epsilon = 0
        
        
    def fit_gp(self):
        # Fit GP model to the data
        if self.mf is not None:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel,mean_function = self.mf)
        else:
            self.m = GPy.models.GPRegression(self.X_sample,self.Y_sample,self.kernel)
        return self

    def optimize_fit(self):
        # Optimize hyperparameters of the GP model 
        return self.m.optimize()
  
    def extract_action_space(self): # Not used... delete?
        nU = self.nX-self.nC
        return np.array(self.X_sample[:,0:nU+1]).reshape(-1,nU)

    def query_next(self,acquisition='EI',constraint = None):
        # Optimize acquistion function to compute the next point to query
        self.acquisition = acquisition
        self.constraint = constraint
        nX = self.X_sample.shape[1]

        min_val = 5000
        min_x = self.X_sample[-1][0:self.nU]#self.X_sample[-1,:]
        n_restarts=25

        def min_obj(X):
            return self.Objective(X)

        for x0 in np.random.uniform(self.bounds[:, 0].T, self.bounds[:, 1].T,
                                        size=(n_restarts, self.nU)):
            res = minimize(min_obj, x0=x0.reshape(self.nU,),
                                bounds=self.bounds, method='L-BFGS-B')  
            if res.fun < min_val and res.success is True:
                assert res.success is True, "Error: Solver Failed!"      
                min_val = res.fun
                min_x = res.x  
        
        self.res = res
        self.U_next = min_x.reshape(-1, self.nU)
        self.X_next = np.concatenate((self.U_next.reshape(-1,self.nU),self.context),axis=1)
        self.min_val = min_val
        return self

    def Objective(self,X):
        # Unconstrained acqusition function
        if self.acquisition=='LCB' or self.acquisition=='UCB' :
            alpha = self.LCB(X)
        elif self.acquisition == 'greedy':
            alpha = self.greedy(X)
        
        if self.constraint is not None:
            assert type(self.constraint) is  list, "Constraint object must be of type list."

            if self.constraint_method is 'PF': # Compute probability of feasibility 
                PF = 1
                for i in range(len(self.constraint)):
                    PF = PF*self.PF(self.constraint[i],np.array(X).reshape(-1,self.nU))

                alpha = PF*alpha # unconstrained acq_fn scaled with the probability of feasibility  

            if self.constraint_method is 'Barrier': # Compute barrier terms
                Br = 0
                Br2 = 0  # For exploration: minimize -[Barrier(LCB) - Barrier(Mean)]
                for i in range(len(self.constraint)):
                    Br = Br + self.LogBarrier(self.constraint[i],np.array(X).reshape(-1,self.nU))
                    Br2 = Br2 + ((self.t*self.LogBarrier(self.constraint[i],np.array(X).reshape(-1,self.nU)) - self.t*self.LogBarrier(self.constraint[i],np.array(X).reshape(-1,self.nU),beta = 0))**2)
                p = np.random.uniform(0, 1, 1)
                if p < self.epsilon:
                    alpha = -Br2 # pure exploration
                else:
                    alpha = alpha + Br # - self.epsilon*Br2
                

            if self.constraint_method is 'Penalty':
                Pn = 0
                for i in range(len(self.constraint)):
                    Pn = Pn + self.Penalty(self.constraint[i],np.array(X).reshape(-1,self.nU))
                alpha = alpha + Pn # unconstrained acq_fn with the barrier terms
        return alpha

    def LCB(self,X,beta = 2):
        testX = np.concatenate((X.reshape(-1,self.nU),self.context),axis=1)
        mu, sigma = self.m.predict(testX.reshape(-1,self.nX))
        return mu - beta*sigma
    
    def greedy(self,X):
        testX = np.concatenate((X.reshape(-1,self.nU),self.context),axis=1)
        mu, sigma = self.m.predict(testX.reshape(-1,self.nX))
        return mu

    def PF(self,constraint,X):
        testX = np.concatenate((X.reshape(-1,self.nU),self.context),axis=1)
        mu, sigma = constraint.m.predict(testX.reshape(-1,self.nX))
        return norm.cdf(0,-mu,sigma) # Probability of feasibility

    def LogBarrier(self,constraint,X,beta = -2):
        testX = np.concatenate((X.reshape(-1,self.nU),np.ones(X.shape)*self.context),axis=1)
        mu, sigma = constraint.m.predict(testX.reshape(-1,self.nX))
        return -1/self.t*np.log(mu+beta*sigma)

    def Penalty(self,constraint,X,beta = -2):
        testX = np.concatenate((X.reshape(-1,self.nU),self.context),axis=1)
        mu, sigma = constraint.m.predict(testX.reshape(-1,self.nX))
        return -1*np.minimum(0,mu-2*sigma)

    def acq(self):
        if self.acquisition=='LCB' or self.acquisition=='UCB':
            self.acq_fn = self.LCB(self.X_grid)
        if self.acquisition=='EI':
            self.acq_fn = -self.EI(self.X_grid)
        if self.acquisition=='PI':
            self.acq_fn = -self.PI(self.X_grid)
        if self.acquisition == 'TS':
            self.acq_fn = self.posterior_sample
        if self.acquisition == 'greedy':
            self.acq_fn = self.greedy(self.X_grid)
        return self

    def plot_action_context_space(self,action_space=np.linspace(0,1,10),
                                    context_space=np.linspace(0,1,10),
                                    projection = '2d',
                                    plot_ideal=False,
                                    ideal_c = None,
                                    ideal_x = None,
                                    xlabel = 'Action $x$',
                                    ylabel = 'Context $d$',
                                    zlabel = 'Objective $f(x,d)$',
                                    fig_name=''):

        nX = action_space.shape[0]
        nC = context_space.shape[0]
        X,C = np.meshgrid(action_space,context_space)
        X_grid = np.vstack((X.flatten(), C.flatten())).T
        mu, sigma = self.m.predict(X_grid)
        Z = mu.reshape(nC,nX)

        C_now = self.context*np.ones([self.grid_size,1])
        testX = np.concatenate((action_space,C_now),axis=1)
        mu1, sigma1 = self.m.predict(testX)

        fig = plt.figure()
        if projection == '2d':
            alphas = np.linspace(0.1, 1, self.X_sample[:,0].size)
            colors = np.zeros((self.X_sample[:,0].size,4))
            colors[:, 3] = alphas
            #plt.scatter(self.X_sample[:,0], self.X_sample[:,1], self.Y_sample,label='Data')
            plt.scatter(self.X_sample[:,0], self.X_sample[:,1],label='Data',color = colors)
            plt.contour(X, C, Z,cmap=cm.coolwarm)
            plt.plot(action_space.reshape(self.grid_size,), C_now.reshape(self.grid_size,),'--',color=(0.4,0.4,0.4),label='Current context')
            plt.plot(self.X_next[:,0], self.context,'rv',label='$x_{next}$')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            if plot_ideal:
                assert ideal_c is not None, 'Missing argument: ideal_c'
                assert ideal_x is not None, 'Missing argument: ideal_x'
                plt.plot(ideal_x,ideal_c,'--',color=(1,0,0,0.5),label='True optimum')  
            plt.legend()
        elif projection == '3d':
            ax = fig.gca(projection='3d')
            ax.scatter(self.X_sample[:,0], self.X_sample[:,1], self.Y_sample,label='Data')
            
            ax.plot_surface(X,C,Z,rstride=8, cstride=8, alpha=0.3,label='Posterior mean')
            cset = ax.contour(X,C,Z, zdir='z', offset=np.min(Z)-1, cmap=cm.coolwarm)
            ax.plot(action_space.reshape(self.grid_size,), C_now.reshape(self.grid_size,), mu1.reshape(self.grid_size,),color=(0,0,0.7))
            ax.plot(action_space.reshape(self.grid_size,), C_now.reshape(self.grid_size,), 0*mu1.reshape(self.grid_size,),'--',color=(0.4,0.4,0.4))
            ax.scatter(self.X_next[:,0], self.context, 0,marker="v", color='r')
            if plot_ideal:
                assert ideal_c is not None, 'Missing argument: ideal_c'
                assert ideal_x is not None, 'Missing argument: ideal_x'
                plt.plot(ideal_x,ideal_c,'--',color=(1,0,0,0.5),zdir='z', zs=0,label='True optimum')  
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            #plt.legend()
        plt.show()
        if fig_name:
            fig.savefig(fig_name+'.pdf',bbox_inches='tight') 
  
    def plot_GP_context(self, context = None, X_sample=None, Y_sample = None,xlabel = '$x$', 
                        ylabel = '$f(x,d)$',fig_name = '', 
                        xlim = None, ylim =None):
        if context is None:
            self.C_grid = self.context*np.ones([self.grid_size,self.nC]).reshape(-1,self.nC)
            testX = np.concatenate((self.X_grid.reshape(-1,self.nU),self.C_grid),axis=1)
            mu, sigma = self.m.predict(testX)

            fig = plt.figure(constrained_layout=True,figsize=(4, 3))
            plt.fill_between(self.X_grid.ravel(), 
                            mu.ravel() + 2 * sigma.ravel(), 
                            mu.ravel() - 2 * sigma.ravel(), 
                            alpha=0.1,label='Confidence') 
            plt.plot(self.X_grid,mu+2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            plt.plot(self.X_grid,mu-2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
            plt.plot(self.X_grid,mu,color=(0,0.4,0.7),linewidth=2.4,label='Mean')
        else:
            fig = plt.figure(constrained_layout=True,figsize=(4, 3))
            for i in range(context.size):
                self.C_grid = context[i]*np.ones([self.grid_size,self.nC]).reshape(-1,self.nC)
                testX = np.concatenate((self.X_grid.reshape(-1,self.nU),self.C_grid),axis=1)
                mu, sigma = self.m.predict(testX)
                plt.fill_between(self.X_grid.ravel(), 
                                mu.ravel() + 2 * sigma.ravel(), 
                                mu.ravel() - 2 * sigma.ravel(),color= (0,0.4,0.7,0.1),
                                alpha=0.1,label='Confidence') 
                plt.plot(self.X_grid,mu+2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
                plt.plot(self.X_grid,mu-2*sigma,color=(0,0.4,0.7,0.1),linewidth=0.3)
                plt.plot(self.X_grid,mu,color=(0,0.4,0.7),linewidth=2.4,label='Mean')
        if X_sample is not None and Y_sample is not None:
            alphas = np.linspace(0.1, 1, X_sample.size)
            colors = np.zeros((X_sample.size,4))
            colors[:, 3] = alphas
            plt.scatter(X_sample,Y_sample,label='Data',color= colors)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if fig_name:
            fig.savefig(fig_name+'.pdf',bbox_inches='tight') 
        plt.show()
        return  