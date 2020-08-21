#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import os, copy
import gvar as gv
# import chipt lib for fit functions
import chipt

fig_width = 6.75 # in inches, 2x as wide as APS column
gr        = 1.618034333 # golden ratio
fig_size  = (fig_width, fig_width / gr)
fig_size2 = (fig_width, fig_width * 1.49)
plt_axes  = [0.145,0.145,0.85,0.85]
fs_text   = 20 # font size of text
fs_leg    = 16 # legend font size
mrk_size  = '5' # marker size
tick_size = 20 # tick size
lw        = 1 # line width

colors = {'a15':'#ec5d57', 'a12':'#70bf41', 'a09':'#51a7f9', 'a06':'#00FFFF'}
shapes = {'m400':'h', 'm350':'p', 'm310':'s', 'm220':'^', 'm180':'d', 'm130':'o', 'm135':'*'}
labels = {
    'a15m400' :'', 'a15m350':'', 'a15m310':'a15', 'a15m220':'','a15m135XL':'',
    'a12m400' :'', 'a12m350':'', 'a12m310':'a12', 'a12m220':'', 'a12m130':'',
    'a12m220L':'', 'a12m220S':'','a12m220ms':'',  'a15m310L':'', 'a12m180L':'',
    'a09m400' :'', 'a09m350':'', 'a09m310':'a09', 'a09m220':'', 'a09m135':'',
    'a06m310L':'a06',
    }
dx_cont = {
    'a15m400'  :0.0050, 'a12m400' :0.0050, 'a09m400':0.0050,
    'a15m350'  :0.0025, 'a12m350' :0.0025, 'a09m350':0.0025,
    'a15m310'  :0.,     'a12m310' :0.,     'a09m310':0.,     'a06m310L':0.,
    'a15m220'  :-0.0025,'a12m220' :-0.0025,'a09m220':-0.0025,
    'a15m135XL':-0.0050,'a12m130' :-0.0050,'a09m135':-0.0050,
    'a12m220L' :-0.0037,'a12m220S':-0.0012,'a12m220ms':+0.0012, 'a15m310L'  :0., 'a12m180L':0.,
    }

def plot_l_s(data,switches,phys_point):
    plt.ion()
    if not os.path.exists('figures'):
        os.makedirs('figures')
    figO = plt.figure('ls_O', figsize=fig_size)
    axO  = plt.axes(plt_axes)
    figF = plt.figure('ls_F', figsize=fig_size)
    axF  = plt.axes(plt_axes)
    for ens in switches['ensembles']:
        a = ens.split('m')[0]
        m = ens[3:7]
        lO_sq = data['p'][(ens,'mpi')]**2 / data['p'][(ens,'m_omega')]**2
        sO_sq = (2*data['p'][(ens,'mk')]**2 - data['p'][(ens,'mpi')]**2) / data['p'][(ens,'m_omega')]**2
        axO.errorbar(lO_sq.mean,sO_sq.mean, xerr=lO_sq.sdev,yerr=sO_sq.sdev,
            color=colors[a],marker=shapes[m],label=labels[ens],linestyle='None')
        lF_sq = data['p'][(ens,'mpi')]**2 / data['p'][(ens,'Lam_F')]**2
        sF_sq = (2*data['p'][(ens,'mk')]**2 -data['p'][(ens,'mpi')]**2) / data['p'][(ens,'Lam_F')]**2
        axF.errorbar(lF_sq.mean,sF_sq.mean, xerr=lF_sq.sdev,yerr=sF_sq.sdev,
            color=colors[a],marker=shapes[m],label=labels[ens],linestyle='None')

    axO.legend(loc=1,fontsize=fs_leg)
    axO.set_xlabel(r'$l_\Omega^2 = m_\pi^2 / m_\Omega^2$', fontsize=fs_text)
    axO.set_ylabel(r'$s_\Omega^2 = (2m_K^2 - m_\pi^2) / m_\Omega^2$', fontsize=fs_text)
    axO.axis([0,.063, 0.161,0.201])
    lO_phys = phys_point['p']['mpi']**2 / phys_point['p']['m_omega']**2
    sO_phys = (2*phys_point['p']['mk']**2 - phys_point['p']['mpi']**2) / phys_point['p']['m_omega']**2
    axO.axvspan(lO_phys.mean-lO_phys.sdev, lO_phys.mean+lO_phys.sdev,color='k',alpha=0.3)
    axO.axhspan(sO_phys.mean-sO_phys.sdev, sO_phys.mean+sO_phys.sdev,color='k',alpha=0.3)
    axO.tick_params(direction='in',labelsize=tick_size)
    if switches['save_figs']:
        plt.figure('ls_O')
        plt.savefig('figures/ls_O.pdf',transparent=True)

    axF.legend(loc=1,fontsize=fs_leg)
    axF.set_xlabel(r'$l_F^2 = m_\pi^2 / (4\pi F_\pi)^2$', fontsize=fs_text)
    axF.set_ylabel(r'$s_F^2 = (2m_K^2 - m_\pi^2) / (4\pi F_\pi)^2$', fontsize=fs_text)
    axF.axis([0,.099, 0.261,0.366])
    lF_phys = phys_point['p']['mpi']**2 / phys_point['p']['Lam_F']**2
    sF_phys = (2*phys_point['p']['mk']**2 - phys_point['p']['mpi']**2) / phys_point['p']['Lam_F']**2
    axF.axvspan(lF_phys.mean-lF_phys.sdev, lF_phys.mean+lF_phys.sdev,color='k',alpha=0.3)
    axF.axhspan(sF_phys.mean-sF_phys.sdev, sF_phys.mean+sF_phys.sdev,color='k',alpha=0.3)
    axF.tick_params(direction='in',labelsize=tick_size)
    if switches['save_figs']:
        plt.figure('ls_F')
        plt.savefig('figures/ls_F.pdf',transparent=True)


    plt.ioff()
    plt.show()

class ExtrapolationPlots:

    def __init__(self, model, model_list, fitEnv, fit_result, switches):
        if not os.path.exists('figures'):
            os.makedirs('figures')
        self.model      = model
        self.FF         = model.split('_')[-1]
        self.fv         = 'FV' in model
        self.model_list = model_list
        self.fitEnv     = fitEnv
        self.fit_result = fit_result
        self.switches   = switches

        # create fit functions for original and shifted points
        self.og_fit     = chipt.FitModel(self.model_list, _fv=self.fv, _FF=self.FF)
        self.shift_list = list(model_list)
        self.shift_fit  = chipt.FitModel(self.shift_list, _fv=False, _FF=self.FF)
        # which ensemble to get aw0 from
        self.aw0_keys = {'a15':'a15m135XL','a12':'a12m130','a09':'a09m135','a06':'a06m310L'}

    def plot_vs_eps_asq(self,shift_points):
        self.shift_xp = copy.deepcopy(shift_points)
        for k in self.fit_result.p:
            if isinstance(k,str):
                self.shift_xp['p'][k] = self.fit_result.p[k]
        y_plot = []
        x_plot = []
        a_range = np.sqrt(np.arange(0, .16**2, .16**2 / 50))
        for a_fm in a_range:
            self.shift_xp['p']['aw0'] = a_fm / self.shift_xp['p']['w0']
            x_plot.append((self.shift_xp['p']['aw0'] / 2)**2)
            y_a = self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p'])
            y_plot.append(y_a)
        x  = np.array([k.mean for k in x_plot])
        y  = np.array([k.mean for k in y_plot])
        dy = np.array([k.sdev for k in y_plot])

        figsize = fig_size
        self.fig_cont = plt.figure('w0_mO_vs_ea_'+self.model,figsize=figsize)
        self.ax_cont  = plt.axes(plt_axes)
        self.ax_cont.fill_between(x, y-dy, y+dy, color='#b36ae2', alpha=0.4)

        self.plot_data(p_type='ea', ax=self.ax_cont)
        handles, labels = self.ax_cont.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        self.ax_cont.legend(handles, labels, ncol=4, fontsize=fs_leg)

        self.ax_cont.set_xlabel(r'$\epsilon_a^2 = a^2 / (2 w_0)^2$',fontsize=fs_text)
        self.ax_cont.set_ylabel(r'$w_0 m_\Omega$',fontsize=fs_text)
        self.ax_cont.set_ylim(1.321, 1.471)
        self.ax_cont.text(0.0175, 1.34, r'%s' %(self.model.replace('_','\_')),\
            horizontalalignment='left', verticalalignment='center', \
            fontsize=fs_text, bbox={'facecolor':'None','boxstyle':'round'})
        self.ax_cont.set_xlim(0,.21)


        if self.switches['save_figs']:
            plt.savefig('figures/'+'w0_mO_vs_ea_'+self.model+'.pdf',transparent=True)

    def plot_vs_eps_pi(self,shift_points,eps='l'):
        self.shift_xp = copy.deepcopy(shift_points)
        eps_pisq_phys = (gv.gvar(self.shift_xp['p']['mpi'] / self.shift_xp['p']['Lam_'+self.FF]))**2
        eps_ksq_phys =  (gv.gvar(self.shift_xp['p']['mk'] / self.shift_xp['p']['Lam_'+self.FF]))**2
        eps_ssq_phys = 2 * eps_ksq_phys - eps_pisq_phys
        for k in self.fit_result.p:
            if isinstance(k,str):
                self.shift_xp['p'][k] = self.fit_result.p[k]
        y_plot = dict()
        # fit for each a and continuum
        y_plot['a15'] = []
        y_plot['a12'] = []
        y_plot['a09'] = []
        y_plot['a06'] = []
        y_plot['a00'] = []
        # xpt convergence - these are summed to a given order
        y_conv = dict()
        y_conv['LO']    = []
        y_conv['NLO']   = []
        y_conv['NNLO']  = []
        lo_lst    = [t for t in self.model_list if 'lo' in t and not any(n in t for n in ['nlo', 'nnlo'])]
        nlo_lst   = [t for t in self.model_list if 'nlo' in t and not any(n in t for n in ['nnlo'])]
        nnlo_lst  = [t for t in self.model_list if 'nnlo' in t]
        lo_fit    = chipt.FitModel(lo_lst, _fv=False, _FF=self.FF)
        nlo_fit   = chipt.FitModel(lo_lst+nlo_lst, _fv=False, _FF=self.FF)
        nnlo_fit  = chipt.FitModel(lo_lst+nlo_lst+nnlo_lst, _fv=False, _FF=self.FF)
        x_plot = []
        mpi_range = np.sqrt(np.arange(100, 411**2, 411**2/200))
        ms_range  = np.sqrt(np.arange(500**2, 900**2, (900**2 - 500**2)/200))
        m_range = {'l':mpi_range, 's':ms_range}

        for a_m in m_range[eps]:
            x_plot.append(a_m**2 / self.shift_xp['p']['Lam_'+self.FF]**2)
            if eps == 'l':
                self.shift_xp['p']['mpi'] = a_m
            elif eps == 's':
                self.shift_xp['p']['mk']  = np.sqrt(0.5* (a_m**2 + (self.shift_xp['p']['mpi']/self.shift_xp['p']['Lam_'+self.FF])**2))
            self.shift_xp['p']['aw0'] = 0
            y_plot['a00'].append(self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p']))
            y_conv['LO'].append(self.fitEnv._fit_function(lo_fit, self.shift_xp['x'], self.shift_xp['p']))
            y_conv['NLO'].append(self.fitEnv._fit_function(nlo_fit, self.shift_xp['x'], self.shift_xp['p']))
            y_conv['NNLO'].append(self.fitEnv._fit_function(nnlo_fit, self.shift_xp['x'], self.shift_xp['p']))
            for aa in ['a15','a12','a09','a06']:
                self.shift_xp['p']['aw0'] = self.fitEnv.p[(self.aw0_keys[aa],'aw0')]
                y_plot[aa].append(self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p']))
        x = np.array([k.mean for k in x_plot])
        y  = dict()
        dy = dict()
        y['a00']  = np.array([k.mean for k in y_plot['a00']])
        dy['a00'] = np.array([k.sdev for k in y_plot['a00']])
        for aa in ['a15','a12','a09','a06']:
            y[aa]  = np.array([k.mean for k in y_plot[aa]])
            dy[aa] = np.array([k.sdev for k in y_plot[aa]])

        fig_name = 'w0_mO_vs_'+eps+'_'+self.model
        fig_x      = plt.figure(fig_name, figsize=fig_size)
        ax_x  = plt.axes(plt_axes)
        ax_x.fill_between(x, y['a00']-dy['a00'], y['a00']+dy['a00'], color='#b36ae2',alpha=0.4)
        for aa in ['a15','a12','a09','a06']:
            ax_x.plot(x, y[aa], color=colors[aa])

        # plot physical eps_pi**2
        if eps == 'l':
            ax_x.axvline(eps_pisq_phys.mean,linestyle='--',color='#a6aaa9')
            ax_x.axvspan(eps_pisq_phys.mean -eps_pisq_phys.sdev, eps_pisq_phys.mean +eps_pisq_phys.sdev,
                alpha=0.4, color='#a6aaa9')
        elif eps == 's':
            ax_x.axvline(eps_ssq_phys.mean,linestyle='--',color='#a6aaa9')
            ax_x.axvspan(eps_ssq_phys.mean -eps_ssq_phys.sdev, eps_ssq_phys.mean +eps_ssq_phys.sdev,
                alpha=0.4, color='#a6aaa9')
        # plot data
        self.plot_data(p_type=eps, ax=ax_x)
        handles, labels = ax_x.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        if self.FF == 'F':
            if eps == 'l':
                loc=4
            elif eps == 's':
                loc=2
        else:
            loc=1
        ax_x.legend(handles, labels, loc=loc, ncol=2, fontsize=fs_leg)

        # labels
        if eps == 'l':
            eps_FF = {'F':r'$l_F^2 = (m_\pi / 4\pi F_\pi)^2$', 'O':r'$l_\Omega^2 = (m_\pi / m_\Omega)^2$'}
            xlim_FF = {'F':(0,0.094), 'O':(0,.061)}
        elif eps == 's':
            eps_FF = {'F':r'$s_F^2 = (2m_K^2 - m_\pi^2) / (4\pi F_\pi)^2$', 'O':r'$s_\Omega^2 = (2m_K^2 - m_\pi^2) / m_\Omega^2$'}
            xlim_FF = {'F':(0.33,0.369), 'O':(0.16,0.234)}
        ax_x.set_xlabel(eps_FF[self.FF],fontsize=fs_text)
        ax_x.set_xlim(xlim_FF[self.FF])
        ax_x.set_ylabel(r'$w_0 m_\Omega$',fontsize=fs_text)
        ax_x.set_ylim(1.281, 1.499)
        ax_x.text(0.0175, 1.3, r'%s' %(self.model.replace('_','\_')),\
            horizontalalignment='left', verticalalignment='center', \
            fontsize=fs_text, bbox={'facecolor':'None','boxstyle':'round'})

        if self.switches['save_figs']:
            plt.savefig('figures/'+fig_name+'.pdf',transparent=True)
        if eps == 'l':
            self.ax_l = ax_x
        elif eps == 's':
            self.ax_s = ax_x

        # Convergence plot
        order_list = ['LO']
        if 'nnlo' in self.model:
            order_list = order_list + ['NLO','NNLO']
        elif 'nlo' in self.model:
            order_list = order_list + ['NLO']
        self.fig_conv = plt.figure(fig_name+'_convergence_'+self.model, figsize=fig_size)
        self.ax_conv  = plt.axes(plt_axes)
        labels = {'LO':'LO', 'NLO':r'NLO','NNLO':r'N$^2$LO'}
        for order in order_list:
            mean = np.array([k.mean for k in y_conv[order]])
            sdev = np.array([k.sdev for k in y_conv[order]])
            self.ax_conv.fill_between(x, mean-sdev, mean+sdev, alpha=.4, label=labels[order])
        self.ax_conv.set_xlabel(eps_FF[self.FF],fontsize=fs_text)
        self.ax_conv.set_xlim(xlim_FF[self.FF])
        self.ax_conv.set_ylabel(r'$w_0 m_\Omega$',fontsize=fs_text)
        self.ax_conv.set_ylim(1.281, 1.499)
        self.ax_conv.text(0.0175, 1.3, r'%s' %(self.model.replace('_','\_')),\
            horizontalalignment='left', verticalalignment='center', \
            fontsize=fs_text, bbox={'facecolor':'None','boxstyle':'round'})
        self.ax_conv.axvline(eps_pisq_phys.mean,linestyle='--',color='#a6aaa9')
        self.ax_conv.axvspan(eps_pisq_phys.mean -eps_pisq_phys.sdev, eps_pisq_phys.mean +eps_pisq_phys.sdev,
            alpha=0.4, color='#a6aaa9')
        self.ax_conv.legend(ncol=3, fontsize=fs_leg)
        if self.switches['save_figs']:
            plt.savefig('figures/'+fig_name+'_convergence_'+self.model+'.pdf',transparent=True)


    def plot_data(self, p_type, ax, offset=False, raw=False):
        y_shift = self.shift_data(p_type=p_type)
        for a_ens in self.switches['ensembles']:
            if a_ens in self.switches['ensembles_fit']:
                c = colors[a_ens.split('m')[0]]
                alpha = 1
            else:
                c = 'k'
                alpha = 0.4
            s = shapes['m'+a_ens.split('m')[1][0:3]]
            if p_type == 'ea':
                x  = (self.fitEnv.p[(a_ens, 'aw0')] / 2)**2
                dx = dx_cont[a_ens]
            elif p_type == 'l':
                x  = (self.fitEnv.p[(a_ens, 'mpi')] / self.fitEnv.p[(a_ens, 'Lam_'+self.FF)])**2
                dx = 0
            elif p_type == 's':
                x  = 2*(self.fitEnv.p[(a_ens, 'mk')]/self.fitEnv.p[(a_ens, 'Lam_'+self.FF)])**2
                x += -(self.shift_xp['p']['mpi'] / self.shift_xp['p']['Lam_'+self.FF])**2
                dx = 0
            label = labels[a_ens]
            if p_type == 'ea':
                if a_ens in self.switches['ensembles_fit']:
                    mfc = c
                else:
                    mfc = 'None'
            elif p_type in ['l','s']:
                mfc = 'None'
            y = self.fitEnv.y[a_ens] + y_shift[a_ens]
            if p_type == 'ea':
                self.ax_cont.errorbar(x=x.mean+dx, y=y.mean,xerr=x.sdev, yerr=y.sdev,
                    marker=s, color=c, mfc=mfc, alpha=alpha, linestyle='None', label=label)
            elif p_type == 'l':
                ax.errorbar(x=x.mean+dx, y=y.mean,xerr=x.sdev, yerr=y.sdev,
                    marker=s, color=c, mfc=mfc, alpha=alpha, linestyle='None', label=label)
            elif p_type == 's':
                ax.errorbar(x=x.mean+dx, y=y.mean,xerr=x.sdev, yerr=y.sdev,
                    marker=s, color=c, mfc=mfc, alpha=alpha, linestyle='None', label=label)

    def shift_data(self, p_type):
        y_shift = dict()
        if self.switches['debug_shift']:
            print('%9s   %11s   y_shift[ens]' %('ensemble', 'y[ens]'))
            print('---------------------------------------------------------------')
        for a_ens in self.switches['ensembles']:
            og_priors = dict()
            for k, v in self.fitEnv.p.items():
                if type(k) == tuple and k[0] == a_ens:
                    #print(a_ens,k,v)
                    if k in self.fit_result.p:
                    #if a_ens in self.switches['ensembles_fit']:
                        og_priors[k[1]] = self.fit_result.p[k] # the x-params which are priors
                    else:
                        # if it is from an excluded ensemble - get from original data
                        og_priors[k[1]] = v
            for k in self.fit_result.p:
                if isinstance(k,str):# grab the LECs from the fit results
                    og_priors[k] = self.fit_result.p[k]    # the LECs of the fit
            aa = a_ens[0:3]
            self.shift_xp['p']['aw0'] = self.fitEnv.p[(self.aw0_keys[aa],'aw0')]
            self.shift_xp['x']['alphaS'] = self.fitEnv.x[a_ens]['alphaS']
            if p_type in ['l','s']:
                if p_type == 'l':
                    self.shift_xp['p']['mpi'] = self.fitEnv.p[(a_ens,'mpi')] / self.fitEnv.p[(a_ens, 'Lam_'+self.FF)]
                    self.shift_xp['p']['mk']  = self.shift_xp['p']['mk'] / self.shift_xp['p']['Lam_'+self.FF]
                elif p_type == 's':
                    self.shift_xp['p']['mpi'] = self.shift_xp['p']['mpi'] / self.shift_xp['p']['Lam_'+self.FF]
                    s_sq = (2*self.fitEnv.p[(a_ens,'mk')]**2 - self.fitEnv.p[(a_ens,'mpi')]**2) / self.fitEnv.p[(a_ens, 'Lam_'+self.FF)]**2
                    self.shift_xp['p']['mk']  = np.sqrt(0.5*(s_sq + self.shift_xp['p']['mpi']**2))
                self.shift_xp['p']['Lam_'+self.FF] = 1
            og_y    = self.fitEnv._fit_function(self.og_fit,    self.fitEnv.x[a_ens], og_priors)
            shift_y = self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'],   self.shift_xp['p'])
            y_shift[a_ens] = shift_y - og_y
            if self.switches['debug_shift']:
                print('%9s   %11s   %s' %(a_ens, og_y, shift_y))
        return y_shift

    def plot_vs_ml(self):
        fv_dict = dict()
        fv_dict['p'] = dict()
        fv_dict['p']['mpi']          = self.fit_result.p['a12m220L', 'mpi']
        fv_dict['p']['mk']           = self.fit_result.p['a12m220L', 'mk']
        fv_dict['p']['Lam_'+self.FF] = self.fit_result.p['a12m220L', 'Lam_'+self.FF]
        fv_dict['p']['aw0']          = self.fit_result.p['a12m220L', 'aw0']
        fv_dict['x'] = dict()
        fv_dict['x'] = dict(self.fit_result.x['a12m220L'])
        for k in self.fit_result.p:
            if isinstance(k, str):
                fv_dict['p'][k] = self.fit_result.p[k]
        mpi = fv_dict['p']['mpi'].mean
        mk  = fv_dict['p']['mk'].mean
        me  = np.sqrt(4./3 * mk**2 - 1./3*mpi**2)
        fv_pred = []
        x   = []
        fv_fit_func = chipt.FitModel(self.model_list, _fv=self.fv, _FF=self.FF)
        #print(fv_dict['p'])
        for mL in np.arange(3.,10.1,.1):
            x.append(np.exp(-mL) / (mL)**1.5)
            fv_dict['x']['mpiL'] = mL
            fv_pred.append(self.fitEnv._fit_function(fv_fit_func, fv_dict['x'], fv_dict['p']))
        x = np.array(x)
        y  = np.array([k.mean for k in fv_pred])
        dy = np.array([k.sdev for k in fv_pred])

        self.fig_Fv = plt.figure('w0_mO_vs_mL_'+self.model, figsize=fig_size)
        self.ax_fv  = plt.axes(plt_axes)
        self.ax_fv.fill_between(x, y-dy, y+dy, color=colors['a12'], alpha=0.4)
        markers = ['s','o','*']
        mL_ens = dict()
        xL_ens = dict()
        fL_ens = dict()
        for i_e,ens in enumerate(['a12m220L', 'a12m220', 'a12m220S']):
            if ens in self.switches['ensembles_fit']:
                c = color=colors['a12']
            else:
                c = 'k'
            mL_ens[ens] = self.fit_result.x[ens]['mpiL']
            y_data = self.fit_result.y[ens]
            self.ax_fv.errorbar(np.exp(-mL_ens[ens])/mL_ens[ens]**1.5, y_data.mean, yerr=y_data.sdev, \
                marker=markers[i_e], color=c, linestyle='None',label=r'$m_\pi L=%.2f$' %(mL_ens[ens]))
            # collect info for making text in band
            fv_dict['x']['mpiL'] = mL_ens[ens]
            fv_dict['x'][k] = mL_ens[ens] * mk/mpi
            fv_dict['x'][k] = mL_ens[ens] * me/mpi
            xL_ens[ens] = np.exp(-mL_ens[ens])/mL_ens[ens]**1.5
            fL_ens[ens] = self.fitEnv._fit_function(fv_fit_func, fv_dict['x'], fv_dict['p']).mean

        self.ax_fv.set_xlabel(r'$e^{-m_\pi L} / (m_\pi L)^{3/2}$',fontsize=fs_text)
        self.ax_fv.set_ylabel(r'$w_0 m_\Omega$',fontsize=fs_text)
        self.ax_fv.legend(ncol=3, fontsize=fs_leg, columnspacing=0.5)
        self.ax_fv.vlines(0,1.12,1.15, color='k', lw=0.4)
        self.ax_fv.set_ylim(1.376, 1.424)
        self.ax_fv.set_xlim(-0.000,.0075)

        # Do a little of trig to get text to line up in band
        x_text  = (xL_ens['a12m220S'] + xL_ens['a12m220']) / 2
        def ml_x(mL):
            return np.exp(-mL)/mL**1.5
        mL_text = minimize_scalar(lambda x: (ml_x(x) -x_text)**2 ,bounds=(3,7), method='bounded').x
        fv_dict['x']['mpiL'] = mL_text
        fv_dict['x'][k] = mL_text * mk/mpi
        fv_dict['x'][k] = mL_text * me/mpi
        y_text = self.fitEnv._fit_function(fv_fit_func, fv_dict['x'], fv_dict['p']).mean
        # scale dy and dx by the limits of the plot to get angle right
        dx = (xL_ens['a12m220S'] - xL_ens['a12m220']) / (self.ax_fv.get_xlim()[1]-self.ax_fv.get_xlim()[0])
        dy = (fL_ens['a12m220S'] - fL_ens['a12m220']) / (self.ax_fv.get_ylim()[1]-self.ax_fv.get_ylim()[0])
        angle = 180/np.pi * np.arctan(dy / dx / gr) # remember the golden ratio scaling
        self.ax_fv.text(x_text, y_text - 0.0003, \
            r'a12m220: $\delta_{\rm FV}^{{\rm NLO}\ \chi{\rm PT}}(\epsilon_\pi^2, m_\pi L)$', \
            horizontalalignment='center', verticalalignment='center', \
            rotation=angle, fontsize=fs_text-1)

        if self.switches['save_figs']:
            plt.savefig('figures/'+'w0_mO_vs_mL_'+self.model+'.pdf',transparent=True)
