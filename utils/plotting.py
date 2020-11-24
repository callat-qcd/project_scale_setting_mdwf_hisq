#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import os, copy
import gvar as gv
# import chipt lib for fit functions
import chipt

if not os.path.exists('figures'):
    os.makedirs('figures')

fig_width = 6.75 # in inches, 2x as wide as APS column
gr        = 1.618034333 # golden ratio
fig_size  = (fig_width, fig_width / gr)
fig_size2 = (fig_width, 2 * fig_width / gr)
plt_axes  = [0.145,0.145,0.85,0.85]
plt_axes2 = [0.145,0.07,0.85,0.92]
fs_text   = 20 # font size of text
fs_leg    = 16 # legend font size
mrk_size  = '5' # marker size
tick_size = 20 # tick size
lw        = 1 # line width

colors = {'a15':'#ec5d57', 'a12':'#70bf41', 'a09':'#51a7f9', 'a06':'#6a5acd'}

def colorFader(c1,c2,mix=0):
    ''' fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        taken from Markus Dutschke answer at stack overflow
        https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    '''
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

shapes = {'m400':'h', 'm350':'p', 'm310':'s', 'm220':'^', 'm180':'d', 'm130':'o', 'm135':'*'}
ls_labels = {
    'a15m310' :r'$a_{15}$',
    'a12m310' :r'$a_{12}$',
    'a09m310' :r'$a_{09}$',
    'a06m310L':r'$a_{06}$',
    }
l_labels = {
    'a15m310' :r'$a_{15}(l_F,s_F^{\rm phys})$',
    'a12m310' :r'$a_{12}(l_F,s_F^{\rm phys})$',
    'a09m310' :r'$a_{09}(l_F,s_F^{\rm phys})$',
    'a06m310L':r'$a_{06}(l_F,s_F^{\rm phys})$',
    }
eps_a_labels = {
    'a15m310' :r'$a_{15}(l_F^{\rm phys},s_F^{\rm phys})$',
    'a12m310' :r'$a_{12}(l_F^{\rm phys},s_F^{\rm phys})$',
    'a09m310' :r'$a_{09}(l_F^{\rm phys},s_F^{\rm phys})$',
    'a06m310L':r'$a_{06}(l_F^{\rm phys},s_F^{\rm phys})$',
    }
dx_cont = {
    'a15m400'  :0.0050, 'a12m400' :0.0050, 'a09m400':0.0050,
    'a15m350'  :0.0025, 'a12m350' :0.0025, 'a09m350':0.0025,
    'a15m310'  :0.,     'a12m310' :0.,     'a09m310':0.,     'a06m310L':0.,
    'a15m220'  :-0.0025,'a12m220' :-0.0025,'a09m220':-0.0025,
    'a15m135XL':-0.0050,'a12m130' :-0.0050,'a09m135':-0.0050,
    'a12m220L' :-0.0037,'a12m220S':-0.0012,'a12m220ms':+0.0012, 'a15m310L'  :0., 'a12m180L':0.,
    }
# for a06, use MILC physical pion mass value
aw0_a06 = 1 / gv.gvar('3.0119(19)')

# LANDSCAPE plots
def plot_l_s(data,switches,phys_point):
    if not os.path.exists('figures'):
        os.makedirs('figures')
    figO = plt.figure('ls_O', figsize=fig_size)
    axO  = plt.axes(plt_axes)
    figF = plt.figure('ls_F', figsize=fig_size)
    axF  = plt.axes(plt_axes)
    for ens in switches['ensembles']:
        a = ens.split('m')[0]
        m = ens[3:7]
        lbl = ''
        if ens in ls_labels:
            lbl = ls_labels[ens]
        lO_sq = data['p'][(ens,'mpi')]**2 / data['p'][(ens,'Lam_O')]**2
        sO_sq = (2*data['p'][(ens,'mk')]**2 - data['p'][(ens,'mpi')]**2) / data['p'][(ens,'Lam_O')]**2
        axO.errorbar(lO_sq.mean,sO_sq.mean, xerr=lO_sq.sdev,yerr=sO_sq.sdev,
            color=colors[a],marker=shapes[m],label=lbl,linestyle='None')
        lF_sq = data['p'][(ens,'mpi')]**2 / data['p'][(ens,'Lam_F')]**2
        sF_sq = (2*data['p'][(ens,'mk')]**2 -data['p'][(ens,'mpi')]**2) / data['p'][(ens,'Lam_F')]**2
        axF.errorbar(lF_sq.mean,sF_sq.mean, xerr=lF_sq.sdev,yerr=sF_sq.sdev,
            color=colors[a],marker=shapes[m],label=lbl,linestyle='None')

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


def plot_lF_a(data,switches,phys_point):
    fig = plt.figure('lF_a', figsize=fig_size)
    ax  = plt.axes(plt_axes)
    for ens in switches['ensembles']:
        a = ens.split('m')[0]
        m = ens[3:7]
        lbl = ''
        if ens in ls_labels:
            lbl = ls_labels[ens]
        lF_sq = data['p'][(ens,'mpi')]**2 / data['p'][(ens,'Lam_F')]**2
        ea_sq = data['p'][(ens,'aw0')]**2 / 4
        ax.errorbar(ea_sq.mean,lF_sq.mean, xerr=ea_sq.sdev,yerr=lF_sq.sdev,
            color=colors[a],marker='o',markersize=8,label=lbl,linestyle='None')

    mt = {'m400':0.0888, 'm350': 0.0726, 'm310':0.0605, 'm220':0.0331, 'm180':0.0228, 'm135':0.0135}
    for m in mt:
        if m != 'm135':
            ax.axhline(mt[m], linestyle='--', color='k', alpha=.2)
        ax.text(0.09, mt[m], m, fontsize=fs_leg, verticalalignment='bottom')
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lbls[::-1], loc=1, fontsize=fs_leg, ncol=4)
    ax.set_ylabel(r'$l_F^2 = m_\pi^2 / (4\pi F_\pi)^2$', fontsize=fs_text)
    ax.set_xlabel(r'$\epsilon_a^2 = (a / 2 w_0)^2$', fontsize=fs_text)
    ax.axis([0,.21, 0.001,0.114])
    lF_phys = phys_point['p']['mpi']**2 / phys_point['p']['Lam_F']**2
    ax.axhspan(lF_phys.mean-lF_phys.sdev, lF_phys.mean+lF_phys.sdev,color='k',alpha=0.3)
    ax.tick_params(direction='in',labelsize=tick_size)
    if switches['save_figs']:
        plt.savefig('figures/lF_a.pdf',transparent=True)

def plot_raw_data(data,switches,phys_point):
    if not os.path.exists('figures'):
        os.makedirs('figures')
    fig = plt.figure('wm_lF', figsize=fig_size)
    ax  = plt.axes(plt_axes)
    for ens in switches['ensembles']:
        a = ens.split('m')[0]
        m = ens[3:7]
        lbl = ''
        if ens in ls_labels:
            lbl = ls_labels[ens]
        lF_sq = data['p'][(ens,'mpi')]**2 / data['p'][(ens,'Lam_F')]**2
        wm    = data['y'][ens]
        ax.errorbar(lF_sq.mean,wm.mean, xerr=lF_sq.sdev,yerr=wm.sdev,
            color=colors[a],marker='o',markersize=8,label=lbl,linestyle='None')

    ax.legend(loc=1, fontsize=fs_leg, ncol=4)
    ax.set_ylabel(r'$w_0 m_\Omega$', fontsize=fs_text)
    ax.set_xlabel(r'$l_F^2 = m_\pi^2 / (4\pi F_\pi)^2$', fontsize=fs_text)
    ax.axis([0,0.094, 1.351,1.499])
    lF_phys = phys_point['p']['mpi']**2 / phys_point['p']['Lam_F']**2
    ax.axvspan(lF_phys.mean-lF_phys.sdev, lF_phys.mean+lF_phys.sdev,color='k',alpha=0.3)
    ax.tick_params(direction='in',labelsize=tick_size)
    if switches['save_figs']:
        plt.savefig('figures/wmO_lFsq.pdf',transparent=True)

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
        self.aw0_keys = {'a15':'a15m135XL','a12':'a12m130','a09':'a09m135'}

    def plot_vs_eps_asq(self,shift_points):
        self.shift_xp = copy.deepcopy(shift_points)
        for k in self.fit_result.p:
            if isinstance(k,str):
                self.shift_xp['p'][k] = self.fit_result.p[k]
        y_plot = []
        x_plot = []
        a_range = np.sqrt(np.arange(0, .16**2, .16**2 / 50))
        eps_aSq_range = np.arange(0,.21,.21/500)
        for aa in eps_aSq_range:
            #self.shift_xp['p']['aw0'] = a_fm / self.shift_xp['p']['w0']
            #x_plot.append((self.shift_xp['p']['aw0'] / 2)**2)
            self.shift_xp['p']['aw0'] = gv.gvar(2,0)*np.sqrt(aa)
            x_plot.append(gv.gvar(aa,0))
            y_a = self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p'])
            y_plot.append(y_a)
        x  = np.array([k.mean for k in x_plot])
        i06 = np.where(x > ((self.fitEnv.p[('a06m310L', 'aw0')] / 2)**2).mean)[0][0]
        i09 = np.where(x > ((self.fitEnv.p[('a09m310',  'aw0')] / 2)**2).mean)[0][0]
        i12 = np.where(x > ((self.fitEnv.p[('a12m310',  'aw0')] / 2)**2).mean)[0][0]
        i15 = np.where(x > ((self.fitEnv.p[('a15m400',  'aw0')] / 2)**2).mean)[0][0]
        y  = np.array([k.mean for k in y_plot])
        dy = np.array([k.sdev for k in y_plot])

        figsize = fig_size
        self.fig_cont = plt.figure('w0_mO_vs_ea_'+self.model,figsize=figsize)
        self.ax_cont  = plt.axes(plt_axes)
        #self.ax_cont.fill_between(x, y-dy, y+dy, facecolor='None',edgecolor='k', hatch='\\')

        for i in range(i06):
            self.ax_cont.fill_between(x[i:i+2], (y-dy)[i:i+2], (y+dy)[i:i+2],
                color=colorFader('k','#6a5acd',i/i06),alpha=.3)
        for ii,i in enumerate(range(i06,i09)):
            self.ax_cont.fill_between(x[i:i+2], (y-dy)[i:i+2], (y+dy)[i:i+2],
                color=colorFader('#6a5acd','#51a7f9',ii/(i09-i06)),alpha=.3)
        for ii,i in enumerate(range(i09,i12)):
            self.ax_cont.fill_between(x[i:i+2], (y-dy)[i:i+2], (y+dy)[i:i+2],
                color=colorFader('#51a7f9','#70bf41',ii/(i12-i09)),alpha=.3)
        for ii,i in enumerate(range(i12,i15)):
            self.ax_cont.fill_between(x[i:i+2], (y-dy)[i:i+2], (y+dy)[i:i+2],
                color=colorFader('#70bf41','#ec5d57',ii/(i15-i12)),alpha=.3)

        self.plot_data(p_type='ea', ax=self.ax_cont)
        handles, labels = self.ax_cont.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        self.ax_cont.legend(handles, labels, ncol=2, columnspacing=0, handletextpad=0.1, fontsize=fs_leg)

        self.ax_cont.set_xlabel(r'$\epsilon_a^2 = a^2 / (2 w_0)^2$',fontsize=fs_text)
        self.ax_cont.set_ylabel(r'$w_0 m_\Omega$',fontsize=fs_text)
        self.ax_cont.text(0.0175, 1.375, r'%s' %(self.model.replace('_','\_')),\
            horizontalalignment='left', verticalalignment='center', \
            fontsize=fs_text, bbox={'facecolor':'None','boxstyle':'round'})
        self.ax_cont.set_xlim(0,.21)
        self.ax_cont.set_ylim(1.351, 1.474)


        if self.switches['save_figs']:
            plt.savefig('figures/'+'w0_mO_vs_ea_'+self.model+'_zoom.pdf',transparent=True)
            self.ax_cont.set_ylim(1.351, 1.559)
            plt.savefig('figures/'+'w0_mO_vs_ea_'+self.model+'.pdf',transparent=True)
            self.ax_cont.set_ylim(1.351, 1.474)


    def plot_vs_eps_pi(self,shift_points,eps='l'):
        self.shift    = shift_points
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
        y_conv['NLO']    = []
        y_conv['NNLO']   = []
        y_conv['NNNLO']  = []
        nlo_lst    = [t for t in self.model_list if 'nlo' in t and not any(n in t for n in ['n2lo', 'n3lo'])]
        n2lo_lst   = [t for t in self.model_list if 'n2lo' in t and not any(n in t for n in ['n3lo'])]
        n3lo_lst  = [t for t in self.model_list if 'n3lo' in t]
        nlo_fit    = chipt.FitModel(nlo_lst, _fv=False, _FF=self.FF)
        n2lo_fit   = chipt.FitModel(nlo_lst+n2lo_lst, _fv=False, _FF=self.FF)
        n3lo_fit  = chipt.FitModel(nlo_lst+n2lo_lst+n3lo_lst, _fv=False, _FF=self.FF)
        x_plot = []
        mpi_range = np.sqrt(np.arange(100, 411**2, 411**2/200))
        ms_range  = np.sqrt(np.arange(400**2, 900**2, (900**2 - 400**2)/200))
        m_range   = {'l':mpi_range, 's':ms_range}

        for a_m in m_range[eps]:
            if eps == 'l':
                x_plot.append(a_m**2 / self.shift_xp['p']['Lam_'+self.FF]**2)
                self.shift_xp['p']['mpi'] = a_m
                # the fitter builds m_ss^2 = 2mK^2 - mpi^2
                # so we need to trick it by moving mK^2 away from the phys value
                mkSq  = shift_points['p']['mk']**2
                mkSq += 0.5*a_m**2
                mkSq -= 0.5*shift_points['p']['mpi']**2
                self.shift_xp['p']['mk']  = np.sqrt(mkSq)
            elif eps == 's':
                # for s, we shift the kaon mass
                self.shift_xp['p']['mk']  = a_m
                self.shift_xp['p']['mpi'] = shift_points['p']['mpi']
                mssSq = 2 * a_m**2 - shift_points['p']['mpi']
                x_plot.append(mssSq / self.shift_xp['p']['Lam_'+self.FF]**2)
            self.shift_xp['p']['aw0'] = 0
            y_plot['a00'].append(self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p']))
            y_conv['NLO'].append(self.fitEnv._fit_function(nlo_fit, self.shift_xp['x'], self.shift_xp['p']))
            y_conv['NNLO'].append(self.fitEnv._fit_function(n2lo_fit, self.shift_xp['x'], self.shift_xp['p']))
            y_conv['NNNLO'].append(self.fitEnv._fit_function(n3lo_fit, self.shift_xp['x'], self.shift_xp['p']))
            for aa in ['a15','a12','a09']:
                self.shift_xp['p']['aw0'] = self.fitEnv.p[(self.aw0_keys[aa],'aw0')]
                y_plot[aa].append(self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p']))
            self.shift_xp['p']['aw0'] = aw0_a06
            y_plot['a06'].append(self.fitEnv._fit_function(self.shift_fit, self.shift_xp['x'], self.shift_xp['p']))

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
        ax_x.fill_between(x, y['a00']-dy['a00'], y['a00']+dy['a00'],
            #facecolor='None',edgecolor='#b36ae2',hatch='/')
            facecolor='None',edgecolor='k',hatch='/')
        for aa in ['a15','a12','a09','a06']:
            ax_x.fill_between(x, y[aa]-dy[aa], y[aa]+dy[aa],
                color=colors[aa],alpha=.3)
            #ax_x.plot(x, y[aa], color=colors[aa])

        # plot physical eps_pi**2
        if eps == 'l':
            ax_x.axvline(eps_pisq_phys.mean,linestyle='--',color='#a6aaa9')
            ax_x.axvspan(eps_pisq_phys.mean -eps_pisq_phys.sdev, eps_pisq_phys.mean +eps_pisq_phys.sdev,
                alpha=0.4, color='#a6aaa9')
            ax_x.text(0.06, 1.375, r'%s' %(self.model.replace('_','\_')),\
                horizontalalignment='left', verticalalignment='center', \
                fontsize=fs_text, bbox={'facecolor':'None','boxstyle':'round'})


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
                loc=2
            elif eps == 's':
                loc=2
        else:
            loc=1
        ax_x.legend(handles, labels, loc=loc, ncol=2, columnspacing=0, handletextpad=0.1, fontsize=fs_leg)

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
        ax_x.set_ylim(1.351, 1.559)

        if self.switches['save_figs']:
            plt.savefig('figures/'+fig_name+'.pdf',transparent=True)
        if eps == 'l':
            self.ax_l = ax_x
        elif eps == 's':
            self.ax_s = ax_x

        # Convergence plot
        order_list = []
        if 'n3lo' in self.model:
            order_list = order_list + ['NLO','NNLO','NNNLO']
        elif 'n2lo' in self.model:
            order_list = order_list + ['NLO','NNLO']
        elif 'nlo' in self.model:
            order_list = order_list + ['NLO']
        self.fig_conv = plt.figure(fig_name+'_convergence_'+self.model, figsize=fig_size)
        self.ax_conv  = plt.axes(plt_axes)
        labels = {'NLO':'NLO', 'NNLO':r'N$^2$LO', 'NNNLO':r'N$^3$LO'}
        for order in order_list:
            mean = np.array([k.mean for k in y_conv[order]])
            sdev = np.array([k.sdev for k in y_conv[order]])
            self.ax_conv.fill_between(x, mean-sdev, mean+sdev, alpha=.4, label=labels[order])
        self.ax_conv.set_xlabel(eps_FF[self.FF],fontsize=fs_text)
        self.ax_conv.set_xlim(xlim_FF[self.FF])
        self.ax_conv.set_ylabel(r'$w_0 m_\Omega$',fontsize=fs_text)
        self.ax_conv.set_ylim(1.351, 1.559)
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
            label = ''
            if a_ens in self.switches['ensembles_fit']:
                c = colors[a_ens.split('m')[0]]
                alpha = 1
            else:
                c = 'k'
                alpha = 0.4
            s = shapes['m'+a_ens.split('m')[1][0:3]]
            if p_type == 'ea':
                x  = (self.fitEnv.p[(a_ens, 'aw0')] / 2)**2
                dx = 0
            elif p_type == 'l':
                x  = (self.fitEnv.p[(a_ens, 'mpi')] / self.fitEnv.p[(a_ens, 'Lam_'+self.FF)])**2
                dx = 0
            elif p_type == 's':
                x  = 2*(self.fitEnv.p[(a_ens, 'mk')]/self.fitEnv.p[(a_ens, 'Lam_'+self.FF)])**2
                x += -(self.shift_xp['p']['mpi'] / self.shift_xp['p']['Lam_'+self.FF])**2
                dx = 0
            if a_ens in self.switches['ensembles_fit']:
                mfc = 'white'
            else:
                mfc = 'None'
            if p_type == 'ea' and a_ens in eps_a_labels:
                label = eps_a_labels[a_ens]
            elif p_type in ['l','s'] and a_ens in l_labels:
                label = l_labels[a_ens]
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
            if aa != 'a06':
                self.shift_xp['p']['aw0'] = self.fitEnv.p[(self.aw0_keys[aa],'aw0')]
            else:
                self.shift_xp['p']['aw0'] = aw0_a06
            self.shift_xp['x']['alphaS'] = self.fitEnv.x[a_ens]['alphaS']
            if p_type in ['l','s']:
                if p_type == 'l':
                    self.shift_xp['p']['mpi'] = self.fitEnv.p[(a_ens,'mpi')] / self.fitEnv.p[(a_ens, 'Lam_'+self.FF)]
                    mkSq  = self.shift['p']['mk']**2 / self.shift['p']['Lam_'+self.FF]**2
                    mkSq += 0.5*self.shift_xp['p']['mpi']**2
                    mkSq -= 0.5*self.shift['p']['mpi']**2 / self.shift['p']['Lam_'+self.FF]**2
                    self.shift_xp['p']['mk']  = np.sqrt(mkSq)
                elif p_type == 's':
                    self.shift_xp['p']['mpi'] = self.shift['p']['mpi'] / self.shift['p']['Lam_'+self.FF]
                    mkSq  = (self.fitEnv.p[(a_ens,'mk')] / self.fitEnv.p[(a_ens, 'Lam_'+self.FF)])**2
                    mkSq -= 0.5*(self.fitEnv.p[(a_ens,'mpi')] / self.fitEnv.p[(a_ens, 'Lam_'+self.FF)])**2
                    mkSq += 0.5*self.shift_xp['p']['mpi']**2
                    self.shift_xp['p']['mk']  = self.fitEnv.p[(a_ens,'mk')] / self.fitEnv.p[(a_ens, 'Lam_'+self.FF)]
                    self.shift_xp['p']['mk'] = np.sqrt(mkSq)
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
            r'a12m220: $\delta_{\rm FV}^{{\rm N^2LO}\ \chi{\rm PT}}(\epsilon_\pi^2, m_\pi L)$', \
            horizontalalignment='center', verticalalignment='center', \
            rotation=angle, fontsize=fs_text-1)

        if self.switches['save_figs']:
            plt.savefig('figures/'+'w0_mO_vs_mL_'+self.model+'.pdf',transparent=True)

class w0Plots:

    def __init__(self, model, model_list, fitEnv, fit_result, switches):
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
        self.aw0_keys = {'a15':'a15m135XL','a12':'a12m130','a09':'a09m135'}

def plot_w0(model, model_list, fitEnv, fit_result, switches, shift_point):
    eps_pisq_phys = (gv.gvar(shift_point['p']['mpi'] / shift_point['p']['Lam_F']))**2
    og_fit    = chipt.FitModel(model_list, _fv=('FV' in model), _FF='F')
    shift_fit = chipt.FitModel(model_list, _fv=False, _FF='F')

    mpi_range = np.sqrt(np.arange(100, 411**2, 411**2/100))

    fig = plt.figure('w0_a',figsize=fig_size2)
    ylim = {'a15':(1.081,1.159), 'a12':(1.301,1.434), 'a09':(1.701,1.999), 'a06':(2.51,3.09)}
    for i_a, aa in enumerate(switches['w0_aa_lst']):
        ax = plt.axes([0.1,.07+i_a*.232,.895,.232])
        # fit band
        shift_xp = copy.deepcopy(shift_point)
        shift_xp['p']['w0_0'] = fit_result.p[(aa,'w0_0')]
        x_plot = []
        y_plot = []
        for a_m in mpi_range:
            l_FSq = (a_m / shift_xp['p']['Lam_F'])**2
            mkSq = shift_xp['p']['mk'] **2
            # shift mkSq so that m_ssSq is constant
            mkSq += 0.5 * a_m**2
            mkSq -= 0.5* shift_xp['p']['mpi'] **2
            shift_xp['p']['mpi'] = a_m
            shift_xp['p']['mk']  = np.sqrt(mkSq)
            x_plot.append(l_FSq)
            y_plot.append(fitEnv._w0_function(shift_fit, shift_xp['x'], shift_xp['p']))
        #print(x_plot)
        y  = np.array([k.mean for k in y_plot])
        dy = np.array([k.sdev for k in y_plot])
        x  = np.array([k.mean for k in x_plot])
        ax.fill_between(x, y-dy, y+dy, color=colors[aa],alpha=0.4)
        # data
        xx   = []
        yy   = []
        y_og = []
        for ens in switches['ensembles']:
            if aa in ens:
                shift_xp = copy.deepcopy(shift_point)
                shift_xp['p']['w0_0'] = fit_result.p[(aa,'w0_0')]
                # compute shift from s_F to s_F^phys
                og_priors = dict()
                for k, v in fitEnv.p.items():
                    if type(k) == tuple and k[0] == ens:
                        if k in fit_result.p:
                            og_priors[k[1]] = fit_result.p[k]
                        else:
                            og_priors[k[1]] = v
                for k in fit_result.p:
                    if isinstance(k,str):
                        og_priors[k] = fit_result.p[k]
                    elif type(k) == tuple and k[0] == aa:
                        og_priors[k[1]] = fit_result.p[(aa,'w0_0')]
                # mpi from ensemble
                l_FSq = (fitEnv.p[(ens,'mpi')] / fitEnv.p[(ens, 'Lam_F')])**2
                # shift mK to phys
                K_FSq  = (shift_xp['p']['mk'] / shift_xp['p']['Lam_F'])**2
                # shift kaon so that l_S is at physical value
                # s_F^2 = 2*K_F^2 - l_F^2
                K_FSq -= 0.5 * shift_xp['p']['mpi']**2 / shift_xp['p']['Lam_F']**2
                K_FSq += 0.5 * l_FSq

                s_FSq  = 2*K_FSq - (shift_xp['p']['mpi'] / shift_xp['p']['Lam_F'])**2
                s_FSq2 = 2*K_FSq - l_FSq
                #print(ens,'mss^2',s_FSq,s_FSq2)

                shift_xp['p']['mpi'] = np.sqrt(l_FSq)
                shift_xp['p']['mk']  = np.sqrt(K_FSq)
                shift_xp['p']['Lam_F'] = 1

                y_fit   = fitEnv._w0_function(og_fit, fitEnv.x[ens], og_priors)
                y_shift = fitEnv._w0_function(shift_fit, shift_xp['x'], shift_xp['p'])
                # get data
                xx.append(l_FSq)
                y_og.append(fit_result.y[ens])
                yy.append(fit_result.y[ens] + y_shift - y_fit)
        x  = [k.mean for k in xx]
        y  = [k.mean for k in y_og]
        dy = [k.sdev for k in y_og]
        ax.errorbar(x,y,yerr=dy, linestyle='None',
            marker='o',c='k',mfc='None',alpha=0.5,label=r'$w_0 / a_{%s}(l_F^{\rm ens},s_F^{\rm ens})$' %aa[1:])
        y  = [k.mean for k in yy]
        dy = [k.sdev for k in yy]
        ax.errorbar(x,y,yerr=dy, linestyle='None',
            marker='s',c=colors[aa],mfc='None',label=r'$w_0 / a_{%s}(l_F^{\rm ens},s_F^{\rm phys})$' %aa[1:])
        ax.legend(loc=3,fontsize=fs_leg)
        if i_a == 0:
            ax.tick_params(bottom=True, labelbottom=True, top=True, direction='in')
            ax.set_xlabel(r'$l_F^2 = m_\pi^2 / (4\pi F_\pi)^2$',fontsize=fs_text)
        else:
            ax.tick_params(bottom=True, labelbottom=False, top=True, direction='in')
        ax.set_ylabel(r'$w_0 / a_{%s}$' %aa[1:],fontsize=fs_text)
        ax.set_xlim(0,0.094)
        ax.set_ylim(ylim[aa])
        ax.axvline(eps_pisq_phys.mean,linestyle='--',color='#a6aaa9')
        ax.axvspan(eps_pisq_phys.mean -eps_pisq_phys.sdev, eps_pisq_phys.mean +eps_pisq_phys.sdev,
            alpha=0.4, color='#a6aaa9')

    if switches['save_figs']:
        plt.savefig('figures/w0_a_vs_lFSq.pdf',transparent=True)
