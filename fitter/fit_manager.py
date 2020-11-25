import lsqfit
import numpy as np
import gvar as gv
import time
import matplotlib
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d.axes3d import Axes3D
import os

# Set defaults for plots
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = [6.75, 6.75/1.618034333]
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True

import fitter.fitter as fit

class fit_manager(object):

    def __init__(self, phys_point_data, fit_data=None, bs_data=None, prior=None, prior_interpolation=None, model_info=None, **kwargs):
        if model_info['chiral_cutoff'] == 'mO':
            params_list = ['mpi', 'mk', 'mO']
        elif model_info['chiral_cutoff'] == 'Fpi': 
            params_list = ['mpi', 'mk', 'mO', 'Fpi']
        
        if fit_data is not None:
            pass
        elif bs_data is not None:
            fit_data = {}
            for ens in sorted(list(bs_data)):
                #if ens not in ['a06m135L', 'a15m135XL', 'a12m130', 'a09m135']:
                fit_data[ens] = gv.BufferDict()
                for param in params_list:
                    fit_data[ens][param] = bs_data[ens][param]

                # Deflate uncertainty on a06m310L
                # (for testing purposes only)
                #if False: #ens in ['a06m310L', 'a12m220_ms']:
                #    mean = np.mean(bs_data[ens]['mO'])
                #    fluctuations = (bs_data[ens]['mO'] - mean)/np.sqrt(2)
                #
                #    fit_data[ens]['mO'] = mean + fluctuations


                fit_data[ens] = gv.dataset.avg_data(fit_data[ens], bstrap=True)
                for param in ['mO', 'mpi', 'mk', 'Fpi']: 
                    if param in fit_data[ens]:
                        fit_data[ens][param] = fit_data[ens][param] - gv.mean(fit_data[ens][param]) + bs_data[ens][param][0]


                if model_info['chiral_cutoff'] == 'mO':
                    fit_data[ens]['lam_chi'] = fit_data[ens]['mO']
                    phys_point_data['lam_chi'] = phys_point_data['mO']
                elif model_info['chiral_cutoff'] == 'Fpi':
                    fit_data[ens]['lam_chi'] = 4 *np.pi *fit_data[ens]['Fpi']
                    phys_point_data['lam_chi'] = 4 *np.pi *phys_point_data['Fpi']

                #if ens == 'a15m135XL':
                #    fit_data[ens]['mO'] = gv.gvar(gv.mean(fit_data[ens]['mO']), gv.mean(fit_data[ens]['mO'])/np.sqrt(2))
                fit_data[ens]['a/w'] = bs_data[ens]['a/w']
                fit_data[ens]['L'] = gv.gvar(bs_data[ens]['L'], bs_data[ens]['L']/10**6)
                fit_data[ens]['alpha_s'] = gv.gvar(bs_data[ens]['alpha_s'], bs_data[ens]['alpha_s']/10**6)
                #print(ens, fit_data[ens]['mO'])


        self.ensembles = list(sorted(fit_data))
        self.model_info = model_info
        self.fit_data = fit_data
        self.fitter = fitter_dict(prior=prior, fit_data=fit_data, prior_interpolation=prior_interpolation)[model_info]
        self.prior_interpolation = prior_interpolation
        
        self._input_prior = prior
        self._phys_point_data = phys_point_data
        self._fit = None


    def __str__(self):
        output = "Model: %s" %(self.model)
        output += "\nw0: %s\n\n" %(self.w0)

        for a_xx in ['a06', 'a09', 'a12', 'a15']:
            w0_a = self.interpolate_w0a(latt_spacing=a_xx)
            output += 'w0/{}: {}'.format(a_xx, w0_a).ljust(22)  + '=> %s/fm: %s\n'%(a_xx, self.w0 / w0_a)

        output += '\nParameters:\n'
        my_str = self.fit.format(pstyle='m')
        for item in my_str.split('\n'):
            for key in self.fit_keys:
                re = key+' '
                if re in item:
                    output += item + '\n'

        output += '\n'
        output += self.fit.format(pstyle=None)

        output += '\nError Budget:\n'
        max_len = np.max([len(key) for key in self.error_budget])
        for key in {k: v for k, v in sorted(self.error_budget.items(), key=lambda item: item[1], reverse=True)}:
            output += '  '
            output += key.ljust(max_len+1)
            output += '{: .1%}\n'.format((self.error_budget[key]/self.w0.sdev)**2).rjust(7)
        return output

    @property
    def error_budget(self):
        return self._get_error_budget()

    def _get_error_budget(self, verbose=False, **kwargs):
        output = {}

        # Fill these out
        disc_keys = ['A_a', 'A_alpha', 'A_aa', 'A_al', 'A_as', 'A_aaa', 'A_aal', 'A_aas', 'A_all', 'A_als', 'A_ass']
        chiral_keys = ['wm0', 'A_l', 'A_s', 'A_ll', 'A_ls', 'A_ss', 'A_ll_g', 'A_lll', 'A_lls', 'A_lss', 'A_sss', 'A_lll_g', 'A_lls_g', 'A_lll_gg']
        phys_keys = list(self.phys_point_data)
        stat_key = 'lam_chi' # Since the input data is correlated, only need a single variable as a proxy for all


        if verbose:
            inputs = {}

            # xpt/chiral contributiosn
            inputs.update({str(param)+' [xpt]' : self.prior[param] for param in chiral_keys if param in self.prior})
            inputs.update({str(param)+' [disc]' : self.prior[param] for param in disc_keys if param in self.prior})

            # phys point contributions
            inputs.update({str(param)+' [pp]' : self.phys_point_data[param] for param in list(phys_keys)})
            del inputs['lam_chi [pp]']

            # stat contribtions
            inputs.update({'x [stat]' : self._get_prior(stat_key) , 'y [stat]' : self.fitter.y})
            
            if kwargs is None:
                kwargs = {}
            kwargs.setdefault('percent', False)
            kwargs.setdefault('ndecimal', 10)
            kwargs.setdefault('verify', True)

            return gv.fmt_errorbudget(outputs={'w0' : self.w0}, inputs=inputs, **kwargs)
        else:
            output['disc'] = self.w0.partialsdev(
                [self.prior[key] for key in disc_keys if key in self.prior]
            )
            output['chiral'] = self.w0.partialsdev(
                [self.prior[key] for key in chiral_keys if key in self.prior]
            )
            output['pp_input'] = self.w0.partialsdev(
                [self.phys_point_data[key] for key in phys_keys]
            )
            output['stat'] = self.w0.partialsdev(
                [self._get_prior(stat_key), self.fitter.y]
                #self.fitter.y
            )

            return output


    @property
    def fit(self):
        if self._fit is None:
            print("Making fit...")

            start_time = time.time()
            temp_fit = self.fitter.fit
            end_time = time.time()

            self._fit = temp_fit
            print("Time (s): ",  end_time - start_time, '\n----\n')

        return self._fit

    @property
    def fit_info(self):
        fit_info = {
            'name' : self.model,
            'w0' : self.w0,
            'logGBF' : self.fit.logGBF,
            'chi2/df' : self.fit.chi2 / self.fit.dof,
            'Q' : self.fit.Q,
            'phys_point' : self.phys_point_data,
            'error_budget' : self.error_budget,
            'prior' : self.prior,
            'posterior' : self.posterior,
        }

        return fit_info

    # Returns names of LECs in prior/posterior
    @property
    def fit_keys(self):
        keys1 = list(self._input_prior.keys())
        keys2 = list(self.fit.p.keys())
        parameters = np.intersect1d(keys1, keys2)
        return parameters

    @property
    def model(self):
        return self.model_info['name']

    @property
    def phys_point_data(self):
        return self._get_phys_point_data()

    # need to convert to/from lattice units
    def _get_phys_point_data(self, parameter=None):
        if parameter is None:
            return self._phys_point_data.copy()
        else:
            return self._phys_point_data[parameter]

    @property
    def posterior(self):
        return self._get_posterior()

    # Returns dictionary with keys fit parameters, entries gvar results
    def _get_posterior(self, param=None):
        if param is None:
            return {param : self.fit.p[param] for param in self.fit_keys}
        elif param == 'all':
            return self.fit.p
        else:
            return self.fit.p[param]

    @property
    def prior(self):
        return self._get_prior()

    def _get_prior(self, param=None):
        if param is None:
            #if self._prior is None:
            #    self._prior = {param : self.fit.prior[param] for param in self.fit_keys}
            #return self._prior
            return {param : self.fit.prior[param] for param in self.fit_keys}
        elif param == 'all':
            return self.fit.prior
        else:
            return self.fit.prior[param]

    @property
    def w0(self):
        return self.fitfcn(fit_data=self.phys_point_data.copy()) / self.phys_point_data['mO'] *self.phys_point_data['hbarc']

    def _extrapolate_to_ens(self, ens=None, phys_params=None):
        if phys_params is None:
            phys_params = []

        extrapolated_values = {}
        for j, ens_j in enumerate(self.ensembles):
            posterior = {}
            xi = {}
            if ens is None or (ens is not None and ens_j == ens):
                for param in self.fit.p:
                    shape = self.fit.p[param].shape
                    if param in phys_params:
                        a_fm = self.w0 / self.interpolate_w0a(ens_j[:3])
                        posterior[param] = self.phys_point_data[param] / self.phys_point_data['hbarc'] *a_fm
                    elif shape == ():
                        posterior[param] = self.fit.p[param]
                    else:
                        posterior[param] = self.fit.p[param][j]

                if 'alpha_s' in phys_params:
                    posterior['alpha_s'] = self.phys_point_data['alpha_s']

                if 'xi_l' in phys_params:
                    xi['l'] = self.phys_point_data['mpi']**2 / self.phys_point_data['lam_chi']**2
                if 'xi_s' in phys_params:
                    xi['s'] = (2 *self.phys_point_data['mk']**2 - self.phys_point_data['mpi']**2)/ self.phys_point_data['lam_chi']**2
                if 'xi_a' in phys_params:
                    xi['a'] = 0

                if ens is not None:
                    return self.fitfcn(posterior=posterior, fit_data={}, xi=xi)
                else:
                    extrapolated_values[ens_j] = self.fitfcn(posterior=posterior, fit_data={}, xi=xi)
        return extrapolated_values


    def _fmt_key_as_latex(self, key):
        convert = {
            # data parameters
            'a/w0' : r'$a$ (fm)',
            'L' : r'$L$ (fm)',
            'mpi' : r'$m_\pi$',
            'mk' : r'$m_K$',

            'V' : r'$e^{-m L} / \sqrt{m L}$',
        }

        if key in convert.keys():
            return convert[key]
        else:
            return key


    def fitfcn(self, fit_data=None, posterior=None, xi=None, debug=False):
        if fit_data is None:
            fit_data = self.phys_point_data.copy()
        if posterior is None:
            posterior = self.posterior.copy()

        model_info = self.model_info.copy()

        model = self.fitter._make_models()[0]
        return model.fitfcn(p=posterior, fit_data=fit_data, xi=xi, debug=debug)


    def fitfcn_interpolation(self, latt_spacing, fit_data=None, posterior=None, xi=None, simultaneous=False):
        if fit_data is None:
            fit_data = self.phys_point_data.copy()

        param_keys = set(list(self.prior_interpolation)).intersection(set(list(self.fitter.fit_interpolation(simultaneous).p)))
        posterior = {param : self.fitter.fit_interpolation(simultaneous).p[param] 
            for param in self.prior_interpolation if param in param_keys}

        model = self.fitter._make_models(interpolation=True, w_a_data=None)[0]
        return model.fitfcn(p=posterior, fit_data=fit_data, xi=xi, latt_spacing=latt_spacing)


    def fmt_error_budget(self, **kwargs):
        return self._get_error_budget(verbose=True, **kwargs)


    def interpolate_w0a(self, latt_spacing, simultaneous=False):
        return self.fitfcn_interpolation(latt_spacing=latt_spacing, simultaneous=simultaneous)


    def optimize_prior(self, empbayes_grouping='order'):
        temp_prior = self.fitter._make_empbayes_fit(empbayes_grouping).prior
        prior = gv.BufferDict()
        for key in self.fit_keys:
            prior[key] = temp_prior[key]

        return prior


    # Takes keys from posterior (eg, 'L_5' and 'L_4')
    def plot_error_ellipsis(self, x_key, y_key):
        x = self._get_posterior(x_key)
        y = self._get_posterior(y_key)


        fig, ax = plt.subplots()

        corr = '{0:.3g}'.format(gv.evalcorr([x, y])[0,1])
        std_x = '{0:.3g}'.format(gv.sdev(x))
        std_y = '{0:.3g}'.format(gv.sdev(y))
        text = ('$R_{x, y}=$ %s\n $\sigma_x =$ %s\n $\sigma_y =$ %s'
                % (corr,std_x,std_y))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        C = gv.evalcov([x, y])
        eVe, eVa = np.linalg.eig(C)
        for e, v in zip(eVe, eVa.T):
            plt.plot([gv.mean(x)-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x)],
                     [gv.mean(y)-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y)],
                     'k-', lw=2)

        #plt.scatter(x-np.mean(x), y-np.mean(y), rasterized=True, marker=".", alpha=100.0/self.bs_N)
        #plt.scatter(x, y, rasterized=True, marker=".", alpha=100.0/self.bs_N)

        plt.grid()
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.xlabel(self._fmt_key_as_latex(x_key), fontsize = 24)
        plt.ylabel(self._fmt_key_as_latex(y_key), fontsize = 24)

        fig = plt.gcf()
        plt.close()
        return fig


    def plot_fit(self, param, show_legend=True, ylim=None):

        x = {}
        y = {}
        c = {}
            
        #fig = plt.figure(figsize=(6.75, 6.75/1.618034333))
        plt.axes([0.145,0.145,0.85,0.85])

        colors = {
            '06' : '#6A5ACD',#'#00FFFF',
            '09' : '#51a7f9',
            '12' : '#70bf41',
            '15' : '#ec5d57',
        }

        latt_spacings = {a_xx[1:] : 1/self.interpolate_w0a(a_xx) for a_xx in ['a06', 'a09' , 'a12', 'a15']}
        latt_spacings['00'] = gv.gvar(0, 0)

        for j, xx in enumerate(reversed(latt_spacings)):
            xi = {}
            phys_data = self.phys_point_data
            phys_data['a/w'] = latt_spacings[xx]

            min_max = lambda mydict : (gv.mean(np.nanmin([mydict[key] for key in mydict.keys()])), 
                                       gv.mean(np.nanmax([mydict[key] for key in mydict.keys()])))


            if param in ['pi', 'l', 'p']:
                plt.axvline(gv.mean((phys_data['mpi'] / phys_data['lam_chi'])**2), ls='--', label='phys. point')
                min_max = min_max({ens : (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2 for ens in self.ensembles})
                xi['l'] = np.linspace(0.0001, min_max[1])
                x_fit = xi['l']

            elif param in ['k', 's']:
                plt.axvline(gv.mean(((2 *phys_data['mk']**2 - phys_data['mpi']**2) / phys_data['lam_chi']**2)), ls='--', label='Phys point')
                min_max = min_max({ens : (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2 for ens in self.ensembles})
                xi['s'] = np.linspace(min_max[0], min_max[1])
                x_fit = xi['s']

            elif param == 'a':
                plt.axvline(0, label='phys. point', ls='--')
                min_max = min_max({ens : self.fit_data[ens]['a/w']**2 / 4 for ens in self.ensembles})
                xi['a'] = np.linspace(0, min_max[1])
                x_fit = xi['a']

                
            y_fit = self.fitfcn(posterior=self.posterior, fit_data=phys_data, xi=xi)

            # For LO fits
            if not hasattr(y_fit, "__len__"):
                y_fit = np.repeat(y_fit, len(x_fit))


            pm = lambda g, k : gv.mean(g) + k *gv.sdev(g)
            if xx != '00' and param != 'a':
                plt.fill_between(pm(x_fit, 0), pm(y_fit, -1), pm(y_fit, +1), color=colors[xx], alpha=0.4)
            elif xx == '00' and param != 'w0mO':
                plt.fill_between(pm(x_fit, 0), pm(y_fit, -1), pm(y_fit, +1), facecolor='None', edgecolor='k', alpha=0.6, hatch='/')
            else:
                pass

        for ens in self.ensembles:

            if param in ['pi', 'l', 'p']:
                x[ens] = (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2
                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['xi_s', 'alpha_s']))
                label = r'$\xi_l$'
                if self.model_info['chiral_cutoff'] == 'Fpi':
                    label = r'$l^2_F = m_\pi^2 / (4 \pi F_\pi)^2$'

            elif param in ['k', 's']:
                x[ens] = (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2
                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['xi_l', 'alpha_s']))
                label = r'$\xi_s$'
                if self.model_info['chiral_cutoff'] == 'Fpi':
                    label = r'$s^2_F$'

            elif param == 'a':
                x[ens] = self.fit_data[ens]['a/w']**2 / 4
                y[ens] = (self.shift_latt_to_phys(ens=ens, phys_params=['xi_l', 'xi_s', 'alpha_s']))
                label = '$\epsilon^2_a = (a / 2 w_0)^2$'

        for ens in reversed(self.ensembles):
            C = gv.evalcov([x[ens], y[ens]])
            eVe, eVa = np.linalg.eig(C)
            for e, v in zip(eVe, eVa.T):
                plt.plot([gv.mean(x[ens])-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x[ens])],
                        [gv.mean(y[ens])-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y[ens])],
                        color=colors[ens[1:3]], alpha=1.0, lw=2)
                plt.plot(gv.mean(x[ens]), gv.mean(y[ens]), 
                         color=colors[ens[1:3]], marker='o', mec='w', zorder=3)

        if show_legend:
            if param in ['pi', 'l', 'p']:
                labels = [
                    r'$a_{06}(l_F,s_F^{\rm phys})$',
                    r'$a_{09}(l_F,s_F^{\rm phys})$',
                    r'$a_{12}(l_F,s_F^{\rm phys})$',
                    r'$a_{15}(l_F,s_F^{\rm phys})$'
                ]
            elif param in ['k', 's']:
                labels = [
                    r'$a_{06}(l_F^{\rm phys},s_F)$',
                    r'$a_{09}(l_F^{\rm phys},s_F)$',
                    r'$a_{12}(l_F^{\rm phys},s_F)$',
                    r'$a_{15}(l_F^{\rm phys},s_F)$'
                ]
            elif param == 'a':
                labels = [
                    r'$a_{06}(l_F^{\rm phys},s_F^{\rm phys})$',
                    r'$a_{09}(l_F^{\rm phys},s_F^{\rm phys})$',
                    r'$a_{12}(l_F^{\rm phys},s_F^{\rm phys})$',
                    r'$a_{15}(l_F^{\rm phys},s_F^{\rm phys})$'
                ]
            handles = [
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['06']),
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['09']),
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['12']),
                plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors['15'])
            ]
            plt.legend(handles=handles, labels=labels, ncol=2)#, bbox_to_anchor=(0,1), loc='lower left')

        #plt.grid()
        plt.xlabel(label)
        plt.ylabel('$w_0 m_\Omega$')

        if ylim is not None:
            plt.ylim(ylim)

        fig = plt.gcf()
        plt.close()
        return fig


    def plot_interpolation(self, latt_spacing, param='p'):
        x = {}
        y = {}
        c = {}
            
        #fig = plt.figure(figsize=(6.75, 6.75/1.618034333))
        #plt.axes([0.145,0.145,0.85,0.85])

        colors = {
            'a06' : '#6A5ACD',#'#00FFFF',
            'a09' : '#51a7f9',
            'a12' : '#70bf41',
            'a15' : '#ec5d57',
        }
        ensembles = [ens for ens in self.ensembles if ens[:3] == latt_spacing]

        xi = {}
        phys_data = self.phys_point_data.copy()

        min_max = lambda mydict : (gv.mean(np.nanmin([mydict[key] for key in mydict.keys()])), 
                                    gv.mean(np.nanmax([mydict[key] for key in mydict.keys()])))


        if param in ['pi', 'l', 'p']:
            plt.axvline(gv.mean((phys_data['mpi'] / phys_data['lam_chi'])**2), ls='--', label='phys. point')
            min_max = min_max({ens : (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2 for ens in ensembles})
            xi['l'] = np.linspace(0.0001, min_max[1])
            x_fit = xi['l']

            xlabel = r'$\xi_l$'
            if self.model_info['chiral_cutoff'] == 'Fpi':
                xlabel = r'$l^2_F = m_\pi^2 / (4 \pi F_\pi)^2$'

        elif param in ['k', 's']:
            plt.axvline(gv.mean(((2 *phys_data['mk']**2 - phys_data['mpi']**2) / phys_data['lam_chi']**2)), ls='--', label='Phys point')
            min_max = min_max({ens : (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2 for ens in ensembles})
            xi['s'] = np.linspace(min_max[0], min_max[1])
            x_fit = xi['s']

            xlabel = r'$\xi_s$'
            if self.model_info['chiral_cutoff'] == 'Fpi':
                xlabel = r'$s^2_F$'
        
        y_fit = self.fitfcn_interpolation(fit_data=phys_data, xi=xi, latt_spacing=latt_spacing)

        # For LO fits
        if not hasattr(y_fit, "__len__"):
            y_fit = np.repeat(y_fit, len(x_fit))
        pm = lambda g, k : gv.mean(g) + k *gv.sdev(g)
        plt.fill_between(pm(x_fit, 0), pm(y_fit, -1), pm(y_fit, +1), color=colors[latt_spacing], alpha=0.4)

        
        for ens in ensembles:
            xi = {}
            xi['l'] = (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2
            xi['s'] = (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2

            value_latt = 1 / self.fit_data[ens]['a/w']
            value_fit = self.fitfcn_interpolation(latt_spacing=latt_spacing, xi=xi)

            if param in ['pi', 'l', 'p']:
                x[ens] = (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2
                xi['s'] = (2 *phys_data['mk']**2 - phys_data['mpi']**2) / phys_data['lam_chi']**2
                label = r'$a_{%s}(l_F,s_F^{\rm phys})$'%(ens[1:3])

            elif param in ['k', 's']:
                xi['l'] = (phys_data['mpi'] / phys_data['lam_chi'])**2
                x[ens] = (2 *self.fit_data[ens]['mk']**2 - self.fit_data[ens]['mpi']**2) / self.fit_data[ens]['lam_chi']**2
                label = r'$a_{%s}(l_F^{\rm phys},s_F)$'%(ens[1:3])

            value_fit_phys = self.fitfcn_interpolation(latt_spacing=latt_spacing, xi=xi)

            y[ens] = value_latt + value_fit_phys - value_fit

            plt.errorbar(x=x[ens].mean, y=y[ens].mean, xerr=x[ens].sdev, yerr=y[ens].sdev, marker='o', color=colors[latt_spacing], capsize=0.0, mec='white',  elinewidth=3.0,)


        handles = [
            plt.errorbar([], [], 0, 0, marker='o', capsize=0.0, mec='white', mew=2.0, ms=8.0, elinewidth=3.0, color=colors[latt_spacing])
        ]
        plt.legend(handles=handles, labels=[label])#, bbox_to_anchor=(0,1), loc='lower left')


        plt.xlabel(xlabel)
        plt.ylabel('$w_0/a$')

        fig = plt.gcf()
        plt.close()
        return fig



    def plot_parameters(self, xparam, yparam=None):
        if yparam is None:
            yparam = 'w0mO'

        x = {}
        y = {}
        c = {}
            
        colors = {
            '06' : '#6A5ACD',#'#00FFFF',
            '09' : '#51a7f9',
            '12' : '#70bf41',
            '15' : '#ec5d57',
        }
        #c = {abbr : colors[ens[1:3]] for abbr in self.ensembles}

        for ens in self.ensembles:
            for j, param in enumerate([xparam, yparam]):
                if param == 'w0mO':
                    value = self.fit_data[ens]['mO'] / self.fit_data[ens]['a/w']
                    label = '$w_0 m_\Omega$'

                elif param == 'l':
                    value = (self.fit_data[ens]['mpi'] / self.fit_data[ens]['lam_chi'])**2
                    label= r'$\xi_l$'
                elif param == 's':
                    value = ((2 *self.fit_data[ens]['mk'] - self.fit_data[ens]['mpi'])/ self.fit_data[ens]['lam_chi'])**2
                    label = r'$\xi_s$'
                elif param == 'a':
                    value = self.fit_data[ens]['a/w']**2 / 4
                    label = r'$\xi_a$'
                elif param == 'mpi':
                    value = self.fit_data[ens]['mpi']
                    label = '$am_\pi$'

                if j == 0:
                    x[ens] = value
                    xlabel = label
                elif j == 1:
                    y[ens] = value
                    ylabel = label

        for ens in self.ensembles:
            C = gv.evalcov([x[ens], y[ens]])
            eVe, eVa = np.linalg.eig(C)
            for e, v in zip(eVe, eVa.T):
                plt.plot([gv.mean(x[ens])-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x[ens])],
                        [gv.mean(y[ens])-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y[ens])],
                        color=colors[ens[1:3]], lw=3, label=ens[0:3])
                plt.scatter(x[ens].mean, y[ens].mean, color=colors[ens[1:3]], edgecolors='k', zorder=5)


        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
            ncol=len(by_label), bbox_to_anchor=(0,1), loc='lower left')
        plt.grid()
        plt.xlabel(xlabel, fontsize = 24)
        plt.ylabel(ylabel, fontsize = 24)

        fig = plt.gcf()
        plt.close()
        return fig


    def plot_qq(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.fit.qqplot_residuals()

        fig = plt.gcf()
        plt.close()
        return fig


    def plot_residuals(self):

        y = 1
        labels = np.array([])
        for ens in self.ensembles:
            # fit result
            fit_value = self._extrapolate_to_ens(ens)
            x = gv.mean(fit_value)
            xerr = gv.sdev(fit_value)
            plt.errorbar(x=x, y=y, xerr=xerr, yerr=0.0,
                         color='deeppink', marker='o', capsize=0.0, mec='white', ms=10.0, alpha=0.6,
                         ecolor='deepskyblue', elinewidth=8.0, label='fit')
            y = y + 1

            # data
            data = self.fit_data[ens]['mO'] / self.fit_data[ens]['a/w']
            x = gv.mean(data)
            xerr = gv.sdev(data)

            plt.errorbar(x=x, y=y, xerr=xerr, yerr=0.0,
                         color='springgreen', marker='o', capsize=0.0, mec='white', ms=10.0, alpha=0.6,
                         ecolor='teal', elinewidth=8.0, label='data')

            labels = np.append(labels, str(""))
            y = y + 1

            labels = np.append(labels, str(ens))
            plt.axhline(y, ls='--')

            y = y + 1
            labels = np.append(labels, str(""))

        plt.yticks(1*list(range(len(labels))), labels, fontsize=15, rotation=45)
        plt.ylim(-1, y)
        plt.xlabel('$w_o m_\Omega$', fontsize = 24)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(reversed(labels), reversed(handles)))
        plt.legend(by_label.values(), by_label.keys(),
            ncol=len(by_label), bbox_to_anchor=(0,1), loc='lower left')

        fig = plt.gcf()
        plt.close()

        return fig


    def shift_latt_to_phys(self, ens=None, phys_params=None):
        value_shifted = {}
        for j, ens_j in enumerate(self.ensembles):
            if ens is None or ens_j == ens:
                value_latt = self.fit.y.values()[0][j]
                value_fit = self._extrapolate_to_ens(ens_j)
                value_fit_phys = self._extrapolate_to_ens(ens_j, phys_params)

                value_shifted[ens_j] = value_latt + value_fit_phys - value_fit
                if ens is not None:
                    return value_shifted[ens_j]

        return value_shifted


class fitter_dict(dict):
    def __init__(self, prior, prior_interpolation=None, fit_data=None, bs_data=None):
        if fit_data is not None:
            pass
        elif bs_data is not None:
            fit_data = {}
            for ens in sorted(list(bs_data)):
                fit_data[ens] = gv.BufferDict()
                for param in ['mpi', 'mk', 'mO', 'Fpi']:
                    if param in bs_data[ens]:
                        fit_data[ens][param] = bs_data[ens][param][:1]
                fit_data[ens] = gv.dataset.avg_data(fit_data[ens], bstrap=True)
                for param in ['mO', 'mpi', 'mk', 'Fpi']: 
                    fit_data[ens][param] = fit_data[ens][param] - gv.mean(fit_data[ens][param]) + bs_data[ens][param][0]

                fit_data[ens]['a/w'] = bs_data[ens]['a/w']
                fit_data[ens]['L'] = gv.gvar(bs_data[ens]['L'], bs_data[ens]['L']/10**6)
                fit_data[ens]['alpha_s'] = gv.gvar(bs_data[ens]['alpha_s'], bs_data[ens]['alpha_s']/10**6)

        self.fit_data = fit_data
        self.prior = prior
        self.prior_interpolation = prior_interpolation
        self.ensembles = sorted(list(fit_data))


    def __getitem__(self, model_info):
        if type(model_info) != str:
            key = model_info['name']
        else:
            key = model_info

        if key not in self:
            super().__setitem__(key, self._make_fitter(model_info))
            #print(model_info.values())
            return super().__getitem__(key)
            #return self.__getitem__(model_info)
        else:
            return super().__getitem__(key)


    def __str__(self):
        output = ''
        for model_name in list(self):
            output += '\n---\nModel: '+model_name+'\n\n'
            #output += str(self.__getitem__(key).fit)

            output += 'Parameters:\n'
            my_str = self.__getitem__(model_name).fit.format(pstyle='m')
            for item in my_str.split('\n'):
                for key in self.prior:
                    re = key+' '
                    if re in item:
                        output += item + '\n'

            output += '\n'
            output += self.__getitem__(model_name).fit.format(pstyle=None)

        return output


    def _make_fitter(self, model_info):
        prepped_data = self._make_fit_data(model_info)
        #print(model_info)
        fitter = fit.fitter(fit_data=prepped_data, prior=self.prior, prior_interpolation=self.prior_interpolation, model_info=model_info)
        return fitter


    def _make_fit_data(self, model_info):
        prepped_data = {}
        for param in list(self.fit_data[self.ensembles[0]]):
            prepped_data[param] = np.array([
                self.fit_data[ens][param] 
                for ens in self.ensembles
            ])
        if model_info['chiral_cutoff'] == 'mO':
            prepped_data['lam_chi'] = prepped_data['mO']
        elif model_info['chiral_cutoff'] == 'Fpi':
            prepped_data['lam_chi'] = 4 *np.pi *prepped_data['Fpi']
        return prepped_data



    