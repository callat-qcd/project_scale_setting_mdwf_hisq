import lsqfit
import numpy as np
import gvar as gv
import sys
import os

import fitter.special_functions as sf

class fitter(object):

    def __init__(self, prior, fit_data, model_info, prior_interpolation=None):
        self.prior = prior
        self.prior_interpolation = prior_interpolation
        self.fit_data = fit_data
        self.model_info = model_info.copy()

        if model_info['chiral_cutoff'] == 'Fpi':
            self.y = fit_data['mO'] / fit_data['a/w']
        else: # In Omega-type fits, shifting lam_chi will also shift y, leading to nonsensical fits 
            make_gvar = lambda g : gv.gvar(gv.mean(g), gv.sdev(g))
            self.y = make_gvar(fit_data['mO'] / fit_data['a/w'])

        
        # attributes of fitter object to fill later
        self.empbayes_grouping = None
        self._counter = {'iters' : 0, 'evals' : 0} # To force empbayes_fit to converge?
        self._empbayes_fit = None
        self._fit = None
        self._fit_interpolation = None
        self._simultaneous = False


    def __str__(self):
        return str(self.fit)


    @property
    def fit(self):
        if self._fit is None:
            models = self._make_models()
            y_data = {self.model_info['name'] : self.y}
            prior = self._make_prior()

            fitter = lsqfit.MultiFitter(models=models)
            fit = fitter.lsqfit(data=y_data, prior=prior, fast=False, mopt=False)

            self._fit = fit

        return self._fit

    #@property
    def fit_interpolation(self, simultaneous=None):
        if simultaneous is None:
            simultaneous = self._simultaneous

        if self._fit_interpolation is None or simultaneous != self._simultaneous:
            self._simultaneous = simultaneous
            #make_gvar = lambda g : gv.gvar(gv.mean(g), gv.sdev(g))
            #y_data = make_gvar(1 / self.fit_data['a/w'])

            make_gvar = lambda g : gv.gvar(gv.mean(g), gv.sdev(g))
            w_a_data = make_gvar(1 / self.fit_data['a/w'])
            y_data = {self.model_info['name']+'_interpolation' : w_a_data }
            if simultaneous:
                y_data[self.model_info['name']] = self.y


            models = self._make_models(interpolation=True, w_a_data=w_a_data, simultaneous=simultaneous)
            prior = self._make_prior(interpolation=True, simultaneous=simultaneous)

            fitter = lsqfit.MultiFitter(models=models)
            fit = fitter.lsqfit(data=y_data, prior=prior, fast=False, mopt=False)
            self._fit_interpolation = fit

        return self._fit_interpolation


    def _empbayes_groupings(self):
        zkeys = {}

        if self.empbayes_grouping == 'all':
            for param in self.prior:
                if param != 'wm0': # Have a good prior for this already
                    zkeys[param] = [param]

        elif self.empbayes_grouping == 'order':
            zkeys['chiral_nlo'] = ['A_l', 'A_s']
            zkeys['chiral_n2lo'] = ['A_ll', 'A_ss', 'A_ls', 'A_ll_g']
            zkeys['chiral_n3lo'] = ['A_lll', 'A_lls', 'A_lss', 'A_sss', 'A_lll_g', 'A_lls_g', 'A_lll_gg']
            zkeys['latt_nlo'] = ['A_a', 'A_alpha']
            zkeys['latt_n2lo'] = ['A_aa', 'A_al', 'A_as']
            zkeys['latt_n3lo'] = ['A_aaa', 'A_aal', 'A_aas', 'A_all', 'A_als', 'A_ass']

        elif self.empbayes_grouping == 'disc':
            zkeys['chiral'] = ['A_l', 'A_s', 'A_ll', 'A_ss', 'A_ls', 'A_ll_g', 'A_lll', 'A_lls', 'A_lss', 'A_sss', 'A_lll_g', 'A_lls_g', 'A_lll_gg']
            zkeys['disc'] = ['A_a', 'A_alpha', 'A_aa', 'A_al', 'A_as', 'A_aaa', 'A_aal', 'A_aas', 'A_all', 'A_als', 'A_ass']


        elif self.empbayes_grouping == 'alphas':
            zkeys['alphas'] = ['A_alpha']

        elif self.empbayes_grouping == 'alphas_eps2a':
            zkeys['alphas'] = ['A_alpha', 'A_a']

        all_keys = np.array([k for g in zkeys for k in zkeys[g]])
        prior_keys = list(self._make_prior())
        ignored_keys = set(all_keys) - set(prior_keys)

        # Don't determine empirical priors in param not in model
        for group in zkeys:
            for key in ignored_keys:
                if key in ignored_keys and key in zkeys[group]:
                    zkeys[group].remove(key)

        # Remove empty groupings
        for group in list(zkeys):
            if len(zkeys[group]) == 0:
                del(zkeys[group])
                

        return zkeys


    def _make_empbayes_fit(self, empbayes_grouping='order'):
        if (self._empbayes_fit is None) or (empbayes_grouping != self.empbayes_grouping):
            self.empbayes_grouping = empbayes_grouping
            self._counter = {'iters' : 0, 'evals' : 0}

            z0 = gv.BufferDict()
            for group in self._empbayes_groupings():
                z0[group] = 1.0


            # Might need to change minargs default values for empbayes_fit to converge:
            # tol=1e-8, svdcut=1e-12, debug=False, maxit=1000, add_svdnoise=False, add_priornoise=False
            # Note: maxit != maxfev. See https://github.com/scipy/scipy/issues/3334
            # For Nelder-Mead algorithm, maxfev < maxit < 3 maxfev?

            # For debugging. Same as 'callback':
            # https://github.com/scipy/scipy/blob/c0dc7fccc53d8a8569cde5d55673fca284bca191/scipy/optimize/optimize.py#L651
            def analyzer(arg):
                self._counter['evals'] += 1
                print("\nEvals: ", self._counter['evals'], arg,"\n")
                print(type(arg[0]))
                return None

            fit, z = lsqfit.empbayes_fit(z0, fitargs=self._make_fitargs, maxit=200, analyzer=None)
            print(z)
            self._empbayes_fit = fit

        return self._empbayes_fit


    def _make_fitargs(self, z):
        y_data = self.y
        prior = self._make_prior()

        # Ideally:
            # Don't bother with more than the hundredth place
            # Don't let z=0 (=> null GBF)
            # Don't bother with negative values (meaningless)
        # But for some reason, these restrictions (other than the last) cause empbayes_fit not to converge
        multiplicity = {}
        for key in z:
            multiplicity[key] = 0
            z[key] = np.abs(z[key])


        # Helps with convergence (minimizer doesn't use extra digits -- bug in lsqfit?)
        sig_fig = lambda x : np.around(x, int(np.floor(-np.log10(x))+3)) # Round to 3 sig figs
        capped = lambda x, x_min, x_max : np.max([np.min([x, x_max]), x_min])

        zkeys = self._empbayes_groupings()
        zmin = 1e-2
        zmax = 1e3
        for group in z.keys():
            for param in prior.keys():
                if param in zkeys[group]:
                    z[group] = sig_fig(capped(z[group], zmin, zmax))
                    prior[param] = gv.gvar(0, 1) *z[group]


        self._counter['iters'] += 1
        fitfcn = self._make_models()[-1].fitfcn
        print(self._counter['iters'], ' ', z)#{key : np.round(1. / z[key], 8) for key in z.keys()})

        # Penalize models outside logGBF
        def plausibility(s):
            plaus = 0
            for key in s:
                k = 1 / np.log(z_max[key]/z_min[key])
                plaus -= np.log(k/s[key]) *multiplicity[key]

            return plaus

        plaus = 0# plausibility(z)
        #print(plaus)

        return (dict(data=y_data, fcn=fitfcn, prior=prior), plaus)


    def _make_models(self, model_info=None, interpolation=False, w_a_data=None, simultaneous=False):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])
        if interpolation:

            model_info_interpolation = {
                'name' : model_info['name'] + '_interpolation',
                'chiral_cutoff': 'Fpi',
                'order': 'n2lo',
                'latt_ct': 'n2lo',
                'include_log': False,
                'include_log2': False,
                'include_fv': False,
                'include_alphas': False,
                'exclude': []
            }

            datatag = model_info_interpolation['name']
            models = np.append(models, model_interpolation(datatag=datatag, model_info=model_info_interpolation, w_a_data=w_a_data))
            if not simultaneous:
                return models

        datatag = model_info['name']
        models = np.append(models, model(datatag=datatag, model_info=model_info))

        return models


    def _make_prior(self, fit_data=None, interpolation=False, simultaneous=False):
        if fit_data is None:
            fit_data = self.fit_data

        newprior = gv.BufferDict()

        if interpolation or simultaneous:
            prior = self.prior_interpolation
            relative_error = lambda c, v: np.abs((c - v)/(c))
            if np.any([relative_error(0.3453, val) < 0.1 for val in gv.mean(fit_data['a/w'])]):
                newprior['w0a06'] = prior['w0a06']
            if np.any([relative_error(0.5257, val) < 0.1 for val in gv.mean(fit_data['a/w'])]):
                newprior['w0a09'] = prior['w0a09']
            if np.any([relative_error(0.7151, val) < 0.1 for val in gv.mean(fit_data['a/w'])]):
                newprior['w0a12'] = prior['w0a12']
            if np.any([relative_error(0.8894, val) < 0.1 for val in gv.mean(fit_data['a/w'])]):
                newprior['w0a15'] = prior['w0a15']
            
            for key in set(list(prior)).difference(['w0a06', 'w0a09', 'w0a12', 'w0a15']):
                newprior[key] = prior[key]

            for key in ['mpi', 'mk', 'lam_chi']:
                newprior[key] = fit_data[key]

            if not simultaneous:
                return newprior


        prior = self.prior

        # xpt terms
        # lo
        newprior['wm0'] = prior['wm0']

        # nlo
        if self.model_info['order'] in ['nlo', 'n2lo', 'n3lo']:
            newprior['A_l'] = prior['A_l']
            newprior['A_s'] = prior['A_s']
            newprior['A_a'] = prior['A_a']

        # n2lo
        if self.model_info['order'] in ['n2lo', 'n3lo']:
            newprior['A_aa'] = prior['A_aa']
            newprior['A_al'] = prior['A_al']
            newprior['A_as'] = prior['A_as']
            newprior['A_ll'] = prior['A_ll']
            newprior['A_ls'] = prior['A_ls']
            newprior['A_ss'] = prior['A_ss']

            if self.model_info['include_log']:
                newprior['A_ll_g'] = prior['A_ll_g']

        # n3lo
        if self.model_info['order'] in ['n3lo']:
            newprior['A_aaa'] = prior['A_aaa']
            newprior['A_aal'] = prior['A_aal']
            newprior['A_aas'] = prior['A_aas']
            newprior['A_all'] = prior['A_all']
            newprior['A_als'] = prior['A_als']
            newprior['A_ass'] = prior['A_ass']

            newprior['A_lll'] = prior['A_lll']
            newprior['A_lls'] = prior['A_lls']
            newprior['A_lss'] = prior['A_lss']

            newprior['A_sss'] = prior['A_sss']

            if self.model_info['include_log']:
                newprior['A_lll_g'] = prior['A_lll_g']
                newprior['A_lls_g'] = prior['A_lls_g']
            if self.model_info['include_log2']:
                newprior['A_lll_gg'] = prior['A_lll_gg']

        # latt terms
        if self.model_info['latt_ct'] in ['nlo', 'n2lo', 'n3lo']:
            newprior['A_a'] = prior['A_a']
        if self.model_info['latt_ct'] in ['n2lo', 'n3lo']:
            newprior['A_aa'] = prior['A_aa']
        if self.model_info['latt_ct'] in ['n3lo']:
            newprior['A_aaa'] = prior['A_aaa']

        # alpha_s corrections
        if self.model_info['include_alphas']:
            newprior['A_alpha'] = prior['A_alpha']

        # Move fit_data into prior
        for key in ['mpi', 'mk', 'lam_chi', 'a/w', 'L', 'alpha_s']:
            if key in fit_data:
                newprior[key] = fit_data[key]


        for key in self.model_info['exclude']:
            if key in newprior.keys():
                del(newprior[key])

        return newprior


class model(lsqfit.MultiFitterModel):

    def __init__(self, datatag, model_info, **kwargs):
        super(model, self).__init__(datatag)

        # Model info
        self.debug = False
        self.model_info = model_info


    def fitfcn(self, p, fit_data=None, xi=None, debug=None):
        if debug:
            self.debug = debug
            self.debug_table = {}

        if fit_data is not None:
            for key in fit_data.keys():
                p[key] = fit_data[key]

        for key in self.model_info['exclude']:
            p[key] = 0

        # Variables
        if xi is None:
            xi = {}
        if 'l' not in xi:
            xi['l'] = (p['mpi'] / p['lam_chi'])**2
        if 's' not in xi:
            xi['s'] = (2 *p['mk']**2 - p['mpi']**2) / p['lam_chi']**2
        if 'a' not in xi:
            xi['a'] = p['a/w']**2 / 4


        # lo
        output = p['wm0']

        if self.debug:
            self.debug_table['lo_ct'] = output

        # nlo
        if self.model_info['order'] in ['nlo', 'n2lo', 'n3lo']:
            output += self.fitfcn_nlo_ct(p, xi)
            if self.model_info['include_alphas']:
                output += self.fitfcn_nlo_latt_alphas(p, xi)
                
        elif self.model_info['latt_ct'] in ['nlo', 'n2lo', 'n3lo']: 
            output += self.fitfcn_nlo_latt_ct(p, xi)
            if self.model_info['include_alphas']:
                output += self.fitfcn_nlo_latt_alphas(p, xi)

        # n2lo 
        if self.model_info['order'] in ['n2lo', 'n3lo']:
            output += self.fitfcn_n2lo_ct(p, xi)
            if self.model_info['include_log']:
                output += self.fitfcn_n2lo_log(p, xi)

        elif self.model_info['latt_ct'] in ['n2lo', 'n3lo']:
            output += self.fitfcn_n2lo_latt_ct(p, xi)

        # n3lo
        if self.model_info['order'] in ['n3lo']:
            output += self.fitfcn_n3lo_ct(p, xi)
            if self.model_info['include_log']:
                output += self.fitfcn_n3lo_log(p, xi)
            if self.model_info['include_log2']:
                output += self.fitfcn_n3lo_log_sq(p, xi)
                
        elif self.model_info['latt_ct'] in ['n3lo']:
            output += self.fitfcn_n3lo_latt_ct(p, xi)



        for key in self.model_info['exclude']:
            del(p[key])

        if debug:
            #print(gv.tabulate(self.debug_table))
            temp_string = ''
            for key in self.debug_table:
                temp_string +='  % .15f:  %s\n' %(gv.mean(self.debug_table[key]), key)
            temp_string +='   -----\n'
            temp_string +='  % .15f:  %s\n' %(gv.mean(output), 'total')
            print(temp_string)

        return output


    def fitfcn_nlo_ct(self, p, xi):
        output = p['A_l'] *xi['l'] + p['A_s'] *xi['s'] + p['A_a'] *xi['a']

        if self.debug:
            self.debug_table['nlo_ct'] = output

        return output


    def fitfcn_n2lo_ct(self, p, xi):     
        output = ( 
            + p['A_aa'] *xi['a'] *xi['a']
            + p['A_al'] *xi['a'] *xi['l']
            + p['A_as'] *xi['a'] *xi['s']
            + p['A_ll'] *xi['l'] *xi['l']
            + p['A_ls'] *xi['l'] *xi['s']
            + p['A_ss'] *xi['s'] *xi['s']
        )

        if self.debug:
            self.debug_table['n2lo_ct'] = output

        return output


    def fitfcn_n2lo_log(self, p, xi):
        if self.model_info['include_fv']:
            output = p['A_ll_g'] *xi['l']**2 *sf.fcn_I_m(xi['l'], p['L'], p['lam_chi'], 10)
        else:
            output = p['A_ll_g'] *xi['l']**2 *np.log(xi['l'])

        if self.debug:
            self.debug_table['n2lo_log'] = output

        return output


    def fitfcn_n3lo_ct(self, p, xi):

        output = (
            + p['A_aaa'] *xi['a'] *xi['a'] *xi['a']
            + p['A_aal'] *xi['a'] *xi['a'] *xi['l']
            + p['A_aas'] *xi['a'] *xi['a'] *xi['s']
            + p['A_all'] *xi['a'] *xi['l'] *xi['l']
            + p['A_als'] *xi['a'] *xi['l'] *xi['s']
            + p['A_ass'] *xi['a'] *xi['s'] *xi['s']

            + p['A_lll'] *xi['l'] *xi['l'] *xi['l']
            + p['A_lls'] *xi['l'] *xi['l'] *xi['s']
            + p['A_lss'] *xi['l'] *xi['s'] *xi['s']

            + p['A_sss'] *xi['s'] *xi['s'] *xi['s']
        )

        if self.debug:
            self.debug_table['n3lo_ct'] = output

        return output


    def fitfcn_n3lo_log(self, p, xi):
        if self.model_info['include_fv']:
            output = (
                + p['A_lll_g'] *xi['l']**3 *sf.fcn_I_m(xi['l'], p['L'], p['lam_chi'], 10)
                + p['A_lls_g'] *xi['l']**2 *xi['s'] *sf.fcn_I_m(xi['l'], p['L'], p['lam_chi'], 10)
            )
        else:
            output = (
                + p['A_lll_g'] *xi['l']**3 *np.log(xi['l'])
                + p['A_lls_g'] *xi['l']**2 *xi['s'] *np.log(xi['l'])
            )

        if self.debug:
            self.debug_table['n3lo_log'] = output

        return output


    def fitfcn_n3lo_log_sq(self, p, xi):
        if self.model_info['include_fv']:
            output = p['A_lll_gg'] *xi['l']**3 *(
                + (sf.fcn_I_m(xi['l'], p['L'], p['lam_chi'], 10))**2
                #- (np.log(xi['l']))**2
            )
        else:
            output = p['A_lll_gg'] *xi['l']**3 *(np.log(xi['l']))**2

        if self.debug:
            self.debug_table['n3lo_log_sq'] = output

        return output


    def fitfcn_nlo_latt_alphas(self, p, xi):
        output = p['A_alpha'] *xi['a'] *p['alpha_s']

        if self.debug:
            self.debug_table['nlo_alphas'] = output

        return output


    def fitfcn_nlo_latt_ct(self, p, xi):
        output = p['A_a'] *xi['a']

        if self.debug:
            self.debug_table['nlo_latt'] = output

        return output


    def fitfcn_n2lo_latt_ct(self, p, xi):
        output = p['A_aa'] *xi['a']**2

        if self.debug:
            self.debug_table['n2lo_latt'] = output

        return output


    def fitfcn_n3lo_latt_ct(self, p, xi):
        output = p['A_aaa'] *xi['a']**3

        if self.debug:
            self.debug_table['n3lo_latt'] = output

        return output


    def buildprior(self, prior, mopt=None, extend=False):
        return prior


    def builddata(self, data):
        return data[self.datatag]


class model_interpolation(lsqfit.MultiFitterModel):

    def __init__(self, datatag, model_info, w_a_data, **kwargs):
        super(model_interpolation, self).__init__(datatag)

        # Model info
        self.model_info = model_info
        self.w_a_data = w_a_data


    def fitfcn(self, p, fit_data=None, xi=None, latt_spacing=None):
        if fit_data is not None:
            for key in fit_data.keys():
                p[key] = fit_data[key]

        for key in self.model_info['exclude']:
            p[key] = 0


        # Variables
        if xi is None:
            xi = {}
        if 'l' not in xi:
            xi['l'] = (p['mpi'] / p['lam_chi'])**2
        if 's' not in xi:
            xi['s'] = (2 *p['mk']**2 - p['mpi']**2) / p['lam_chi']**2

        w0ch_a = self.fitfcn_lo_ct(p, xi, latt_spacing)

        if 'a' not in xi:
            xi['a'] =  1 / (2 *w0ch_a)**2

        # lo
        #output = w0ch_a

        # nlo
        #output += w0ch_a *self.fitfcn_nlo_ct(p, xi)

        # n2lo
        #output += w0ch_a *self.fitfcn_n2lo_ct(p, xi)
        #output += w0ch_a *self.fitfcn_n2lo_log(p, xi)

        #return output

        #print(np.sqrt(xi['a'] ))

        #if isinstance(w0ch_a[0], gv._gvarcore.GVar):
        #    print(gv.evalcorr([w0ch_a[0], self.fitfcn_lo_ct(p, latt_spacing)[0]]))

        #output = 1 / (2 *np.sqrt(xi['a'])) *(
        output = w0ch_a *(
            + 1
            + self.fitfcn_nlo_ct(p, xi)
            + self.fitfcn_n2lo_ct(p, xi)
            + self.fitfcn_n2lo_log(p, xi)
        )

        #print('good')
        return output


    def fitfcn_lo_ct(self, p, xi, latt_spacing=None):

        if latt_spacing == 'a06':
            output = p['w0a06']
        elif latt_spacing == 'a09':
            output= p['w0a09']
        elif latt_spacing == 'a12':
            output = p['w0a12']
        elif latt_spacing == 'a15':
            output= p['w0a15']

        else:
            relative_error = lambda c, v: np.abs((c - v.mean)/(c))
            output = xi['l'] *xi['s'] *0 # returns correct shape
            for j, a_w0 in enumerate(1 / self.w_a_data):
                if (relative_error(0.3453, a_w0) < 0.1):
                    output[j] = p['w0a06']
                elif (relative_error(0.5257, a_w0) < 0.1):
                    output[j] = p['w0a09']
                elif (relative_error(0.7151, a_w0) < 0.1):
                    output[j] = p['w0a12']
                elif (relative_error(0.8894, a_w0) < 0.1):
                    output[j] = p['w0a15']
                else:
                    output[j] = 0

        return output


    def fitfcn_nlo_ct(self, p, xi):
        output = (
            + p['k_l'] *xi['l'] 
            + p['k_s'] *xi['s'] 
            + p['k_a'] *xi['a']
        )
        return output


    def fitfcn_n2lo_ct(self, p, xi):     
        output = ( 
            + p['k_aa'] *xi['a'] *xi['a']
            + p['k_al'] *xi['a'] *xi['l']
            + p['k_as'] *xi['a'] *xi['s']
            + p['k_ll'] *xi['l'] *xi['l']
            + p['k_ls'] *xi['l'] *xi['s']
            + p['k_ss'] *xi['s'] *xi['s']
        )
        return output


    def fitfcn_n2lo_log(self, p, xi):
        output = p['k_ll_g'] *xi['l']**2 *np.log(xi['l'])
        return output


    def buildprior(self, prior, mopt=None, extend=False):
        return prior


    def builddata(self, data):
        return data[self.datatag]