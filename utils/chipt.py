#!/usr/bin/env python3
import sys
import numpy as np
import functools # for LRU cache
lru_cache_size=400 #increase this if need be to store all bessel function evaluations in cache memory
import warnings # supress divide by zero warning in fakeData determination of terms
import scipy.special as spsp # Bessel functions
import gvar as gv

pi = np.pi

#@functools.lru_cache(maxsize=lru_cache_size)

'''
lazily evaluated dictionary that will compute convenience observables
as they are needed by calling into the parent model's convenience
functions (with a prepended underscore)
'''
class ConvenienceDict(dict):

    def __init__(self, parent_model, x, p, *args, **kwargs):
        self.p_model = parent_model
        self.x = x
        self.p = p
        dict.__init__(self,*args, **kwargs)

    def __getitem__(self, key):
        if key not in self.keys():
            dict.__setitem__(self, key, getattr(FitModel, "_"+key)(self.p_model,self.x,self.p,self))
        return dict.__getitem__(self,key)


'''
This class defines the functions that go into the various fit models
'''
class FitModel:

    def __init__(self, _term_list, _fv, _FF):
        self.term_list       = _term_list
        self.fv              = _fv
        self.FF              = _FF
        self.required_params = self._get_used_params()

    def __call__(self, x, p):
        if len(self.term_list)==0: return 0. # convenience value for plotting purposes
        convenience_p = ConvenienceDict(self, x, p)
        return sum(getattr(FitModel, term)(self, x, p, convenience_p) for term in self.term_list)

    def get_required_parameters(self):
        return self.required_params[1]

    ''' this function self-reflects to find out
        which x, p and convenience_p (cp) are going to be required '''
    def _get_used_params(self):
        class FakeDict(dict):
            def __init__(self):
                self.param_list = set()
            def __getitem__(self, key):
                self.param_list.add(key)
                return 1.
        fake_x  = FakeDict()
        fake_p  = FakeDict()
        fake_cp = ConvenienceDict(self, fake_x, fake_p)
        # the regular function calls (which automatically recurse into the
        # convenience functions)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dummy = sum(getattr(FitModel, term)(self,fake_x,fake_p,fake_cp)
                            for term in self.term_list)
            except Exception as e:
                pass
        return fake_x.param_list, fake_p.param_list, set(fake_cp.keys())

    ''' define all the convenience functions; whichever attribute is accessed of
        `cP` in the physics functions above needs to have a corresponding
        function defined here with the name and an underscore prepended '''
    # xpt masses
    def _p2(self, x, p, cP):
        if self.FF == 'Of':
            return (p['mpi'] / x['Lam_'+self.FF])**2
        else:
            return (p['mpi'] / p['Lam_'+self.FF])**2
    def _s2(self, x, p, cP):
        if self.FF == 'Of':
            return (2*p['mk']**2 - p['mpi']**2) / (x['Lam_'+self.FF])**2
        else:
            return (2*p['mk']**2 - p['mpi']**2) / (p['Lam_'+self.FF])**2
    # logs
    def _lp(self, x, p, cP): return np.log(cP['p2'])
    # eps_a**2
    def _a2(self, x, p, cP):  return (p['aw0'] / 2)**2

    # Finite Volume Corrections to Tadpole Integral
    @functools.lru_cache(maxsize=lru_cache_size)
    def k1(self, mL):
        cn = np.array([6,12,8,6,24,24,0,12,30,24,24,8,24,48,0,6,48,36,24,24])
        n_mag = np.sqrt(np.arange(1,len(cn)+1,1))
        k1_r = np.sum(cn * spsp.kn(1,n_mag * mL) / mL / n_mag)
        return k1_r

    # Tadpole Integrals
    def _I(self, eps_sq, mL):
        return eps_sq*np.log(eps_sq) + (4.*eps_sq*self.k1(mL) if self.fv else 0.)
    def _Ip(self, x, p, cP): return self._I(cP['p2'], (x['mpiL'] if self.fv else None))
    # mock Taylor Expansion FV correction
    def _IT(self, mL):
        return (4*self.k1(mL) if self.fv else 0.)
    def _ITp(self, x, p, cP): return p['t_fv']*self._IT( x['mpiL'] if self.fv else None)

    ''' Define all the fit functions to be used in the analysis.  We describe them
        in pieces, which are assembled based upon the term_list, to form a given
        fit function.
    '''
    # Fit functions
    def xpt_nlo(self,x,p,cP):
        #print('DEBUG p:',p)
        a_result  = p['c0']
        a_result +=   p['c_l'] * cP['p2']\
                    + p['c_s'] * cP['s2']\
                    + p['d_a'] * cP['a2']
        #print('DEBUG: p2',cP['p2'], 's2', cP['s2'])
        return a_result

    def taylor_nlo(self,x,p,cP):
        return self.xpt_nlo(x,p,cP)

    def nlo_alphaS(self,x,p,cP):
        return p['d_a_aS'] * x['alphaS'] * cP['a2']

    def n2lo_ct(self,x,p,cP):
        a_result  =   p['c_ll'] * cP['p2']**2\
                    + p['c_ls'] * cP['p2'] * cP['s2']\
                    + p['c_ss'] * cP['s2']**2
        a_result += cP['a2'] *( p['d_aa'] * cP['a2'] + p['d_al'] * cP['p2'] + p['d_as'] * cP['s2'])
        return a_result

    def n2lo_log(self,x,p,cP):
        # note - Ip = p2 * log(p2)
        return p['c_lln'] * cP['p2'] * cP['Ip']

    def n2lo_ct_fv(self,x,p,cP):
        return cP['p2']**2 * cP['ITp'] #NOTE, ITp contains t_fv

    def n3lo_ct(self,x,p,cP):
        a_result  =   p['c_lll'] * cP['p2']**3\
                    + p['c_lls'] * cP['p2']**2 * cP['s2']\
                    + p['c_lss'] * cP['p2']    * cP['s2']**2\
                    + p['c_sss'] * cP['s2']**3\
                    + p['d_aaa'] * cP['a2']**3\
                    + p['d_aal'] * cP['a2']**2 * cP['p2']\
                    + p['d_aas'] * cP['a2']**2 * cP['s2']\
                    + p['d_all'] * cP['a2']    * cP['p2']**2\
                    + p['d_als'] * cP['a2']    * cP['p2']*cP['s2']\
                    + p['d_ass'] * cP['a2']    * cP['s2']**2
        return a_result

    def n3lo_log(self,x,p,cP):
        # note - Ip = p2 * log(p2)
        #        p2 * Ip**2 = p2**3 * log(p2)**2
        a_result  =   p['c_llln']  * cP['p2']**2 * cP['Ip'] \
                    + p['c_llln2'] * cP['p2'] * cP['Ip']**2 \
                    + p['c_lsln']  * cP['p2'] * cP['s2'] * cP['Ip']
        return a_result

    ''' w0/a extrapolation functions
    '''
    def w0_lo(self, x,p,cP):
        return p['w0_0']

    def w0_nlo(self, x,p,cP):
        return p['w0_0'] * (p['k_l'] * cP['p2'] + p['k_s'] * cP['s2'])

    def w0_nlo_a0(self, x, p, cP):
        return p['w0_0'] * p['k_a'] / (2 * p['w0_0'])**2

    def w0_nlo_a(self, x, p, cP):
        return p['w0_0'] * cP['a2']

    def w0_n2lo(self, x, p, cP):
        a_result  = p['w0_0'] * p['k_ll']  * cP['p2']**2
        a_result += p['w0_0'] * p['k_ls']  * cP['p2']*cP['s2']
        a_result += p['w0_0'] * p['k_ss']  * cP['s2']**2
        a_result += p['w0_0'] * p['k_lln'] * cP['p2']**2 * cP['Ip']

        return a_result

    def w0_n2lo_a0(self, x, p, cP):
        a_result  = p['w0_0'] * p['k_aa'] / (2*p['w0_0'])**4
        a_result += p['w0_0'] * p['k_la'] * cP['p2'] / (2*p['w0_0'])**2
        a_result += p['w0_0'] * p['k_sa'] * cP['s2'] / (2*p['w0_0'])**2

        return a_result

    def w0_n2lo_a(self, x, p, cP):
        a_result  = p['w0_0'] * p['k_aa'] * cP['a2']**2
        a_result += p['w0_0'] * p['k_la'] * cP['p2'] * cP['a2']
        a_result += p['w0_0'] * p['k_sa'] * cP['s2'] * cP['a2']

        return a_result