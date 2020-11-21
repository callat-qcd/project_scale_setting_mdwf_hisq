import gvar as gv
import numpy as np

switches = dict()
# Make two sets of ensembles - so we can plot data excluded from fit
switches['ensembles'] = [
    'a15m400'  ,'a12m400' ,'a09m400',
    'a15m350'  ,'a12m350' ,'a09m350',
    'a15m310'  ,'a12m310' ,'a09m310','a06m310L','a15m310L',
    'a15m220'  ,'a12m220' ,'a09m220',
    'a12m220S', 'a12m220L','a12m220ms',
    'a12m180L',
    'a15m135XL','a12m130' ,'a09m135',]
switches['ensembles_fit'] = [
    'a15m400'  ,'a12m400' ,'a09m400',
    'a15m350'  ,'a12m350' ,'a09m350',
    'a15m310'  ,'a12m310' ,'a09m310','a06m310L','a15m310L',
    'a15m220'  ,'a12m220' ,'a09m220',
    'a12m220S', 'a12m220L','a12m220ms',
    'a12m180L',
    'a15m135XL','a12m130' ,'a09m135',]

# FIT MODELS
switches['ansatz'] = dict()
switches['ansatz']['models'] = [
        'xpt_nnnlo_alphaS_FV', 'taylor_nnnlo_alphaS_FV'
        #'xpt_nlo','xpt_nnlo', 'xpt_nnnlo',
        #'taylor_nlo', 'taylor_nnlo', 'taylor_nnnlo'
        #'xpt_nlo_FV','xpt_nnlo_FV', 'xpt_nnnlo_FV',
        #'taylor_nlo_FV', 'taylor_nnlo_FV', 'taylor_nnnlo_FV'
    ]
'''
    The full list of models can be rather long.  The sys switches help loop
    over them.  Example other base models are
        taylor_nnnlo_FV
        ma_nnnlo_FV
'''

switches['w0'] = 'callat' # or milc

# SYSTEMATIC SWITCHES
switches['sys'] = dict()     # these cause the fitter to loop over various options
switches['sys']['Lam_chi']   = False # FF = F, O
switches['sys']['alphaS']    = False # include alphaS at NLO?
switches['scales']           = ['F','O']
# OLDER SYSTEMATICS - still work, but not used
switches['sys']['FV']        = True # turn on/off FV corrections
                               # scale is used when the loop over scales is not triggered
switches['scale']            = 'F' # F: Lam = 4pi Fpi; O: Lam = m_O

switches['print_lattice']    = False # print data for paper - no fitting will occur

# Fitting options
switches['model_avg']         = True # perform Bayes Model Avg
switches['print_fit']         = False # print lsqfit results?
switches['report_phys']       = False  # report physical point for each fit?
switches['bs_bias']           = True  # shift bs avg to b0?
switches['save_fits']         = False  # save fits in pickle file?
switches['prior_search']      = False # perform a crude grid search to optimize
switches['prior_verbose']     = False # NNLO and NNNLO prior widths
switches['scipy']             = True # use scipy minimizer instead of gsl?
# w0 interpolation fit options
switches['w0_interpolate']    = True
switches['w0_a_model']        = 'w0_nnlo_a0_FV_all' # w0_nnlo_a0_FV_all, w0_nnlo_FV_all
switches['print_w0_interp']   = False

switches['check_fit']         = False # print pieces of fit function - no fitting will occur
# check reweighting and stochastic uncertainty improvement
switches['reweight']          = False
switches['deflate_a06']       = False
switches['deflate_a09']       = False
switches['deflate_a12m220ms'] = False

# Plotting options
switches['save_figs']        = True  # save figures
switches['make_extrap']      = True # make plots
switches['make_interp']      = False
switches['make_hist']        = False # make plots
switches['make_fv']          = False
switches['plot_ls']          = True # make parameter space plots

# DEBUGGING
switches['debug_models']     = False # print list of models being generated
switches['debug_save_fit']   = False # check pickling of fit works
switches['debug_phys_point'] = False # run report_phys_point even if fit is just loaded
switches['debug_shift']      = False # check the shifting of raw data to extrapolated points
switches['debug_bs']         = False # debug shape of bs lists

# Taylor priors - beyond NLO - use "XPT" NiLO priors
priors = dict()
priors['c0']   = gv.gvar(1.5,1)

def make_priors(priors, s):
    nlo_x = 1
    nlo_a = 1
    priors['c_l']    = gv.gvar(1   ,nlo_x * s)
    priors['c_s']    = gv.gvar(1   ,nlo_x * s)
    priors['d_a']    = gv.gvar(-0.5,nlo_a)
    priors['d_a_aS'] = gv.gvar(0 ,nlo_a)

    nnlo_x = 1
    nnlo_a = 1
    priors['c_ll']  = gv.gvar(0., nnlo_x * s**2)
    priors['c_ls']  = gv.gvar(0., nnlo_x * s**2)
    priors['c_ss']  = gv.gvar(0., nnlo_x * s**2)
    priors['c_lln'] = gv.gvar(0., nnlo_x * s**2)
    priors['d_aa']  = gv.gvar(0., nnlo_a)
    priors['d_al']  = gv.gvar(0., nnlo_a * s)
    priors['d_as']  = gv.gvar(0., nnlo_a * s)
    priors['t_fv']  = gv.gvar(0,nnlo_a * s)

    nnnlo_x = 1
    nnnlo_a = 1
    priors['c_lll']   = gv.gvar(0., nnnlo_x * s**3)
    priors['c_lls']   = gv.gvar(0., nnnlo_x * s**3)
    priors['c_lss']   = gv.gvar(0., nnnlo_x * s**3)
    priors['c_sss']   = gv.gvar(0., nnnlo_x * s**3)
    priors['c_llln2'] = gv.gvar(0., nnnlo_x * s**3)
    priors['c_llln']  = gv.gvar(0., nnnlo_x * s**3)
    priors['d_aaa']   = gv.gvar(0., nnnlo_a)
    priors['d_aal']   = gv.gvar(0., nnnlo_a * s)
    priors['d_aas']   = gv.gvar(0., nnnlo_a * s)
    priors['d_all']   = gv.gvar(0., nnnlo_a * s**2)
    priors['d_als']   = gv.gvar(0., nnnlo_a * s**2)
    priors['d_ass']   = gv.gvar(0., nnnlo_a * s**2)

    return priors

priors[('a15','w0_0')] = gv.gvar(1.0,1)
priors[('a12','w0_0')] = gv.gvar(1.5,1)
priors[('a09','w0_0')] = gv.gvar(2.0,1)
priors[('a06','w0_0')] = gv.gvar(3.0,1)

priors['k_l'] = gv.gvar(0,2)
priors['k_s'] = gv.gvar(0,2)
priors['k_a'] = gv.gvar(2,2)

priors['k_ll']  = gv.gvar(0,2)
priors['k_lln'] = gv.gvar(0,2)
priors['k_ls']  = gv.gvar(0,2)
priors['k_ss']  = gv.gvar(0,2)

priors['k_aa'] = gv.gvar(0,2)
priors['k_la'] = gv.gvar(0,2)
priors['k_sa'] = gv.gvar(0,2)


''' Physical point extrapolation
'''
phys_point = dict()
# Physical point values taken from FLAG
# FLAG[2019] = 1902.08191
# FLAG[2017] = 1607.00299
FPi_phys = gv.gvar(130.2/np.sqrt(2), 0.8/np.sqrt(2)) # FLAG[2019] (84)
FK_phys  = gv.gvar(155.7/np.sqrt(2), 0.7/np.sqrt(2)) # FLAG[2019] (85) - use NF=2+1 for consistency
m_omega_phys = gv.gvar(1672.43,0.32) # PDG 2020

phys_point = {
    'p':{
        'Fpi'     : FPi_phys,
        'Lam_F'   : 4 * np.pi * FPi_phys,
        'Lam_O'   : m_omega_phys,
        'mpi'     : gv.gvar(134.8, 0.3), #FLAG 2017 (16)
        'mk'      : gv.gvar(494.2, 0.3), #FLAG 2017 (16) isospin symmetric
        'm_omega' : m_omega_phys,
        'hbar_c'  : 197.327,
        'mk+'     : gv.gvar(491.2, 0.5), #FLAG 2017 (15) strong isospin breaking only
        'mk0'     : gv.gvar(497.2, 0.4), #FLAG 2017 (15) strong isospin breaking only
        'aw0'     : gv.gvar(0,0),
        'a2DI'    : gv.gvar(0,0),
        'w0'      : gv.gvar(0.1714,0),
    },
    'x' : {'alphaS':0},
    'y' : {},
}

''' Values for checking fit function
'''
Fpi_check = 92.2
FK_check  = 110.
mpi_check = 135.0
mk_check  = 495.5
me_check  = np.sqrt(4./3*mk_check**2 - 1./3*mpi_check**2)
mO_check  = 1670
L_check   = 3.5 / mpi_check
check_fit = {
    'p':{
        'mpi'    : mpi_check,
        'mk'     : mk_check,
        'm_omega': mO_check,
        'Fpi'    : Fpi_check,
        'aw0'    : 0.6,
        'Lam_F'  : 4 * np.pi * Fpi_check,
        'Lam_O'  : mO_check,
        # chiral LECs
        'c0'     : 1.2,
        'c_l'    : 1.1,
        'c_s'    : 1.3,
        'c_ll'   : 0.5,
        't_fv'   : 0.19,
        'c_lln'  : 0.2,
        'c_ls'   : 0.6,
        'c_ss'   : 0.7,
        'c_lll'  : 0.2,
        'c_llln' : 0.4,
        'c_llln2': 0.9,
        'c_lls'  : 0.3,
        'c_lss'  : 0.45,
        'c_sss'  : 0.53,
        # discretization
        'd_a'    : -0.6,
        'd_a_aS' : -0.4,
        'd_aa'   : 0.11,
        'd_al'   : 0.22,
        'd_as'   : 0.33,
        'd_aaa'  : 0.44,
        'd_aal'  : 0.12,
        'd_aas'  : 0.13,
        'd_all'  : 0.14,
        'd_als'  : 0.16,
        'd_ass'  : 0.25,
    },
    'x':{'alphaS':0.2, 'meL':me_check*L_check,'mpiL':mpi_check*L_check, 'mkL':mk_check*L_check}
}
