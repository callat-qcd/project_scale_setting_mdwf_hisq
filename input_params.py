import gvar as gv
import numpy as np

switches = dict()
# Make two sets of ensembles - so we can plot data excluded from fit
switches['ensembles'] = [
    'a15m400'  ,'a12m400' ,'a09m400',
    'a15m350'  ,'a12m350' ,'a09m350',
    'a15m310'  ,'a12m310' ,'a09m310','a06m310L',#'a15m310L',
    'a15m220'  ,'a12m220' ,'a09m220',
    'a12m220S', 'a12m220L',#'a12m220ms',
    #'a12m180L',
    'a15m135XL','a12m130' ,'a09m135',]
switches['ensembles_fit'] = [
    'a15m400'  ,'a12m400' ,'a09m400',
    'a15m350'  ,'a12m350' ,'a09m350',
    'a15m310'  ,'a12m310' ,'a09m310','a06m310L',#'a15m310L',
    'a15m220'  ,'a12m220' ,'a09m220',
    'a12m220S', 'a12m220L',#'a12m220ms',
    #'a12m180L',
    'a15m135XL','a12m130' ,'a09m135',]

# FIT MODELS
switches['ansatz'] = dict()
switches['ansatz']['models'] = ['xpt_nlo','taylor_nlo']
'''
    The full list of models can be rather long.  The sys switches help loop
    over them.  Example other base models are
        taylor_nnnlo_FV
        ma_nnnlo_FV
'''

switches['w0'] = 'callat' # or milc

# SYSTEMATIC SWITCHES
switches['sys'] = dict()     # these cause the fitter to loop over various options
switches['sys']['Lam_chi']   = True # FF = F, O
switches['sys']['alphaS']    = True # include alphaS at NNLO?
# OLDER SYSTEMATICS - still work, but not used
switches['sys']['FV']        = True # turn on/off FV corrections
switches['scales']           = ['F','O']
                               # scale is used when the loop over scales is not triggered
switches['scale']            = 'F' # PP, PK, KK, LamChi = 4 * pi * sqrt(FA * FB)

switches['print_lattice']    = False # print data for paper - not fitting will occur

# Fitting options
switches['bs_bias']          = True  # shift bs avg to b0?
switches['print_fit']        = False # print lsqfit results?
switches['report_phys']      = False  # report physical point for each fit?
switches['save_fits']        = False  # save fits in pickle file?
switches['model_avg']        = True # perform Bayes Model Avg
switches['prior_search']     = False # perform a crude grid search to optimize
switches['prior_verbose']    = False # NNLO and NNNLO prior widths
switches['scipy']            = False # use scipy minimizer instead of gsl?

switches['check_fit']        = False # print pieces of fit function - no fitting will occur

# Plotting options
switches['save_figs']        = True  # save figures
switches['make_extrap']      = False # make plots
switches['make_hist']        = False # make plots
switches['make_fv']          = True
switches['milc_compare']     = False # compare with MILCs result
switches['plot_ls']          = False # report fitted Li values

# DEBUGGING
switches['debug_models']     = True # print list of models being generated
switches['debug_save_fit']   = False # check pickling of fit works
switches['debug_phys_point'] = False # run report_phys_point even if fit is just loaded
switches['debug_shift']      = False # check the shifting of raw data to extrapolated points
switches['debug_bs']         = False # debug shape of bs lists

# Taylor priors - beyond NLO - use "XPT" NiLO priors
priors = dict()
priors['c0']   = gv.gvar(1.,0.5)
priors['t_fv'] = gv.gvar(0,100)

priors['c_l'] = gv.gvar(0,10)
priors['c_s'] = gv.gvar(0,10)
priors['d_2'] = gv.gvar(0,10)

nlo_x = 10
nlo_a = 10
priors['c_ll']  = gv.gvar(0., nlo_x)
priors['c_ls']  = gv.gvar(0., nlo_x)
priors['c_ss']  = gv.gvar(0., nlo_x)
priors['c_lln'] = gv.gvar(0., nlo_x)
priors['d_4']   = gv.gvar(0., nlo_a)
priors['d_l4']  = gv.gvar(0., nlo_a)
priors['d_s4']  = gv.gvar(0., nlo_a)

n3lo_x = 5
#n3lo_a = 5
n3lo_a = n3lo_x
priors['kp_6']  = gv.gvar(0.0, n3lo_x) # (eps_K^2 - eps_pi^2 ) * eps_K^2 * eps_pi^2
priors['k_6']   = gv.gvar(0.0, n3lo_x) # (eps_K^2 - eps_pi^2 )^2 * eps_K^2
priors['p_6']   = gv.gvar(0.0, n3lo_x) # (eps_K^2 - eps_pi^2 )^2 * eps_pi^2
priors['s_6']   = gv.gvar(0.0, n3lo_a) # (eps_K^2 - eps_pi^2 ) * eps_a^4
priors['sk_6']  = gv.gvar(0.0, n3lo_a) # (eps_K^2 - eps_pi^2 ) * eps_K^2 * eps_a^2
priors['sp_6']  = gv.gvar(0.0, n3lo_a) # (eps_K^2 - eps_pi^2 ) * eps_pi^2 * eps_a^2

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
L_check   = 3.5 / mpi_check
check_fit = {
    'p':{
        'mpi'    : mpi_check,
        'mk'     : mk_check,
        'Fpi'    : Fpi_check,
        'FK'     : FK_check,
        'Lam_PP': 4 * np.pi * Fpi_check,
        'L1'     :  0.000372,
        'L2'     :  0.000493,
        'L3'     : -0.003070,
        'L4'     :  0.000089,
        'L5'     :  0.000377,
        'L6'     :  0.000011,
        'L7'     : -0.000340,
        'L8'     :  0.000294,
        'k_4'    : -3.0,
        'p_4'    :  4.0,
        # discretization
        'aw0'    : 0.8,
        's_4'    : 1.5,
        'saS_4'  : 2.5,
        'kp_6'   : 2.1,
        'k_6'    : 2.2,
        'p_6'    : 2.3,
        's_6'    : 2.4,
        'sk_6'   : 2.5,
        'sp_6'   : 2.6,
        # mixed action
        'mss'    : 520.,
        'mju'    : 200.,
        'mjs'    : 510.,
        'mru'    : 505.,
        'mrs'    : 525.,
        'a2DI'   : 400.**2,
    },
    'x':{'alphaS':0.2, 'meL':me_check*L_check}
}
for mphi in ['mpi','mk', 'mju', 'mjs', 'mru', 'mrs', 'mss']:
    if mphi in check_fit['p']:
        check_fit['x'][mphi+'L'] = check_fit['p'][mphi] * L_check
check_fit['x']['mxL'] = L_check * np.sqrt(4./3*mk_check**2 - 1./3*mpi_check**2 + check_fit['p']['a2DI'])
