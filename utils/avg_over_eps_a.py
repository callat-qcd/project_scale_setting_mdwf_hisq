import os, sys
import gvar as gv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
import analysis
import input_params as ip
# Load input params
switches   = ip.switches
phys_point = ip.phys_point

switches['ansatz']['models'] = [
        'xpt_n2lo', 'xpt_n3lo',
        'taylor_n2lo', 'taylor_n3lo'
    ]

switches['sys'] = dict()     # these cause the fitter to loop over various options
switches['sys']['Lam_chi']   = True # FF = F, O, [Of = O with fixed m_Omega value as x-par]
switches['scales']           = ['F','O'] # Of can also be added
switches['sys']['alphaS']    = True # include alphaS at NLO?
switches['sys']['FV']        = True # turn on/off FV corrections

models = analysis.sys_models(switches)

for gf in ['t0', 't0_imp', 'w0', 'w0_imp']:
    switches['gf_scale'] = gf
    fit_results = dict()
    for eps_a in ['fixed_eps_a', 'variable_eps_a']:
        for model in models:
            p_fit = 'pickled_fits/'+model+'_'+switches['gf_scale']+'_'+eps_a
            p_fit += '_nlo_x_'+str(ip.nlo_x)+'_nlo_a_'+str(ip.nlo_a)
            p_fit += '_n2lo_x_'+str(ip.n2lo_x)+'_n2lo_a_'+str(ip.n2lo_a)
            p_fit += '_n3lo_x_'+str(ip.n3lo_x)+'_n3lo_a_'+str(ip.n3lo_a)+'.p'
            fit_results[switches['gf_scale']+'_'+eps_a+'_'+model] = gv.load(p_fit)
    model_avg = analysis.BayesModelAvg(fit_results)
    #model_avg.print_weighted_models()
    to_fm = phys_point['p']['hbar_c'] / phys_point['p']['m_omega']
    w0_mO_model_avg, model_sig = model_avg.bayes_model_avg(to_fm)
