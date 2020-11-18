#!/usr/bin/env python3
# python libraries
import os, sys, shutil, copy
import matplotlib.pyplot as plt
import numpy as np
# Peter Lepage's libraries
import gvar as gv
import lsqfit

# FK / Fpi libraries
sys.path.append('utils')
import io_utils
import chipt
import analysis
import plotting

''' Write description
'''

''' useful for changing matplotlib.plt.show behavior and accessing results
    from main() if run interactively in iPython
'''
def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def main():
    print("python     version:", sys.version)
    import input_params as ip
    # Load input params
    switches   = ip.switches
    priors     = ip.priors
    phys_point = ip.phys_point
    check_fit  = ip.check_fit

    # if check_fit: - add support
    if switches['check_fit']:
        models = analysis.sys_models(switches)
        print('DEBUGGING FIT FUNCTION')
        print('p')
        for k in check_fit['p']:
            print("%7s" %k,check_fit['p'][k])
        print('x')
        for k in check_fit['x']:
            print("%7s" %k,check_fit['x'][k])
        for model in models:
            print('===============================================================')
            print('DEBUGGING Terms in ',model)
            print('---------------------------------------------------------------')
            model_list, FF, fv = analysis.gather_model_elements(model)
            debug_fit_function(check_fit, model_list, FF, fv)
        sys.exit()

    if switches['save_fits']:
        if not os.path.exists('pickled_fits'):
            os.makedirs('pickled_fits')

    # load data
    gv_data = io_utils.format_h5_data('data/omega_pi_k_spec.h5',switches)

    # l s plots
    if switches['plot_ls']:
        plt.ion()
        plotting.plot_l_s(gv_data, switches, phys_point)
        plotting.plot_lF_a(gv_data, switches, phys_point)
        plotting.plot_raw_data(gv_data, switches, phys_point)
        plt.ioff()
        plt.show()
        sys.exit()

    models = analysis.sys_models(switches)
    if switches['prior_search']:
        print('Performing Prior Width Scan')
        print(r'%33s &  nnlo_x &  nnlo_a &  n3lo_x &  n3lo_a & logGBF_max' %'model')
        print('======================================================================================')
        for model in models:
            model_list, FF, fv = analysis.gather_model_elements(model)
            fit_model  = chipt.FitModel(model_list, _fv=fv, _FF=FF)
            fitEnv     = FitEnv(gv_data, fit_model, switches)
            analysis.prior_width_scan(model, fitEnv, fit_model, priors, switches)
    else: # do analysis
        fit_results = dict()
        w0_results  = dict()

        if True:
            model = 'w0_nnlo_a0_FV_all'
            model_list, FF, fv, aa = analysis.gather_w0_elements(model)
            fit_model  = chipt.FitModel(model_list, _fv=fv, _FF=FF)
            switches['w0_aa_lst'] = aa
            fitEnv     = FitEnv(gv_data, fit_model, switches)
            p_copy = copy.deepcopy(priors)
            tmp_w0_result = fitEnv.fit_w0(p_copy)
            phys_data = copy.deepcopy(phys_point)
            for k in tmp_w0_result.p:
                if isinstance(k,str):
                    phys_data['p'][k] = tmp_w0_result.p[k]
            tmp_w0_result.phys = dict()
            aw0_keys = {'a15':'a15m135XL','a12':'a12m130','a09':'a09m135'}
            for a in aa:
                phys_data['p']['w0_0'] = tmp_w0_result.p[(a,'w0_0')]
                if a in ['a15','a12','a09'] and 'a0' not in model:
                    phys_data['p']['aw0'] = tmp_w0_result.p[(aw0_keys[a],'aw0')]
                elif a == 'a06' and 'a0' not in model:
                    phys_data['p']['aw0'] = 1 / gv.gvar('3.0119(19)')
                fit_model_tmp = chipt.FitModel(model_list, _fv=False, _FF=FF)
                tmp_w0_result.phys['w0_a_'+a] = FitEnv._w0_function(fit_model_tmp, phys_data['x'], phys_data['p'])

            w0_results['all'] = tmp_w0_result
            if 'a0' in model:
                plt.ion()
                plotting.plot_w0(model, model_list, fitEnv, tmp_w0_result, switches, phys_data)
                plt.ioff()
                plt.show()
            if True: # if do individual also
                for a in aa:
                    model_a = model.replace('all',a)
                    model_list, FF, fv, a_a = analysis.gather_w0_elements(model_a)
                    print(a,model_list)
                    switches['w0_aa_lst'] = a_a
                    fit_model  = chipt.FitModel(model_list, _fv=fv, _FF=FF)
                    fitEnv     = FitEnv(gv_data, fit_model, switches)
                    p_copy = copy.deepcopy(priors)
                    tmp_w0_result = fitEnv.fit_w0(p_copy)
                    phys_data = copy.deepcopy(phys_point)
                    for k in tmp_w0_result.p:
                        if isinstance(k,str):
                            phys_data['p'][k] = tmp_w0_result.p[k]
                    tmp_w0_result.phys = dict()
                    phys_data['p']['w0_0'] = tmp_w0_result.p[(a,'w0_0')]
                    fit_model_tmp = chipt.FitModel(model_list, _fv=False, _FF=FF)
                    tmp_w0_result.phys['w0_'+a] = FitEnv._w0_function(fit_model_tmp, phys_data['x'], phys_data['p'])
                    w0_results[a] = tmp_w0_result
                switches['w0_aa_lst'] = aa


        plt.ion()
        for model in models:
            print('===============================================================')
            print(model)
            model_list, FF, fv = analysis.gather_model_elements(model)
            fit_model  = chipt.FitModel(model_list, _fv=fv, _FF=FF)
            fitEnv     = FitEnv(gv_data, fit_model, switches)

            do_fit = False
            if switches['save_fits'] or switches['debug_save_fit']:
                pickled_fit = 'pickled_fits/'+model+'_lo_x_'+str(ip.lo_x)+'_lo_a_'+str(ip.lo_a)
                pickled_fit += '_nlo_x_'+str(ip.nlo_x)+'_nlo_a_'+str(ip.nlo_a)
                pickled_fit += '_n2lo_x_'+str(ip.nnlo_x)+'_n2lo_a_'+str(ip.nnlo_a)+'.p'
                if os.path.exists(pickled_fit):
                    print('reading %s' %pickled_fit)
                    fit_result = gv.load(pickled_fit)
                    check_pickled_fit(fit_result,switches,priors)
                else:
                    do_fit = True
            else:
                do_fit = True
            if do_fit or switches['debug_save_fit']:
                tmp_result = fitEnv.fit_data(priors)
                tmp_result.phys_point = dict()
                tmp_result.phys_point.update({k:v for k,v in phys_point['p'].items() if ('Lchi' in k) or k in ['mpi','mk','mkp']})
                if switches['debug_save_fit']:
                    print('live fit')
                    report_phys_point(tmp_result, phys_point, model_list, FF, report=True)
                    analysis.uncertainty_breakdown(tmp_result,'FKFpi',print_error=True)
                    print('pickled fit')
                    if not os.path.exists(pickled_fit):
                        gv.dump(tmp_result, pickled_fit, add_dependencies=True)
                        fit_result = gv.load(pickled_fit)
                    report_phys_point(fit_result, phys_point, model_list, FF, report=True)
                    analysis.uncertainty_breakdown(fit_result,'FKFpi',print_error=True)
                if do_fit:
                    fit_result = tmp_result

            if switches['print_fit']:
                print(fit_result.format(maxline=True))
            fit_result.phys_point.update({k:v for k,v in phys_point['p'].items() if ('Lchi' in k) or k in ['mpi','mk','mkp']})
            fit_result.ensembles_fit = switches['ensembles_fit']
            report_phys_point(fit_result, phys_point, model_list, FF, report=switches['report_phys'])
            print(w0_results['all'].format(maxline=True))
            print('---------------------------------------')
            print('a        global         individual     ')
            print('---------------------------------------')
            for a in aa:
                print("%s = %11s fm  %11s fm" %(a, \
                    fit_result.phys['w0'] / w0_results['all'].phys['w0_a_'+a],
                    fit_result.phys['w0'] / w0_results[a].phys['w0_'+a]))
            print('---------------------------------------')
            fit_results[model] = fit_result
            if switches['save_fits']:
                gv.dump(fit_result, pickled_fit, add_dependencies=True)

            if switches['debug_phys_point'] and not do_fit:
                report_phys_point(fit_result, phys_point, model_list, FF)
            if switches['make_extrap'] or switches['make_fv']:
                plots = plotting.ExtrapolationPlots(model, model_list, fitEnv, fit_result, switches)
            if switches['make_extrap']:
                plots.plot_vs_eps_pi(phys_point,eps='l')
                #plots.plot_vs_eps_pi(phys_point,eps='s')
                if 'alphaS' not in model and 'ma' not in model:
                    plots.plot_vs_eps_asq(phys_point)
            if switches['make_fv']:
                if 'xpt' in model and 'FV' in model and 'F' in model:
                    plots.plot_vs_ml()

        if switches['model_avg']:
            model_avg = analysis.BayesModelAvg(fit_results)
            model_avg.print_weighted_models()
            model_avg.bayes_model_avg()
            if switches['make_hist']:
                model_avg.plot_bma_hist('FF',save_fig=switches['save_figs'])
                model_avg.plot_bma_hist('FF_xpt',save_fig=switches['save_figs'])
                model_avg.plot_bma_hist('ratio',save_fig=switches['save_figs'])
                model_avg.plot_bma_hist('ct',save_fig=switches['save_figs'])

        plt.ioff()
        if run_from_ipython():
            plt.show(block=False)
            return fit_results
        else:
            plt.show()

        return fit_results

'''
    This is the main class that runs a given fit specified by a model
'''
class FitEnv:
    # xyp_dict is a dictionary with keys 'x', 'y', 'p'
    # 'y' value is a dict{ensemble : yval},
    # 'x' value is a dict{ensemble : {<all the x variables which are not priors>}
    # 'p' value is a dict{(ensemble, pKey) : aPal} for all the
    #       ensemble-specific priors like meson masses as well as teh LECs
    def __init__(self, xyp_dict, model, switches):
        self.switches   = switches
        self.ensembles  = switches['ensembles_fit']
        self.x          = xyp_dict['x']
        self.pruned_x   = {ens : { k : v for k, v in xyp_dict['x'][ens].items()}
                                for ens in self.ensembles}
        self.y          = xyp_dict['y']
        self.y_w0       = {ens: xyp_dict['p'][(ens,'w0a')] for ens in self.ensembles}
        self.pruned_y   = {ens : xyp_dict['y'][ens] for ens in self.ensembles}
        self.p          = xyp_dict['p']
        required_params = model.get_required_parameters()
        self.pruned_p   = {(ens, k) : v for (ens, k), v in xyp_dict['p'].items()
                                if k in required_params and ens in self.ensembles}
        self.model      = model

    # create a callable function that acts on a single x and p (not all ensembles)
    @classmethod
    def _fit_function(cls, a_model, x, p):
        return a_model(x,p)

    def fit_function(self, x, p):
        a_result = dict()
        for ens in x.keys():
            p_ens = dict()
            for k, v in p.items():
                if type(k) == tuple and k[0] == ens:
                    p_ens[k[1]] = v # the x-params which are priors
                else:
                    p_ens[k] = v    # the LECs of the fit
            model = self.model
            #print(ens,p_ens)
            a_result[ens] = FitEnv._fit_function(model, x[ens], p_ens)
        return a_result

    def fit_data(self, lec_priors):
        required_params = self.model.get_required_parameters()
        # add the LEC priors to our list of priors for the fit
        self.pruned_p.update({ k:v for k,v in lec_priors.items() if k in required_params})
        x = self.pruned_x
        y = self.pruned_y
        p = self.pruned_p
        if self.switches['scipy']:
            fitter='scipy_least_squares'
        else:
            fitter='gsl_multifit'
        return lsqfit.nonlinear_fit(data=(x,y), prior=p, fcn=self.fit_function, fitter=fitter, debug=True)

    @classmethod
    def _w0_function(cls, a_model, x, p):
        return a_model(x,p)

    def w0_function(self, x, p):
        a_result = dict()
        for aa in self.switches['w0_aa_lst']:
            for ens in x.keys():
                if aa in ens:
                    p_ens = dict()
                    for k, v in p.items():
                        if type(k) == tuple and (k[0] in [ens, aa]):
                            p_ens[k[1]] = v # the x-params which are priors
                        else:
                            p_ens[k] = v    # the LECs of the fit
                    model = self.model
                    #print(ens,p_ens)
                    a_result[ens] = FitEnv._w0_function(model, x[ens], p_ens)
        return a_result

    def fit_w0(self, lec_priors):
        required_params = self.model.get_required_parameters()
        # add the LEC priors to our list of priors for the fit
        self.pruned_p.update({ k:v for k,v in lec_priors.items() if k in required_params})
        if 'w0_0' in required_params:
            self.pruned_p.update({(aa,'w0_0'):lec_priors[(aa,'w0_0')] for aa in self.switches['w0_aa_lst']})
        x = self.pruned_x
        y = {k:v for k,v in self.y_w0.items() for aa in self.switches['w0_aa_lst'] if aa in k}
        p = self.pruned_p
        if self.switches['scipy']:
            fitter='scipy_least_squares'
        else:
            fitter='gsl_multifit'
        return lsqfit.nonlinear_fit(data=(x,y), prior=p, fcn=self.w0_function, fitter=fitter, debug=True)


def report_phys_point(fit_result, phys_point_params, model_list, FF, report=False):
    phys_data = copy.deepcopy(phys_point_params)
    fit_model = chipt.FitModel(model_list, _fv=False, _FF=FF)
    for k in fit_result.p:
        if isinstance(k,str):
            phys_data['p'][k] = fit_result.p[k]
    fit_result.phys                = dict()
    fit_result.phys['w0_mO']       = FitEnv._fit_function(fit_model, phys_data['x'], phys_data['p'])
    fit_result.phys['w0']          = fit_result.phys['w0_mO'] *197.327 / phys_point_params['p']['m_omega']
    if report:
        print('  chi2/dof [dof] = %.2f [%d]   Q=%.3f   logGBF = %.3f' \
            %(fit_result.chi2/fit_result.dof, fit_result.dof, fit_result.Q, fit_result.logGBF))
        print('  w0 * m_O              = %s' %fit_result.phys['w0_mO'])
        print('  w0                    = %s' %(fit_result.phys['w0']))


def debug_fit_function(check_fit, model_list, FF, fv):
    x = check_fit['x']
    p = check_fit['p']
    fit_model = chipt.FitModel(model_list, _fv=False, _FF=FF)
    cP        = chipt.ConvenienceDict(fit_model, x, p)
    result    = 0.
    result_FV = 0.
    if fv:
        fit_model_fv = chipt.FitModel(model_list, _fv=True, _FF=FF)
        cP_FV        = chipt.ConvenienceDict(fit_model_fv, x, p)
        for term in model_list:
            if '_nlo' in term:
                t_FV = getattr(chipt.FitModel, term)(fit_model_fv, x, p, cP_FV)
                t    = getattr(chipt.FitModel, term)(fit_model, x, p, cP)
                result    += t
                result_FV += t_FV
                print('%16s   ' %(term+'_FV'), t_FV)
                print('%16s   ' %(term), t)
            else:
                t = getattr(chipt.FitModel, term)(fit_model, x, p, cP)
                result_FV += t
                result    += t
                print('%16s   ' %(term), t)
    else:
        for term in model_list:
            t = getattr(chipt.FitModel, term)(fit_model, x, p, cP)
            result += t
            print('%16s   ' %(term), t)
    print('---------------------------------------')
    if fv:
        print('%16s   %f' %('total_FV', result_FV))
    print('%16s   %f' %('total', result))

def check_pickled_fit(fit,switches,priors):
    ''' make sure the pickled data is consistent with choices in switches '''
    if not set(fit.ensembles_fit) == set(switches['ensembles_fit']):
        sys.exit('ensembles_fit from the pickled fit does not match ensembles_fit from input_params')
    for p in priors:
        if p in fit.prior:
            if priors[p].mean != fit.prior[p].mean or priors[p].sdev != fit.prior[p].sdev:
                sys.exit('prior %s from fit, %s, does not match from input_params %s' \
                    %(p,fit.prior[p],priors[p]))


if __name__ == "__main__":
    main()
