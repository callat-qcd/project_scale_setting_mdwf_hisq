import numpy as np
import gvar as gvar
import os
import sys
import time
import pathlib
import argparse
from progress.bar import Bar

sys.path.append(pathlib.Path(__file__).parent.absolute())
import fitter.model_average as md
import fitter.data_loader as dl
import fitter.fit_manager as fm

project_dir = pathlib.Path(__file__).parent.absolute()

# Parser
parser = argparse.ArgumentParser(description='Perform scale setting')

# data_loader options
parser.add_argument(
    '-c', '--collection', dest='collection_name', type=str, default=None,
    help='fit with priors and models specified in /results/[collection]/{prior.yaml,settings.yaml} and save results'
)
parser.add_argument(
    '-m', '--models', dest='models', nargs='+', default=None,
    help='fit specified models'
)
parser.add_argument(
    '-ex', '--exclude', dest='excluded_ensembles', nargs='+', default=None,
    help='exclude specified ensembles from fit'
)
parser.add_argument(
    '-em', '--empirical_priors', dest='empirical_priors', type=str, choices=['all', 'order', 'disc', 'alphas', 'disc_only'], default=None,
    help='determine empirical priors for models'
)
parser.add_argument(
    '-df', '--data_file', dest='data_file', type=str, default=None,
    help='fit with specified h5 file'
)
parser.add_argument(
    '-re', '--reweight', dest='use_charm_reweighting', default=None, action='store_true',
    help='use charm reweightings on a06m310L'
)
parser.add_argument(
    '-mc', '--milc', dest='use_milc_aw0', default=False, action='store_true',
    help="use milc's determinations of a/w0"
)
parser.add_argument(
    '-o', '--original', dest='improved_observables', default=True, action='store_false',
    help="Use original, not discretization improved t0/a^2 and w0/a values"
)

# fitting/model-averaging options
parser.add_argument(
    '-nf', '--no_fit', dest='perform_fits', default=True, action='store_false', 
    help='do not fit models'
)
parser.add_argument(
    '-na', '--no_average', dest='average_models', default=True, action='store_false',
    help='do not average models'
)
parser.add_argument(
    '-d', '--default', dest='use_default_priors', default=False, action='store_true',
    help='use default priors; defaults to using optimized priors if present, otherwise default priors'
)
args = vars(parser.parse_args())

data_loader = dl.data_loader(
    collection=args['collection_name'], 
    models=args['models'], 
    excluded_ensembles=args['excluded_ensembles'], 
    empirical_priors=args['empirical_priors'],
    data_file=args['data_file'], 
    use_charm_reweighting=args['use_charm_reweighting'],
    use_milc_aw0=args['use_milc_aw0'],
    improved_observables=args['improved_observables'],

)
data_loader.save_settings()


collection = data_loader.collection

model_list = collection['models']

# Optimize fits via empirical Bayes
if args['empirical_priors'] is None:
    if collection['empirical_priors'] is not None:
        args['empirical_priors'] = collection['empirical_priors']

if args['empirical_priors'] is not None:
    t0 = time.time()
    bar = Bar('Optimizing priors', max=len(model_list))
    for j, model in enumerate(model_list):
        bar.next()
        print('\n', model)

        # Load data
        gv_data = data_loader.gv_data
        phys_point_data = data_loader.phys_point_data
        prior = data_loader.get_prior(model=model, default=True)
        model_info = data_loader.get_model_info_from_name(model)

        fit_manager = fm.fit_manager(
            fit_data=gv_data,
            phys_point_data=phys_point_data, 
            prior=prior, 
            model_info = model_info
        )

        optimal_prior = fit_manager.optimize_prior(empbayes_grouping=args['empirical_priors'])
        data_loader.save_prior('w0_'+fit_manager.model, optimal_prior['w0'])
        data_loader.save_prior('t0_'+fit_manager.model, optimal_prior['t0'])

        
    t1 = time.time()
    print("Total time (s): ", t1 - t0, "\n")

# Perform scale setting
if args['perform_fits']:
    t0 = time.time()
    bar = Bar('Fitting models', max=len(model_list))
    for j, model in enumerate(model_list):
        bar.next()
        print('\n', model)

        # Load data
        gv_data = data_loader.gv_data
        phys_point_data = data_loader.phys_point_data
        model_info = data_loader.get_model_info_from_name(model)
        if args['use_default_priors']:
            prior = data_loader.get_prior(model=model, default=True)
        else:
            prior = data_loader.get_prior(model=model)

        fit_manager = fm.fit_manager(
            fit_data=gv_data,
            phys_point_data=phys_point_data, 
            prior=prior, 
            model_info = model_info
        )

        #print(fit_manager)
        data_loader.save_fit_info(fit_manager.fit_info)

    t1 = time.time()
    print("Total time (s): ", t1 - t0, "\n")



# Average results
if args['average_models']:
    model_average = md.model_average(collection['name'])
    str_output = '### Model Average\n```yaml\n'+str(model_average)+'```\n'


    # w0/t0 comparison figs
    fig = model_average.plot_comparison(param='w0', observable='w0')
    data_loader.save_fig(fig, output_filename='/figs/w0_comparison_fits')
    str_output += '![image](./figs/w0_comparison_fits.png)\n' 

    fig = model_average.plot_comparison(param='sqrt_t0', observable='t0')
    data_loader.save_fig(fig, output_filename='/figs/t0_comparison_fits')
    str_output += '![image](./figs/t0_comparison_fits.png)\n' 

    # w0/t0 histogram figs
    fig = model_average.plot_histogram(param='w0', observable='w0', compare='order')
    data_loader.save_fig(fig, output_filename='/figs/w0_histogram_fit_order')
    str_output += '![image](./figs/w0_histogram_fit_order.png)\n' 

    fig = model_average.plot_histogram(param='sqrt_t0', observable='t0', compare='order')
    data_loader.save_fig(fig, output_filename='/figs/t0_histogram_fit_order')
    str_output += '![image](./figs/t0_histogram_fit_order.png)\n' 

    # Plot all fits
    fig = model_average.plot_fits('mpi', observable='w0')
    data_loader.save_fig(fig, output_filename='/figs/w0_fits_vs_mpi')
    str_output += '![image](./figs/w0_fits_vs_mpi.png)\n' 

    fig = model_average.plot_fits('mpi', observable='t0')
    data_loader.save_fig(fig, output_filename='/figs/t0_fits_vs_mpi')
    str_output += '![image](./figs/t0_fits_vs_mpi.png)\n' 


    #str_output += '\n## Highest Weight Models'
    str_output += '\n## Representative model'

    #for j, model in enumerate(model_average.get_model_names(observable='w0', by_weight=True)[:5]):
    for j, model in enumerate(['Fpi_n3lo_log_log2_fv']):
        print('Making figs for model:', model)

        # Load data
        gv_data = data_loader.gv_data
        phys_point_data = data_loader.phys_point_data
        model_info = data_loader.get_model_info_from_name(model)
        if args['use_default_priors']:
            prior = data_loader.get_prior(model=model, default=True)
        else:
            prior = data_loader.get_prior(model=model)
        

        fit_manager = fm.fit_manager(
            fit_data=gv_data,
            phys_point_data=phys_point_data, 
            prior=prior, 
            model_info = model_info
        )

        str_output += '\n```yaml\n'+str(fit_manager)+'```\n'

        # add w/a interpolation plots for highest weight fig
        if j == 0:
            str_output += '\n### w0 interpolation\n'
            for latt_spacing in np.unique([ens[:3] for ens in fit_manager.ensembles]):
                fig = fit_manager.plot_interpolation(latt_spacing=latt_spacing, observable='w0')
                data_loader.save_fig(fig, output_filename='/figs/w0_interpolation_'+latt_spacing)
                str_output += '![image](./figs/w0_interpolation_'+latt_spacing+'.png)\n' 
            str_output += '\n### t0 interpolation\n'
            for latt_spacing in np.unique([ens[:3] for ens in fit_manager.ensembles]):
                fig = fit_manager.plot_interpolation(latt_spacing=latt_spacing, observable='t0')
                data_loader.save_fig(fig, output_filename='/figs/t0_interpolation_'+latt_spacing)
                str_output += '![image](./figs/t0_interpolation_'+latt_spacing+'.png)\n' 


        # Plot observables vs eps2_a
        str_output += '\n### Lattice dependence\n'
        str_output += '![image](./figs/fits/'+str(j+1)+'w0_vs_a--'+fit_manager.model+'.png)\n'
        str_output += '![image](./figs/fits/'+str(j+1)+'sqrt_t0_vs_a--'+fit_manager.model+'.png)\n'
        fig = fit_manager.plot_fit('a', observable='w0')
        data_loader.save_fig(fig, output_filename='/figs/fits/'+str(j+1)+'w0_vs_a--'+fit_manager.model)
        fig = fit_manager.plot_fit('a', observable='t0')
        data_loader.save_fig(fig, output_filename='/figs/fits/'+str(j+1)+'sqrt_t0_vs_a--'+fit_manager.model)

        # Plot observables vs l^2
        str_output += '\n### Light quark mass dependence\n'
        str_output += '![image](./figs/fits/'+str(j+1)+'w0_vs_l--'+fit_manager.model+'.png)\n' 
        str_output += '![image](./figs/fits/'+str(j+1)+'sqrt_t0_vs_l--'+fit_manager.model+'.png)\n' 
        fig = fit_manager.plot_fit('pi', observable='w0')
        data_loader.save_fig(fig, output_filename='/figs/fits/'+str(j+1)+'w0_vs_l--'+fit_manager.model)
        fig = fit_manager.plot_fit('pi', observable='t0')
        data_loader.save_fig(fig, output_filename='/figs/fits/'+str(j+1)+'sqrt_t0_vs_l--'+fit_manager.model)        



        # Plot observables vs s^2
        str_output += '\n### Strange quark mass dependence\n'
        str_output += '![image](./figs/fits/'+str(j+1)+'w0_vs_s--'+fit_manager.model+'.png)\n'
        str_output += '![image](./figs/fits/'+str(j+1)+'sqrt_t0_vs_s--'+fit_manager.model+'.png)\n'
        fig = fit_manager.plot_fit('k', observable='w0')
        data_loader.save_fig(fig, output_filename='/figs/fits/'+str(j+1)+'w0_vs_s--'+fit_manager.model)
        fig = fit_manager.plot_fit('k', observable='t0')
        data_loader.save_fig(fig, output_filename='/figs/fits/'+str(j+1)+'sqrt_t0_vs_s--'+fit_manager.model)


    # Save fit info to /results/{collection}/README.md
    data_loader.save_results_summary(str_output)