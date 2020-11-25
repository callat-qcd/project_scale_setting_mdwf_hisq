import pandas as pd
import numpy as np
import gvar as gv
import sys
import datetime
import re
import os
import yaml
import h5py
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path

# Set defaults for plots
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = (6.75, 6.75/1.618034333)
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True

class data_loader(object):

    def __init__(
            self, 
            collection=None, 
            models=None, 
            excluded_ensembles=None, 
            empirical_priors=None,
            data_file=None, 
            use_charm_reweighting=None,
            use_milc_aw0=None):

        self.project_path = os.path.normpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))

        # Set options for fits
        # Default values
        if collection is None:
            name = str(datetime.datetime.now())
            for c in [' ', ':', '.', '-']:
                name = name.replace(c, '_')
        else:
            name = collection

        defaults = {
            'name' : name, 
            'models' : [
                'Fpi_n2lo',
                'Fpi_n2lo_alphas',
                'Fpi_n2lo_alphas_fv',
                'Fpi_n2lo_fv',
                'Fpi_n2lo_log',
                'Fpi_n2lo_log_alphas',
                'Fpi_n2lo_log_alphas_fv',
                'Fpi_n2lo_log_fv',
                'Fpi_n3lo',
                'Fpi_n3lo_alphas',
                'Fpi_n3lo_alphas_fv',
                'Fpi_n3lo_fv',
                'Fpi_n3lo_log_log2',
                'Fpi_n3lo_log_log2_alphas',
                'Fpi_n3lo_log_log2_alphas_fv',
                'Fpi_n3lo_log_log2_fv',
                'Om_n2lo',
                'Om_n2lo_alphas',
                'Om_n2lo_alphas_fv',
                'Om_n2lo_fv',
                'Om_n2lo_log',
                'Om_n2lo_log_alphas',
                'Om_n2lo_log_alphas_fv',
                'Om_n2lo_log_fv',
                'Om_n3lo',
                'Om_n3lo_alphas',
                'Om_n3lo_alphas_fv',
                'Om_n3lo_fv',
                'Om_n3lo_log_log2',
                'Om_n3lo_log_log2_alphas',
                'Om_n3lo_log_log2_alphas_fv',
                'Om_n3lo_log_log2_fv'],
            'excluded_ensembles' : None,
            'data_file' : 'omega_pi_k_spec',
            'empirical_priors' : None,
            'use_charm_reweighting' : False,
            'use_milc_aw0' : False,
        }

        # Override values with those from yaml file
        try:
            with open(self.project_path+'/results/'+name+'/settings.yaml') as file:
                output = yaml.safe_load(file)
            defaults.update(output)
        except FileNotFoundError:
            pass

        # Override default/yaml values with those from initializing data_loader object
        collection = {
            'name' : name, 
            'models' : models,
            'excluded_ensembles' : excluded_ensembles,
            'data_file' : data_file,
            'empirical_priors' : empirical_priors,
            'use_charm_reweighting' : use_charm_reweighting,
            'use_milc_aw0' : use_milc_aw0,
        }
        for key in collection:
            if collection[key] is None:
                collection[key] = defaults[key]

        self.collection = collection


    @property
    def bs_data(self):

        with h5py.File(self.project_path +'/data/omega_pi_k_spec.h5', 'r') as f:
            ensembles = sorted(list(f.keys()))

        if self.collection['excluded_ensembles'] is not None:
            for ens in self.collection['excluded_ensembles']:
                ensembles.remove(ens)
        #ensembles.remove('a12m220ms')
        #ensembles.remove('a06m310L')

        #print('load data')
        data = {}
        with h5py.File(self.project_path +'/data/omega_pi_k_spec.h5', 'r') as f:
            for ens in ensembles:
                data[ens] = {}

                # Old charm reweighting data
                #if self.use_charm_reweighting:
                #    data[ens]['mO'] = f[ens]['m_omega_reweight'][:]
                #else:
                #    data[ens]['mO'] = f[ens]['m_omega'][:]
                data[ens]['mO'] = f[ens]['m_omega'][:]

                to_gvar = lambda arr : gv.gvar(arr[0], arr[1])
                if self.collection['use_milc_aw0']:
                    data[ens]['a/w'] = to_gvar(f[ens]['aw0_milc'][:])
                else:
                    data[ens]['a/w'] = 1.0 / to_gvar(f[ens]['w0a_callat'][:])

                # arrays
                for param in ['Fpi', 'mk', 'mpi']:
                    data[ens][param] = f[ens][param][:]

                # scalars
                for param in ['L', 'alpha_s']:
                    data[ens][param] = f[ens][param][()]
                    
            if self.collection['use_charm_reweighting']:
                data['a06m310L']['mO'] = f['a06m310L']['m_omega_reweight'][:]
                data['a06m310L']['mk'] = f['a06m310L']['mk_reweight'][:]
                data['a06m310L']['mpi'] = f['a06m310L']['mpi_reweight'][:]


        if self.collection['data_file'] == 'omega_pi_k_spec':
            return data
        else:
            h5_filepath = self.project_path +'/data/' +self.collection['data_file'] +'.h5'
            with h5py.File(h5_filepath, 'r') as f:        
                for ens in ensembles:
                    # Fix mismatched naming schemes
                    if ens == 'a12m220ms':
                        data['a12m220ms']['mO'] = f['a12m220_ms']['mO'][:]
                    elif ens in list(f.keys()):
                        data[ens]['mO'] = f[ens]['mO'][:]

            return data

    @property
    def gv_data(self):
        gv_data = {}

        for ens in sorted(self.bs_data):
            gv_data[ens] = gv.BufferDict()
            for param in ['mO', 'mpi', 'mk', 'Fpi']:
                gv_data[ens][param] = self.bs_data[ens][param][1:]

            gv_data[ens] = gv.dataset.avg_data(gv_data[ens], bstrap=True) 

            for param in ['mO', 'mpi', 'mk', 'Fpi']: 
                gv_data[ens][param] = gv_data[ens][param] - gv.mean(gv_data[ens][param]) + self.bs_data[ens][param][0]
            
            gv_data[ens]['a/w'] = self.bs_data[ens]['a/w']
            gv_data[ens]['L'] = self.bs_data[ens]['L']
            gv_data[ens]['alpha_s'] = self.bs_data[ens]['alpha_s']

        return gv_data


    @property
    def phys_point_data(self):
        phys_point_data = {
            'a/w' : gv.gvar(0),
            'a' : gv.gvar(0),
            'alpha_s' : gv.gvar(0.0),
            'L' : gv.gvar(np.infty),
            'hbarc' : gv.gvar(197.3269602),

            'Fpi' : gv.gvar('92.07(57)'),
            'mpi' : gv.gvar('134.8(3)'), # '138.05638(37)'
            'mk' : gv.gvar('494.2(3)'), # '495.6479(92)'
            #'mss' : gv.gvar('688.5(2.2)'), # Taken from arxiv/1303.1670
            'mO' : gv.gvar('1672.43(32)')
        }
        return phys_point_data

    def _pickle_fit_info(self, fit_info):
        model = fit_info['name']
        if not os.path.exists(self.project_path +'/results/'+ self.collection['name'] +'/pickles/'):
            os.makedirs(self.project_path +'/results/'+ self.collection['name'] +'/pickles/')
        filename = self.project_path +'/results/'+ self.collection['name'] +'/pickles/'+ model +'.p'

        output = {
            'w0' : fit_info['w0'],
            'logGBF' : gv.gvar(fit_info['logGBF']),
            'chi2/df' : gv.gvar(fit_info['chi2/df']),
            'Q' : gv.gvar(fit_info['Q']),
        }

        for key in fit_info['prior'].keys():
            output['prior:'+key] = fit_info['prior'][key]
            #output[key] = [fit_info['prior'][key], fit_info['posterior'][key]]

        for key in fit_info['posterior'].keys():
            output['posterior:'+key] = fit_info['posterior'][key]

        for key in fit_info['phys_point'].keys():
            # gvar can't handle integers -- entries not in correlation matrix
            output['phys_point:'+key] = fit_info['phys_point'][key]

        for key in fit_info['error_budget']:
            output['error_budget:'+key] = gv.gvar(fit_info['error_budget'][key])

        gv.dump(output, filename)
        return None

    def _unpickle_fit_info(self, model):
        filepath = self.project_path +'/results/'+ self.collection['name'] +'/pickles/'+ model +'.p'
        if os.path.isfile(filepath):
            return gv.load(filepath)
        else:
            return None

    def get_fit(self, model):
        return None

    def get_fit_collection(self):
        if os.path.exists(self.project_path +'/results/'+ self.collection['name'] +'/pickles/'):
            output = {}

            models = []
            for file in os.listdir(self.project_path +'/results/'+ self.collection['name'] +'/pickles/'):
                if(file.endswith('.p')):
                    models.append(file.split('.')[0])

            for model in models:
                fit_info_model = self._unpickle_fit_info(model=model)
                output[model] = {}
                output[model]['name'] = model
                output[model]['w0'] = fit_info_model['w0']
                output[model]['logGBF'] = fit_info_model['logGBF'].mean
                output[model]['chi2/df'] = fit_info_model['chi2/df'].mean
                output[model]['Q'] = fit_info_model['Q'].mean
                output[model]['prior'] = {}
                output[model]['posterior'] = {}
                output[model]['phys_point'] = {}
                output[model]['error_budget'] = {}

                for key in fit_info_model.keys():
                    if key.startswith('prior'):
                        output[model]['prior'][key.split(':')[-1]] = fit_info_model[key]
                    elif key.startswith('posterior'):
                        output[model]['posterior'][key.split(':')[-1]] = fit_info_model[key]
                    elif key.startswith('phys_point'):
                        output[model]['phys_point'][key.split(':')[-1]] = fit_info_model[key]
                    elif key.startswith('error_budget'):
                        output[model]['error_budget'][key.split(':')[-1]] = fit_info_model[key].mean

            return output


    def get_model_info_from_name(self, name):
        model_info = {}
        model_info['name'] = name

        if name.startswith('Fpi'):
            model_info['chiral_cutoff'] = 'Fpi'
        elif name.startswith('Om'):
            model_info['chiral_cutoff'] = 'mO'
        

        # Order
        if '_nlo' in name:
            model_info['order'] = 'nlo'
            model_info['latt_ct'] = 'nlo'
        elif '_n2lo' in name:
            model_info['order'] = 'n2lo'
            model_info['latt_ct'] = 'n2lo'
        elif '_n3lo' in name:
            model_info['order'] = 'n3lo'
            model_info['latt_ct'] = 'n3lo'
        else:
            model_info['order'] = 'lo'
            model_info['latt_ct'] = 'lo'

        # Order for latt terms
        if '_a2' in name:
            model_info['latt_ct'] = 'nlo'
        elif '_a4' in name:
            model_info['latt_ct'] = 'n2lo'
        elif '_a6' in name:
            model_info['latt_ct'] = 'n3lo'

        # Include other corrections
        model_info['include_log'] = bool('_log_' in name or name.endswith('log'))
        model_info['include_log2'] = bool('_log2' in name)
        model_info['include_fv'] = bool('_fv' in name)
        model_info['include_alphas'] = bool('_alphas' in name)

        model_info['exclude'] = []

        model_info['name']  = self.get_model_name_from_info(model_info)


        return model_info

    def get_model_name_from_info(self, model_info):
        if model_info['chiral_cutoff'] == 'mO':
            name = 'Om'
        elif model_info['chiral_cutoff'] == 'Fpi':
            name = 'Fpi'

        if model_info['order'] == 'lo':
            name += '_lo'
        elif model_info['order'] == 'nlo': 
            name += '_nlo'
        elif model_info['order'] == 'n2lo': 
            name += '_n2lo'
        elif model_info['order'] == 'n3lo': 
            name += '_n3lo'

        if model_info['include_log']:
            name += '_log'
        if model_info['include_log2']:
            name += '_log2'

        if 'latt_ct' in model_info:
            if model_info['latt_ct'] != model_info['order']:
                if model_info['latt_ct'] == 'nlo':
                    name += '_a2'
                elif model_info['latt_ct'] == 'n2lo':
                    name += '_a4'
                elif model_info['latt_ct'] == 'n3lo':
                    name += '_a6'

        if model_info['include_alphas']:
            name += '_alphas'


        if model_info['include_fv']:
            name += '_fv'

        return name

    def get_prior(self, model=None, model_info=None, default=False, interpolation=False):

        def get_default_prior(model):
            with open(self.project_path+'/data/default_priors.yaml') as file:
                output = yaml.safe_load(file)
                if interpolation:
                    prior = gv.gvar(output['interpolation'])
                    self.save_prior('interpolation', prior)
                    return prior

                model_info = self.get_model_info_from_name(model)
                if model_info['chiral_cutoff'] == 'mO':
                    prior = gv.gvar(output['default_mO'])
                    self.save_prior('default_mO', prior)
                elif model_info['chiral_cutoff'] ==  'Fpi':
                    prior = gv.gvar(output['default_Fpi'])
                    self.save_prior('default_Fpi', prior)
                return prior

        if not interpolation:
            if model is None:
                model = self.get_model_name_from_info(model_info)
            elif model not in ['default_Fpi', 'default_mO']:
                # Standardize name formatting (eg, '*_fv_alphas' -> '_alphas_fv')
                model = self.get_model_info_from_name(model)['name']

        if default:
            return get_default_prior(model)


        try:
            with open(self.project_path+'/results/'+self.collection['name']+'/priors.yaml') as file:
                output = yaml.safe_load(file)
                try:
                    return gv.gvar(output[model])
                except KeyError:
                    try:
                        if interpolation:
                            return gv.gvar(output['interpolation'])

                        if model_info is None:
                            model_info = self.get_model_info_from_name(model)
                            
                        if model_info['chiral_cutoff'] == 'mO':
                            prior = gv.gvar(output['default_mO'])
                        elif model_info['chiral_cutoff'] ==  'Fpi':
                            prior = gv.gvar(output['default_Fpi'])
                        print('Using default prior for', model)
                        return prior

                    except KeyError:
                        return get_default_prior(model)
        except FileNotFoundError:
            return get_default_prior(model)
            


    def plot_qq(self, ens, param):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        stats.probplot(self.bs_data[ens][param], dist='norm', plot=ax)

        fig = plt.gcf()
        plt.close()

        return fig

    def save_fit_info(self, fit_info):
        self._pickle_fit_info(fit_info)
        return None

    def save_results_summary(self, output_string):

        # converts a dictionary of dictionaries to a markdown table
        def dict_dict_to_table(dict_dict, column0='', exclude=[], sort_key=None):
            header = []
            for item in dict_dict:
                header = np.union1d(header, list(dict_dict[item]))

            header = sorted(header, key=sort_key)
            for key in exclude:
                if key in header:
                    header.remove(key)
                

            table = '| %s |' %(column0)
            for key in header:
                table += ' %s |' %(key)
            table += '\n'

            table += '| --- |'
            for key in header:
                table += ' --- |'
            table += '\n'

            for item in dict_dict:
                table += '| %s |' %(item)
                for key in header:
                    if key in dict_dict[item]:
                        table += ' %s |' %(str(dict_dict[item][key]))
                    else:
                        table += '  |'
                table += '\n'

            return table

        filename = self.project_path +'/results/'+ self.collection['name'] + '/README.md'

        # Make table for priors
        models = self.collection['models'].copy()
        if any([True if self.get_model_info_from_name(mdl)['chiral_cutoff'] == 'Fpi' else False 
                    for mdl in self.collection['models']]):
            models.insert(0, 'default_Fpi')
        if any([True if self.get_model_info_from_name(mdl)['chiral_cutoff'] == 'mO' else False
                    for mdl in self.collection['models']]):
            models.insert(0, 'default_mO')
        priors_models = {mdl : self.get_prior(mdl) for mdl in models}

        # Sort keys by lo, nlo, n2lo, etc
        length = lambda x : len(x.split('_')[1]) if len(x.split('_')) > 1 else 0
        prior_table = dict_dict_to_table(priors_models, column0='model', sort_key=length)

        # Generate table for models    
        models = self.collection['models']
        models_descriptions = {mdl : self.get_model_info_from_name(mdl) for mdl in models}
        models_table = dict_dict_to_table(models_descriptions, exclude=['name'], column0='model')

        # Generate table for input params
        inputs_table = dict_dict_to_table(self.gv_data, column0='ens')

        output_string  = '### Priors\n' +prior_table+ '\n### Model List\n' +models_table+ '\n### Inputs\n' +inputs_table+ '\n' +output_string

        if os.path.exists(filename):
            with open(filename, 'r') as file:
                file_content = file.read()
            if '## Autogenerated' in file_content:
                file_content = re.sub(r'## Autogenerated\s(.*\s*)*\Z',
                                      '## Autogenerated\n'+output_string,
                                      file_content)
                with open(filename, 'w') as file:
                    print(file_content)
                    file.write(file_content)

                return None

        # else
        with open(filename, 'a+') as file:
            file_content = '\n## Autogenerated\n'
            file_content += output_string
            file.write(file_content)

        return None

    def save_fig(self, fig=None, output_filename=None):

        if fig is None:
            print('Nothing here!')
            return None

        if output_filename is None:
            if not os.path.exists(os.path.normpath(self.project_path+'/tmp/')):
                os.makedirs(os.path.normpath(self.project_path+'/tmp/'))
            output_file = os.path.normpath(self.project_path+'/tmp/temp.png')
        else:
            output_file = os.path.normpath(self.project_path+'/results/'+self.collection['name']+'/'+output_filename+'.png')

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        fig.savefig(output_file, bbox_inches='tight')
        return None

    def save_settings(self):
        filepath = self.project_path+'/results/'+self.collection['name']+'/settings.yaml'
        Path(self.project_path+'/results/'+self.collection['name']).mkdir(parents=True, exist_ok=True)

        output = self.collection.copy()
        del(output['name'])
        with open(filepath, 'w') as file:
            yaml.dump(output, file)

        return None

    def save_prior(self, model, prior):
        filepath = self.project_path+'/results/'+self.collection['name']+'/priors.yaml'
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                yaml_content = yaml.safe_load(file) or {}
                yaml_content.update({model : {str(key) : str(prior[key]) for key in prior}})
        else:
            yaml_content = {model : {str(key) : str(prior[key]) for key in prior}}

        with open(filepath, 'w') as file:
            yaml.safe_dump(yaml_content, file)

        return None
