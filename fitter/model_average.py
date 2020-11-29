import lsqfit
import numpy as np
import gvar as gv
import time
#import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import scipy.stats as stats

# plot defaults
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
import fitter.data_loader as dl

class model_average(object):
    def __init__(self, collection):
        self.collection = collection
        self.data_loader = dl.data_loader(collection=collection)
        self.fit_results = self.data_loader.get_fit_collection()
        self.other_results = {
            #'ETMC' : '0.1782(0)', 
            #'QCDSF-UKQCD' : '0.179(6)',
            'HPQCD [2013]' : '0.1715(9)', # hep-lat/1303.1670, Mar 2013
            'ALPHA [2013]' : '0.1757(13)', # hep-lat/1311.5585, Nov 2013
            'HotQCD [2015]' : '0.1749(14)',  # hep-lat/1501.07652, Jan 2015
            'MILC [2015]' : '0.1714(15)', # hep-lat/1503.02769, Mar 2015
            'BMWc [2020]' : '0.17180(39)', # Unpublished?, ?? 2020
        }
        self.other_results = gv.gvar(self.other_results)


    def __str__(self):
        output = ''
        for observable in list(self.fit_results):
            param = observable
            extrapolated_value = self.average(param, observable=observable, split_unc=True)


            output  += '%s: %s\n'%(observable, self.average(param, observable=observable))
            #output += '[FLAG:     %s]\n'%(self._get_phys_point_data()[param])

            #sig_fig = lambda x : np.around(x, int(np.floor(-np.log10(x))+3)) if x>0 else x
            output += '\n---\n'
            output += 'Uncertainty: \n'
            output += '   RMS model sdev:  % .5f \n' %(extrapolated_value[1])
            output += '   Model unc:       % .5f \n' %(extrapolated_value[2])

            error_budget = self.error_budget(observable=observable)
            if error_budget['chiral'] is not None:
                output += '\n---\n'
                output += 'Error budget (RMS model sdev): \n'
                output += '   Statistical: % .5f \n' %(error_budget['stat'])
                output += '   Chiral:      % .5f \n' %(error_budget['chiral'])
                output += '   Disc:        % .5f \n' %(error_budget['disc'])
                output += '   Phys point:  % .5f \n' %(error_budget['pp_input'])
                

            model_list = self.get_model_names(observable=observable, by_weight=True)
            weight = lambda model_k : np.exp(self.fit_results[observable][model_k]['logGBF']) / np.sum([np.exp(self.fit_results[observable][model_l]['logGBF']) for model_l in model_list])
            output += '\n---\n'
            output += 'Highest Weight: \n'
            for k in range(np.min([5, len(model_list)])):
                output += '  % .3f:  %s\n' %(weight(model_list[k]), model_list[k])
            output += '\n------\n'

        return output


    def _get_model_info_from_name(self, name):
        return self.data_loader.get_model_info_from_name(name)


    def _get_fit_posterior(self, model, observable):
        if 'posterior' in self.fit_results[observable][model] and bool(self.fit_results[observable][model]['posterior']):
            return self.fit_results[observable][model]['posterior']
        else:
            return None


    def _get_fit_prior(self, model, observable):
        if 'prior' in self.fit_results[observable][model] and bool(self.fit_results[observable][model]['prior']):
            return self.fit_results[observable][model]['prior']
        else:
            return None


    def _get_phys_point_data(self):
        phys_point_data = self.data_loader.phys_point_data
        return phys_point_data


    def _param_keys_dict(self, param):
        if param == 'w0':
            return '$w_0$'
        elif param == 'Fpi':
            return '$F_\pi$'
        elif param == 'mO':
            return '$m_\Omega$'

        else:
            return param


    def average(self, param, observable=None, models=None, split_unc=False, include_unc=True):
        if observable is None:
            observable = param
        if models is None:
            models = self.get_model_names(observable=observable)

        y = {}
        for mdl in models:

            # LECs
            if self._get_fit_posterior(mdl, observable=observable) is not None and param in self._get_fit_posterior(mdl, observable=observable):
                y[mdl] =  self._get_fit_posterior(mdl, observable=observable)[param]

            # Error budget
            elif param.startswith('eb:') and 'error_budget' in self.fit_results[observable][mdl]:
                y[mdl] = self.fit_results[observable][mdl]['error_budget'][param.split(':')[-1]]

            # w0, t0, etc
            elif param in self.fit_results[observable][mdl]:
                y[mdl] = self.fit_results[observable][mdl][param]

            else:
                y[mdl] = None


        # Only get results that aren't None
        nonempty_keys = []
        for mdl in models:
            if (y[mdl] is not np.nan) and (y[mdl]is not None):
                nonempty_keys.append(mdl)

        if nonempty_keys == []:
            return None

        # calculate P( M_k | D )
        prob_Mk_given_D = lambda model_k : (
             np.exp(self.fit_results[observable][model_k]['logGBF']) / np.sum([np.exp(self.fit_results[observable][model_l]['logGBF']) for model_l in nonempty_keys])
        )

        # Get central value
        expct_y = 0
        for mdl in nonempty_keys:
            expct_y += gv.mean(gv.gvar(y[mdl])) *prob_Mk_given_D(mdl)

        if not include_unc:
            return expct_y

        # Get variance
        if not split_unc:
            var_y = 0
            for mdl in nonempty_keys:
                var_y += gv.var(gv.gvar(y[mdl])) *prob_Mk_given_D(mdl)
            for mdl in nonempty_keys:
                var_y += (gv.mean(gv.gvar(y[mdl])))**2 *prob_Mk_given_D(mdl)

            var_y -= (expct_y)**2

            return gv.gvar(expct_y, np.sqrt(var_y))

        # Split statistics (unexplained var)/model selection (explained var)
        if split_unc:
            var_model = 0
            for mdl in nonempty_keys:
                var_model += gv.var(gv.gvar(y[mdl])) *prob_Mk_given_D(mdl)

            var_selection = 0
            for mdl in nonempty_keys:
                var_selection += (gv.mean(gv.gvar(y[mdl])))**2 *prob_Mk_given_D(mdl)
            var_selection -= (expct_y)**2

            return [expct_y, np.sqrt(var_model), np.sqrt(var_selection)]


    def error_budget(self, observable):
        output = {}
        for key in ['chiral', 'pp_input', 'stat', 'disc']:
            output[key] = self.average('eb:'+key, observable=observable, include_unc=False)

        return output


    def fitfcn(self, model, data, observable, p=None):
        model_info = self._get_model_info_from_name(model).copy()

        if p is None:
            p = self._get_fit_posterior(model, observable=observable)

        fitfcn = fit.model(datatag='xpt', model_info=model_info).fitfcn
        return fitfcn


    def get_model_names(self, observable, by_weight=False):
        if by_weight:
            temp = {model : self.fit_results[observable][model]['logGBF'] for model in self.fit_results[observable]}
            sorted_list = [model for model, logGBF
                           in sorted(temp.items(), key=lambda item: item[1], reverse=True)]
            return sorted_list

        else:
            return sorted(list(self.fit_results[observable]))


    def plot_comparison(self, param, observable=None, title=None, xlabel=None,
                        show_model_avg=True):

        if observable is None:
            observable = param
        if title is None:
            title = ""
        if xlabel is None:
            xlabel = self._param_keys_dict(param)

        colors = ['salmon', 'darkorange', 'mediumaquamarine', 'orchid', 'navy']

        #results_array = [self.fit_results[observable], {name : {name : self.other_results[name]} for name in self.other_results}]
        #results = self.fit_results[observable]

        fig = plt.figure(figsize=(8, 8))

        # These axes compare fits
        ax_fits = plt.axes([0.10,0.10,0.49,0.8])

        def by_order(model):
            number = len(model) / 1000
            if len (model.split('_')) == 1:
                return -1

            order = model.split('_')[1]
            
            if order == 'lo':
                number += 0
            elif order == 'nlo':
                number += 1
            else:
                number += int(order[1])
            
            if '_log' in model:
                number += 0.25
            if '_log2' in model:
                number += 0.5

            return number

        y=0
        labels = np.array([])
        if param == 'w0':
            for collab in self.other_results:
                param_value = self.other_results[collab]
                color = 'deepskyblue'
                x = gv.mean(param_value)
                xerr = gv.sdev(param_value)
                plt.errorbar(x=x, y=y, xerr=xerr, yerr=0.0,
                            alpha=0.8, color=color, elinewidth=5.0)
                labels = np.append(labels, str(collab))

                y = y + 1

        y_other = y

        plt.axhline(y-0.5, ls='--')
        for name in sorted(self.fit_results[observable], key=by_order):

            param_value = None

            if param in self.fit_results[observable][name].keys():
                param_value = gv.gvar(self.fit_results[observable][name][param])
            elif 'posterior' in self.fit_results[observable][name] and param in self.fit_results[observable][name]['posterior'].keys():
                param_value = gv.gvar(self.fit_results[observable][name]['posterior'][param])
            else:
                param_value = None

            if param_value is not None:
                # Color by fit_type model
                model_info = self._get_model_info_from_name(name)
                if model_info['order'] == 'lo':
                    color = colors[0]
                elif model_info['order'] == 'nlo':
                    color = colors[1]
                elif model_info['order'] == 'n2lo':
                    color = colors[2]
                elif model_info['order'] == 'n3lo':
                    color = colors[3]
                else:
                    color = colors[4]

                x = gv.mean(param_value)
                xerr = gv.sdev(param_value)

                plt.errorbar(x=x, y=y, xerr=xerr, yerr=0.0,
                                capsize=0.0, mec='white', ms=10.0, alpha=0.8,
                                color=color, elinewidth=5.0, label=name.split('_')[1])
                y = y + 1
                labels = np.append(labels, str(name))

        ymax = y

        # Show model average
        if show_model_avg:
            try:
                avg = self.average(param, observable=observable)
                pm = lambda g, k : gv.mean(g) + k*gv.sdev(g)
                plt.axvspan(pm(avg, -1), pm(avg, +1), alpha=0.3, color='cornflowerblue', label='avg')
                plt.axvline(pm(avg, 0), ls='-.', color='m')
            except:
                pass

        labels = [l.replace('_', ' ') for l in sorted(labels, key=by_order)]
        plt.yticks(range(len(labels)), labels)
        plt.tick_params('y', labelsize=mpl.rcParams['font.size'])
        #plt.yticks(list(self.fit_results[observable]))
        plt.ylim(-1, y)
        if param == 'w0':
            plt.xlim(0.1685, 0.1775)

        # Get unique labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
            ncol=len(by_label), bbox_to_anchor=(0,1), loc='lower left')


        locs, labels = plt.xticks()
        plt.xticks(locs[:-1])
        plt.xlabel(xlabel)
        plt.title(title)

        plt.grid(ls='--')

        # These axes compare the logGBFs
        ax_logGBF = plt.axes([0.60,0.10,0.09,0.8])

        # Get max logGBF
        logGBF_max = np.nanmax([gv.mean(gv.gvar(self.fit_results[observable][model]['logGBF']))
                               for model in self.fit_results[observable].keys()])

        y=y_other
        labels = np.array([])
        for name in sorted(self.fit_results[observable], key=by_order):
            model_info = self._get_model_info_from_name(name)
            if model_info['order'] == 'lo':
                color = colors[0]
            elif model_info['order'] == 'nlo':
                color = colors[1]
            elif model_info['order'] == 'n2lo':
                color = colors[2]
            elif model_info['order'] == 'n3lo':
                color = colors[3]
            else:
                color = colors[4]


            logGBF = gv.mean(gv.gvar(self.fit_results[observable][name]['logGBF']))
            x = np.exp(logGBF - logGBF_max)

            alpha = 1
            if x < 0.01:
                alpha = 0

            #plt.axvline(x, ls='--', alpha=0.4)
            plt.scatter(x=x, y=y, color=color, alpha=alpha)
            y = y + 1
            labels = np.append(labels, str(name))

        for ti in np.arange(5)/4.0:
            plt.axvline(ti, ls='--', alpha=0.2)

        plt.xticks([0, 1])
        plt.yticks([])
        plt.xlim(-0.15, 1.15)
        plt.ylim(-1, ymax)
        plt.xlabel("GBF")

        # These axes compare the Q-values
        ax_Q = plt.axes([0.70,0.10,0.09,0.8])

        y=y_other
        labels = np.array([])
        for name in sorted(self.fit_results[observable], key=by_order):
            model_info = self._get_model_info_from_name(name)
            if model_info['order'] == 'lo':
                color = colors[0]
            elif model_info['order'] == 'nlo':
                color = colors[1]
            elif model_info['order'] == 'n2lo':
                color = colors[2]
            elif model_info['order'] == 'n3lo':
                color = colors[3]
            else:
                color = colors[4]

            x = gv.mean(gv.gvar(self.fit_results[observable][name]['Q']))
            #plt.axvline(x, ls='--', alpha=0.4)
            plt.scatter(x=x, y=y, color=color)
            y = y + 1
            labels = np.append(labels, str(name))

        for ti in np.arange(5)/4.0:
            plt.axvline(ti, ls='--', alpha=0.2)

        plt.xticks([0, 1])
        plt.yticks([])
        plt.xlim(-0.15, 1.15)
        plt.ylim(-1, ymax)
        plt.xlabel("$Q$")


        # These axes compare the reduced chi2
        ax_chi2 = plt.axes([0.80,0.10,0.1,0.8])

        y=y_other
        labels = np.array([])
        for name in sorted(self.fit_results[observable], key=by_order):
            model_info = self._get_model_info_from_name(name)
            if model_info['order'] == 'lo':
                color = colors[0]
            elif model_info['order'] == 'nlo':
                color = colors[1]
            elif model_info['order'] == 'n2lo':
                color = colors[2]
            elif model_info['order'] == 'n3lo':
                color = colors[3]
            else:
                color = colors[4]

            x = gv.mean(gv.gvar(self.fit_results[observable][name]['chi2/df']))
            #plt.axvline(x, ls='--', alpha=0.4)
            plt.scatter(x=x, y=y, color=color)
            y = y + 1
            labels = np.append(labels, str(name))

        for ti in np.arange(9)/4.0:
            plt.axvline(ti, ls='--', alpha=0.2)

        plt.xticks([0, 1, 2])
        plt.yticks([])
        plt.xlim(-0.25, 2.25)
        plt.ylim(-1, ymax)
        plt.xlabel(r"$\chi^2_\nu$")

        fig = plt.gcf()
        #plt.show()
        plt.close()

        return fig


    # parameter = 'a', 'mpi', 'volume'
    # !Only works for lam_chi = 4 pi Fpi!
    def plot_fits(self, parameter, observable):

        # Check that posterior contains covarianace matrix
        temp_model_name = list(self.get_model_names(observable=observable))[0]
        temp_po = self._get_fit_posterior(temp_model_name, observable=observable)
        temp_p = self._get_fit_prior(temp_model_name, observable=observable)
        xi = None

        if (temp_p is None) or (temp_po is None) or (gv.uncorrelated(temp_po['wm0'], temp_p['wm0'])):
            print('Correlations between gvar variables lost! Fit plots will be inaccurate.')
            return None

        colors = ['skyblue', 'orchid', 'mediumaquamarine', 'darkorange', 'silver']
        #colors = ['cyan', 'magenta', 'yellow', 'black', 'silver']

        data = self._get_phys_point_data()
        if parameter == 'a':
            xlabel = r'$\epsilon^2_a = (a/2w_0)^2$'

            x = np.linspace(0, (0.16/(2 *0.1710))**2, 50)
        elif parameter in ['mpi', 'pi', 'p', 'l']:
            xlabel = r'$l^2_F$'# = m_\pi^2 / (4 \pi F_\pi)^2$'
            x = (np.linspace(10, 400, 50) / (4 *np.pi *self._get_phys_point_data()['Fpi']))**2
            plt.axvline(gv.mean(self._get_phys_point_data()['mpi'] / (4 *np.pi *self._get_phys_point_data()['Fpi']))**2, color=colors[0])
            plt.axvline(gv.mean(self._get_phys_point_data()['mpi'] / (self._get_phys_point_data()['mO']))**2, color=colors[1])
            
            #xlabel = r'$l^2_F = m_\pi^2 / m_\Omega^2$'
            #x = (np.linspace(10, 400, 50) / (self._get_phys_point_data()['mO']))**2
            #plt.axvline(gv.mean(self._get_phys_point_data()['mpi'] / (self._get_phys_point_data()['mO']))**2)

            
        else:
            return None

        total_GBF = np.sum([np.exp(self.fit_results[observable][model_l]['logGBF']) for model_l in self.fit_results[observable].keys()])

        pm = lambda g, k : gv.mean(g) + k*gv.sdev(g)

        def by_order(model):
            number = len(model) / 1000
            if len (model.split('_')) == 1:
                return -1

            order = model.split('_')[1]
            
            if order == 'lo':
                number += 0
            elif order == 'nlo':
                number += 1
            else:
                number += int(order[1])
            
            if '_log' in model:
                number += 0.25
            if '_log2' in model:
                number += 0.5

            return number

        for j, name in enumerate(sorted(self.fit_results[observable], key=by_order)):

            model_info = self._get_model_info_from_name(name)

            xi = {}
            if parameter == 'a':
                xi['a'] = x
            elif parameter in ['mpi', 'pi', 'p', 'l']:
                xi['l'] = x #{'l' : (np.linspace(10, 400, 50) / (4 *np.pi *self._get_phys_point_data()['Fpi']))**2}

            p = self.fit_results[observable][name]['posterior'].copy()
            if model_info['chiral_cutoff'] == 'Fpi':
                data['lam_chi'] = 4 *np.pi *self._get_phys_point_data()['Fpi']
                color = colors[0]
            elif model_info['chiral_cutoff'] == 'mO':
                data['lam_chi'] = self._get_phys_point_data()['mO']
                color = colors[1]
            #if model_info[''] == '':
            #    pass
            #else:
            #    color = colors[3]

            #color = colors[j%len(colors)]

            p = self.fit_results[observable][name]['posterior']

            y = self.fitfcn(name, model_info, observable=observable)(p=p, fit_data=data, xi=xi) 
            #y = y / self._get_phys_point_data()['mO'] *self._get_phys_point_data()['hbarc']
            weight = np.exp(self.fit_results[observable][name]['logGBF']) / total_GBF
            plt.fill_between(pm(x, 0), pm(y, -1), pm(y, 1), 
                             alpha=np.max([np.min([weight, 1]), 0.1]), color=color,
                             rasterized=False, label=model_info['order']) #


        #handles, labels = plt.gca().get_legend_handles_labels()

        # format as latex
        #for j, l in enumerate(labels):
        #    if l.startswith('n') and l.endswith('lo') and len(l) > 3:
        #        nx = l[1:-2]
        #        labels[j] = 'n$^%s$lo'%(nx)
        
        #by_label = dict(zip(labels, handles))
        #plt.legend(by_label.values(), by_label.keys(),
        #    ncol=int(len(by_label)), bbox_to_anchor=(0,1), loc='lower left')
        #extrapolation_param = 'w0'
        #y = np.repeat(self._get_phys_point_data()[extrapolation_param],len(x))
        #plt.fill_between(pm(x, 0), pm(y, -1), pm(y, 1), color=colors[3], alpha=0.5, rasterized=False)


        # Add legend
        Fpi_patch = mpatches.Patch(color=colors[0], alpha=0.5, label=r'$F_\pi$')
        mO_patch = mpatches.Patch(color=colors[1], alpha=0.5, label=r'$m_\Omega$')

        plt.legend(handles=[Fpi_patch, mO_patch])

        plt.xlim(np.min(gv.mean(x)), np.max(gv.mean(x)))
        plt.xlabel(xlabel)
        plt.ylabel('$w_0 m_\Omega$')


        #w0_avg = self.average('w0')
        #pm = lambda g, k : g.mean + k *g.sdev
        #plt.ylim(pm(w0_avg, -5), pm(w0_avg, +5))

        fig = plt.gcf()
        #plt.show()
        plt.close()
        return fig


    # See self._get_model_info_from_name for possible values for 'compare'
    def plot_histogram(self, param, observable=None, title=None, xlabel=None, compare='order'):
        if observable is None:
            observable = param
        if xlabel is None:
            xlabel = self._param_keys_dict(param)
        if title is None:
            title = ""

        param_avg = self.average(param=param, observable=observable)
        pm = lambda g, k : g.mean + k *g.sdev
        x = np.linspace(pm(param_avg, -4), pm(param_avg, +4), 2000)

        # Determine ordering
        # Have larger contributions behind smaller contributions
        choices = np.unique([self._get_model_info_from_name(model)[compare] for model in self.get_model_names(observable=observable)])
        temp_dict = {choice : 0 for choice in choices}
        for model in self.get_model_names(observable=observable):
            model_info = self._get_model_info_from_name(model)
            temp_dict[model_info[compare]] += np.exp(self.fit_results[observable][model]['logGBF'])
            choices = sorted(temp_dict, key=temp_dict.get, reverse=True)

        # Set colors
        #cmap = matplotlib.cm.get_cmap('gist_rainbow')
        #colors =  ['whitesmoke']
        #colors.extend([cmap(c) for c in np.linspace(0, 1, len(choices)+1)])
        #colors = ['whitesmoke', 'salmon', 'palegreen', 'lightskyblue', 'plum']
        #colors = ['whitesmoke', 'crimson', 'springgreen', 'deepskyblue', 'magenta']
        #colors = ['whitesmoke', 'salmon', 'darkorange', 'mediumaquamarine', 'orchid', 'navy']
        colors = ['whitesmoke', 'mediumaquamarine', 'darkorange', 'salmon']
        colors = ['whitesmoke', 'mediumaquamarine', 'orchid', 'skyblue', 'darkorange']
        
        #colors.reverse()

        def by_order(order):
            number = -1
            if order == 'lo':
                number = 0
            elif order == 'nlo':
                number = 1
            elif order == 'n2lo':
                number = 2
            elif order == 'n3lo': #unused
                number = 2
            elif order=='All':
                number = 1000

            return number

        #for j, choice in enumerate(np.append(['All'], choices)):
        for j, choice in enumerate(sorted(np.append(['All'], choices), key=by_order, reverse=True)):

            # read Bayes Factors
            logGBF_list = [self.fit_results[observable][model]['logGBF'] for model in self.get_model_names(observable=observable)]

            # initiate a bunch of parameters
            y = 0
            y_list = []
            y_dict = dict()

            # weights
            w_lst = []
            wd = dict()

            # p. dist. fcn
            pdf = 0
            pdfdict = dict()

            # c. dist. fcn.
            cdf = 0
            cdfdict = dict()

            for model in self.get_model_names(observable=observable):
                model_info = self._get_model_info_from_name(model)

                r = np.nan
                if param == '':
                    r = self._get_fit_extrapolation(model)

                elif (self._get_fit_posterior(model, observable=observable) is not None) and (param in self._get_fit_posterior(model, observable=observable)):
                    r =  self._get_fit_posterior(model, observable=observable)[param]

                elif param in self.fit_results[observable][model]:
                    r = gv.gvar(self.fit_results[observable][model][param])

                else:
                    r = np.nan

                if r is not np.nan and r is not None and (str(model_info[compare]) == choice or choice=='All'):
                    #r = gv.gvar(self.fit_results[observable][model][param])
                    y_dict[model] = r

                    w = 1/sum(np.exp(np.array(logGBF_list)-self.fit_results[observable][model]['logGBF']))
                    sqrtw = np.sqrt(w) # sqrt scales the std dev correctly
                    wd[model] = w
                    w_lst.append(w)

                    y += gv.gvar(w*r.mean,sqrtw*r.sdev)
                    y_list.append(r.mean)

                    p = stats.norm.pdf(x,r.mean,r.sdev)
                    pdf += w*p
                    pdfdict[model] = w*p


                    c = stats.norm.cdf(x,r.mean,r.sdev)
                    cdf += w*c
                    cdfdict[model] = w*c

            y_list = np.array(y_list)
            w_lst = np.array(w_lst)

            plot_params = {'x':x, 'pdf':pdf, 'pdfdict':pdfdict, 'cdf':cdf, 'cdfdict':cdfdict}

            gr = 1.618034333
            fs2_base = 3.50394
            lw = 0.5
            fs_l = 15
            fs_xy = 24
            ts = 15

            x = plot_params['x']
            ysum = plot_params['pdf']

            ydict = plot_params['pdfdict']
            cdf = plot_params['cdf']


            fig = plt.figure('result histogram')#,figsize=fig_size2)
            ax = plt.axes()

            #for a in ydict.keys():
            #    ax.plot(x,ydict[a], color=colors[j], alpha=1.0, ls='dotted')


            ax.fill_between(x=x,y1=ysum,facecolor=colors[j], edgecolor='black',alpha=0.4,label=self._param_keys_dict(choice))
            #ax.plot(x, ysum, color=colors[j], alpha=1.0, lw=4.0)
            ax.plot(x, ysum, color='k', alpha=1.0)
            if choice == 'All':
                # get 95% confidence
                lidx95 = abs(cdf-0.025).argmin()
                uidx95 = abs(cdf-0.975).argmin()
                ax.fill_between(x=x[lidx95:uidx95],y1=ysum[lidx95:uidx95],facecolor=colors[j],edgecolor='black',alpha=0.4)
                # get 68% confidence
                lidx68 = abs(cdf-0.158655254).argmin()
                uidx68 = abs(cdf-0.841344746).argmin()
                ax.fill_between(x=x[lidx68:uidx68],y1=ysum[lidx68:uidx68],facecolor=colors[j],edgecolor='black',alpha=0.4)
                # plot black curve over
                ax.errorbar(x=[x[lidx95],x[lidx95]],y=[0,ysum[lidx95]],color='black',lw=lw)
                ax.errorbar(x=[x[uidx95],x[uidx95]],y=[0,ysum[uidx95]],color='black',lw=lw)
                ax.errorbar(x=[x[lidx68],x[lidx68]],y=[0,ysum[lidx68]],color='black',lw=lw)
                ax.errorbar(x=[x[uidx68],x[uidx68]],y=[0,ysum[uidx68]],color='black',lw=lw)

                ax.errorbar(x=x,y=ysum,ls='-',color='black',lw=lw)


                
            leg = ax.legend(edgecolor='k',fancybox=False)
            ax.set_ylim(bottom=0)
            #ax.set_xlim([1.225,1.335])
            ax.set_xlabel('$w_0$ (fm)')
            frame = plt.gca()
            frame.axes.get_yaxis().set_visible(False)
            #ax.xaxis.set_tick_params(labelsize=ts,width=lw)

            # legend line width
            [ax.spines[key].set_linewidth(lw) for key in ax.spines]
            leg.get_frame().set_linewidth(lw)

        fig = plt.gcf()
        plt.close()
        return fig
