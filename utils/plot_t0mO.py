import os, sys
import gvar as gv
import matplotlib.pyplot as plt

ext = 'pdf'
#sys.path.append('..')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
import fit_mOmega

import io_utils
import chipt
import analysis
import plotting

import input_params as ip
# Load input params
switches   = ip.switches
phys_point = ip.phys_point

model = 'xpt_n3lo_FV_F'
add_eps_var=False

''' t0 with fixed eps_a = a / (2w_0,orig) '''
switches['fixed_eps_a'] = False
gv_data = io_utils.format_h5_data('data/omega_pi_k_spec.h5',switches)
p_fits = {}
for gf in ['t0','t0_imp']:#,'t0_imp_2','t0_imp_3']:
    p_fit  = 'pickled_fits/'+model+'_'+gf+'_fixed_eps_a'
    p_fit += '_nlo_x_'+str(ip.nlo_x)+'_nlo_a_'+str(ip.nlo_a)
    p_fit += '_n2lo_x_'+str(ip.n2lo_x)+'_n2lo_a_'+str(ip.n2lo_a)
    p_fit += '_n3lo_x_'+str(ip.n3lo_x)+'_n3lo_a_'+str(ip.n3lo_a)+'.p'
    p_fits[gf] = p_fit
    if not os.path.exists(p_fit):
        sys.exit('missing %s' %p_fit)

plt.ion()
model_list, FF, fv = analysis.gather_model_elements(model)
fit_model  = chipt.FitModel(model_list, _fv=fv, _FF=FF)

switches['save_figs'] = False
switches['return_ax'] = True
for ig,gf in enumerate(p_fits):
    switches['gf_scale'] = gf
    fitEnv     = fit_mOmega.FitEnv(gv_data, fit_model, switches)
    fit_result = gv.load(p_fits[gf])
    fit_mOmega.report_phys_point(fit_result, phys_point, model_list, FF, report=True, store_phys=False)
    plots = plotting.ExtrapolationPlots(model, model_list, fitEnv, fit_result, switches)
    if ig == 0:
        ax = plots.plot_vs_eps_asq(phys_point)
    else:
        ax = plots.plot_vs_eps_asq(phys_point,ax=ax)
for t in ax.texts:
    t.set_visible(False)
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
tmp_label = dict(zip(labels, handles))
by_label = dict()
for a in ['a_{06}', 'a_{09}', 'a_{12}', 'a_{15}']:
    for k in tmp_label:
        if a in k:
            by_label[r'$%s$' %a] = tmp_label[k]
ax.legend(by_label.values(), by_label.keys(), ncol=4, loc=1, fontsize=plotting.fs_leg)
ax.text(0.15,1.21, r'$\sqrt{t_{0,\rm imp}} m_\Omega$',
    horizontalalignment='left',verticalalignment='center',fontsize=plotting.fs_text)

ax.text(0.15,1.345, r'$\sqrt{t_{0,\rm orig}} m_\Omega$', rotation=10,
    horizontalalignment='left',verticalalignment='center',fontsize=plotting.fs_text)

ax.text(0.023, 1.34, r'%s' %(model.replace('_','\_')),\
    horizontalalignment='left', verticalalignment='center', \
    fontsize=plotting.fs_text, bbox={'facecolor':'None','boxstyle':'round'})

ax.set_ylim(1.171,1.399)
ax.set_xlabel(r'$\epsilon_a^2 = a^2/(2w_{0,\rm orig})^2$',fontsize=plotting.fs_text)
ax.set_ylabel(r'$\sqrt{t_{0}}\, m_\Omega(l_F^{\rm phys}, s_F^{\rm phys}, \epsilon_a^2)$',fontsize=plotting.fs_text)

plt.savefig('figures/'+'sqrtt0_mO_vs_ea_combined_'+model+'_fixed_eps_a.'+ext,transparent=True)

if add_eps_var:
    ''' t0 with variable eps_a = a / (2gf_0) '''
    switches['fixed_eps_a'] = True
    p_fits = {}
    for gf in ['t0','t0_imp']:#,'t0_imp_2','t0_imp_3']:
        p_fit  = 'pickled_fits/'+model+'_'+gf+'_variable_eps_a'
        p_fit += '_nlo_x_'+str(ip.nlo_x)+'_nlo_a_'+str(ip.nlo_a)
        p_fit += '_n2lo_x_'+str(ip.n2lo_x)+'_n2lo_a_'+str(ip.n2lo_a)
        p_fit += '_n3lo_x_'+str(ip.n3lo_x)+'_n3lo_a_'+str(ip.n3lo_a)+'.p'
        p_fits[gf] = p_fit
        gv_data = io_utils.format_h5_data('data/omega_pi_k_spec.h5',switches)
        if not os.path.exists(p_fit):
            sys.exit('missing %s' %p_fit)

    model_list, FF, fv = analysis.gather_model_elements(model)
    fit_model  = chipt.FitModel(model_list, _fv=fv, _FF=FF)

    switches['save_figs'] = False
    switches['return_ax'] = True
    for ig,gf in enumerate(p_fits):
        switches['gf_scale'] = gf
        fitEnv     = fit_mOmega.FitEnv(gv_data, fit_model, switches)
        fit_result = gv.load(p_fits[gf])
        fit_mOmega.report_phys_point(fit_result, phys_point, model_list, FF, report=True, store_phys=False)
        plots = plotting.ExtrapolationPlots(model, model_list, fitEnv, fit_result, switches)
        if ig == 0:
            ax = plots.plot_vs_eps_asq(phys_point)
        else:
            ax = plots.plot_vs_eps_asq(phys_point,ax=ax)
    for t in ax.texts:
        t.set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    tmp_label = dict(zip(labels, handles))
    by_label = dict()
    for a in ['a_{06}', 'a_{09}', 'a_{12}', 'a_{15}']:
        for k in tmp_label:
            if a in k:
                by_label[r'$%s$' %a] = tmp_label[k]
    ax.legend(by_label.values(), by_label.keys(), ncol=4, loc=1, fontsize=plotting.fs_leg)
    ax.text(0.15,1.21, r'$\sqrt{t_{0,\rm imp}} m_\Omega$',
        horizontalalignment='left',verticalalignment='center',fontsize=plotting.fs_text)

    ax.text(0.15,1.345, r'$\sqrt{t_{0,\rm orig}} m_\Omega$', rotation=10,
        horizontalalignment='left',verticalalignment='center',fontsize=plotting.fs_text)

    ax.text(0.023, 1.34, r'%s' %(model.replace('_','\_')),\
        horizontalalignment='left', verticalalignment='center', \
        fontsize=plotting.fs_text, bbox={'facecolor':'None','boxstyle':'round'})

    ax.set_ylim(1.171,1.399)
    ax.set_xlabel(r'$\epsilon_a^2 = a^2/(2w_{0,\rm orig})^2$',fontsize=plotting.fs_text)
    ax.set_ylabel(r'$\sqrt{t_{0}}\, m_\Omega(l_F^{\rm phys}, s_F^{\rm phys}, \epsilon_a^2)$',fontsize=plotting.fs_text)

    plt.savefig('figures/'+'sqrtt0_mO_vs_ea_combined_'+model+'_variable_eps_a.'+ext,transparent=True)


#plt.cla()
p_fits = {}
for gf in ['w0','w0_imp']:
    p_fit  = 'pickled_fits/'+model+'_'+gf+'_fixed_eps_a'
    p_fit += '_nlo_x_'+str(ip.nlo_x)+'_nlo_a_'+str(ip.nlo_a)
    p_fit += '_n2lo_x_'+str(ip.n2lo_x)+'_n2lo_a_'+str(ip.n2lo_a)
    p_fit += '_n3lo_x_'+str(ip.n3lo_x)+'_n3lo_a_'+str(ip.n3lo_a)+'.p'
    p_fits[gf] = p_fit
    if not os.path.exists(p_fit):
        sys.exit('missing %s' %p_fit)

model_list, FF, fv = analysis.gather_model_elements(model)
fit_model  = chipt.FitModel(model_list, _fv=fv, _FF=FF)

switches['save_figs'] = False
switches['return_ax'] = True
for ig,gf in enumerate(p_fits):
    switches['gf_scale'] = gf
    fitEnv     = fit_mOmega.FitEnv(gv_data, fit_model, switches)
    fit_result = gv.load(p_fits[gf])
    fit_mOmega.report_phys_point(fit_result, phys_point, model_list, FF, report=True, store_phys=False)
    plots = plotting.ExtrapolationPlots(model, model_list, fitEnv, fit_result, switches)
    if ig == 0:
        ax = plots.plot_vs_eps_asq(phys_point)
    else:
        ax = plots.plot_vs_eps_asq(phys_point,ax=ax)
for t in ax.texts:
    t.set_visible(False)
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
tmp_label = dict(zip(labels, handles))
by_label = dict()

for ik,k in enumerate(labels):
    for a in ['a_{06}', 'a_{09}', 'a_{12}', 'a_{15}']:
        if a in k:
            a_new  = r'$%s^{\rm orig}$' %a
            aa_new = r'$%s^{\rm imp}$' %a
            if a_new not in by_label:
                by_label[a_new] = handles[ik]
            else:
                by_label[aa_new] = handles[ik]
'''
for a in ['a_{06}', 'a_{09}', 'a_{12}', 'a_{15}']:
    for k in tmp_label:
        if a in k:
            a_new = r'$%s^{\rm orig}$' %a
            if a_new not in by_label:
                by_label[a_new] = tmp_label[k]
            else:
                aa_new = r'$%s^{\rm imp}$' %a
                by_label[aa_new] = tmp_label[k]
'''
ax.legend(by_label.values(), by_label.keys(), ncol=4, loc=1, fontsize=plotting.fs_leg)
ax.text(0.15,1.36, r'$w_{0,\rm imp} m_\Omega$', rotation=-20,
    horizontalalignment='left',verticalalignment='center',fontsize=plotting.fs_text)

ax.text(0.15,1.4, r'$w_{0,\rm orig} m_\Omega$', rotation=-5,
    horizontalalignment='left',verticalalignment='center',fontsize=plotting.fs_text)

ax.text(0.023, 1.34, r'%s' %(model.replace('_','\_')),\
    horizontalalignment='left', verticalalignment='center', \
    fontsize=plotting.fs_text, bbox={'facecolor':'None','boxstyle':'round'})

ax.set_ylim(1.301,1.529)
ax.set_xlabel(r'$\epsilon_a^2 = a^2/(2w_{0,\rm orig})^2$',fontsize=plotting.fs_text)
ax.set_ylabel(r'$w_{0}\, m_\Omega(l_F^{\rm phys}, s_F^{\rm phys}, \epsilon_a^2)$',fontsize=plotting.fs_text)

plt.savefig('figures/'+'w0_mO_vs_ea_combined_'+model+'.'+ext,transparent=True)

plt.ioff()
plt.show()
