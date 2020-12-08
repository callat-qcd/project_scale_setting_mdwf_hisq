#!/usr/bin/env python3

import tables as h5
import numpy as np
import gvar as gv
import sys

'''
This file reads the data from an hdf5 file and prepares it into a correlated
gvar data set.

It also handles reading/writing yaml files containing prior width studies
'''

def sort_ens(ensembles):
    # sort ensembles in a useful order
    sorted = list(ensembles)
    sorted.sort(reverse=True)
    return sorted

def format_h5_data(data_path, switches):
    data     = h5.open_file(data_path,'r')
    x        = dict()
    y_w0     = dict()
    y_w0_imp = dict()
    y_t0     = dict()
    y_t0_imp = dict()
    p        = dict()
    if switches['bs_bias']:
        print('Shifting BS data to boot0')
    else:
        print('Treating mean as one of the bootstrap samples')
    if switches['print_lattice']:
        fits_y = []
        fits_x = []

    print('%9s w_0 m_Omega' %'ensemble')
    print('-----------------------------------------------------------------')
    for ens in sort_ens(switches['ensembles']):
        x[ens] = dict()
        data_dict = dict()
        for q in ['mpi','mk', 'm_omega']:
            data_dict[q] = data.get_node('/'+ens+'/'+q).read()
        try:
            data_dict['Fpi'] = data.get_node('/'+ens+'/Fpi').read()
        except Exception as e:
            print(e)

        if ens == 'a06m310L' and switches['reweight']:
            for q in ['mpi','mk', 'm_omega']:
                data_dict[q] = data.get_node('/'+ens+'/'+q+'_reweight').read()

        if ens == 'a09m135' and switches['deflate_a09']:
            print('deflating %s uncertainty' %ens)
            print(data_dict['m_omega'].shape)
            print(data_dict['m_omega'].mean(), data_dict['m_omega'].std())
            dy = data_dict['m_omega'] - data_dict['m_omega'].mean()
            dy = dy / np.sqrt(4)
            data_dict['m_omega'] = data_dict['m_omega'].mean() + dy
            print(data_dict['m_omega'].mean(), data_dict['m_omega'].std())

        if ens == 'a06m310L' and switches['deflate_a06']:
            print('deflating %s uncertainty' %ens)
            print(data_dict['m_omega'].shape)
            print(data_dict['m_omega'].mean(), data_dict['m_omega'].std())
            dy = data_dict['m_omega'] - data_dict['m_omega'].mean()
            dy = dy / np.sqrt(2)
            data_dict['m_omega'] = data_dict['m_omega'].mean() + dy
            print(data_dict['m_omega'].mean(), data_dict['m_omega'].std())

        if ens == 'a12m220ms' and switches['deflate_a12m220ms']:
            print('deflating %s uncertainty' %ens)
            print(data_dict['m_omega'].shape)
            print(data_dict['m_omega'].mean(), data_dict['m_omega'].std())
            dy = data_dict['m_omega'] - data_dict['m_omega'].mean()
            dy = dy / 3
            data_dict['m_omega'] = data_dict['m_omega'].mean() + dy
            print(data_dict['m_omega'].mean(), data_dict['m_omega'].std())

        if switches['bs_bias']:
            data_bs = dict()
            for d in data_dict:
                data_bs[d] = data_dict[d][1:]
            gvdata = gv.dataset.avg_data(data_bs,bstrap=True)
            if switches['debug_bs']:
                print('-----------------------------------------------')
                print(ens)
                print('data set  | bs mean    | bs bias corrected mean')
                print('----------|------------|-----------------------')
                gvdata_copy = dict(gvdata)
            for d in data_bs:
                gvdata[d] = gvdata[d] + (data_dict[d][0] - gvdata[d].mean)
                if switches['debug_bs']:
                    print("%7s    %.10f  %.10f" %(d, gvdata_copy[d].mean, gvdata[d].mean))
        else:
            gvdata = gv.dataset.avg_data(data_dict,bstrap=True)

        L_ens = data.get_node('/'+ens+'/L').read()
        x[ens]['mpiL'] = gvdata['mpi'].mean * L_ens
        x[ens]['alphaS'] = data.get_node('/'+ens+'/alpha_s').read()

        if switches['w0'] == 'callat':
            w0_a = data.get_node('/'+ens+'/w0a_callat').read()
            p[(ens,'w0a')] = gv.gvar(w0_a[0],w0_a[1])
            p[(ens,'aw0')] = 1 / p[(ens,'w0a')]

            w0_a_imp = data.get_node('/'+ens+'/w0a_callat_imp').read()
            p[(ens,'w0a_imp')] = gv.gvar(w0_a_imp[0],w0_a_imp[1])
            p[(ens,'aw0_imp')] = 1 / p[(ens,'w0a_imp')]

            t0_a2 = data.get_node('/'+ens+'/t0aSq').read()
            p[(ens,'t0a2')] = gv.gvar(t0_a2[0],t0_a2[1])
            p[(ens,'at0')]  = 1 / np.sqrt(p[(ens,'t0a2')])

            t0_imp = data.get_node('/'+ens+'/t0aSq_imp').read()
            p[(ens,'t0a2_imp')] = gv.gvar(t0_imp[0],t0_imp[1])
            p[(ens,'at0_imp')]  = 1 / np.sqrt(p[(ens,'t0a2_imp')])

            # try using definition of eps_a from a particular GF scale
            if not switches['fixed_eps_a']:
                eps_a_dict = {
                    'w0':p[(ens,'aw0')], 'w0_imp':p[(ens,'aw0_imp')],
                    't0':p[(ens,'at0')], 't0_imp':p[(ens,'at0_imp')]
                    }
                p[(ens,'aw0')] = eps_a_dict[switches['gf_scale']]
            else:
                pass
                #p[(ens,'aw0')] = p[(ens,'at0')]

            if switches['gf_scale'] in ['t0_imp_1','t0_imp_2','t0_imp_3']:
                imp_study = True
                t0_imp_n = data.get_node('/'+ens+'/t0aSq_imp_'+switches['gf_scale'].split('_')[-1]).read()
                p[(ens,'t0a2_imp_n')] = gv.gvar(t0_imp_n[0],t0_imp_n[1])
            else:
                imp_study = False
        elif switches['w0'] == 'milc':
            a_w0 = data.get_node('/'+ens+'/aw0_milc').read()
            p[(ens,'aw0')] = gv.gvar(a_w0[0], a_w0[1])
            p[(ens,'w0a')] = 1 / p[(ens,'aw0')]
        y_w0[ens]     = gvdata['m_omega'] * p[(ens,'w0a')]
        y_w0_imp[ens] = gvdata['m_omega'] * p[(ens,'w0a_imp')]
        y_t0[ens]     = gvdata['m_omega'] * np.sqrt(p[(ens,'t0a2')])
        if imp_study:
            y_t0_imp[ens] = gvdata['m_omega'] * np.sqrt(p[(ens,'t0a2_imp_n')])
        else:
            y_t0_imp[ens] = gvdata['m_omega'] * np.sqrt(p[(ens,'t0a2_imp')])

        if switches['gf_scale'] == 'w0':
            print("%9s %s" %(ens,y_w0[ens]))
        elif switches['gf_scale'] == 'w0_imp':
            print("%9s %s" %(ens,y_w0_imp[ens]))
        elif switches['gf_scale'] == 't0':
            print("%9s %s" %(ens,y_t0[ens]))
        elif switches['gf_scale'] in ['t0_imp','t0_imp_1','t0_imp_2','t0_imp_3']:
            print("%9s %s" %(ens,y_t0_imp[ens]))

        # MASSES
        p[(ens,'mpi')]     = gvdata['mpi']
        p[(ens,'mk')]      = gvdata['mk']
        x[ens]['Lam_Of']   = gvdata['m_omega'].mean
        p[(ens,'Lam_O')]   = gvdata['m_omega']
        if 'Fpi' in gvdata:
            p[(ens,'Lam_F')] = 4 * np.pi * gvdata['Fpi']

        if switches['print_lattice']:
            # ens mO w0_o w0_i w0_omO w0_imO t0_o t0_i t0_omO t0_imO
            fits_y.append('%9s& %s& %s& %s& %s& %s& %s& %s& %s& %s\\\\' \
                %(ens, gvdata['m_omega'],\
                p[(ens,'t0a2')], p[(ens,'t0a2_imp')], y_t0[ens], y_t0_imp[ens],\
                p[(ens,'w0a')],  p[(ens,'w0a_imp')],  y_w0[ens], y_w0_imp[ens]\
                ))
            # ens lFsq sFsq lOsq sOsq mpiL eps_a alphaS
            fits_x.append('%9s& %s& %s& %s& %s& %.2f& %s& %s\\\\' \
                %(ens,\
                (gvdata['mpi']/4/np.pi/gvdata['Fpi'])**2,\
                (2*gvdata['mk']**2 - gvdata['mpi']**2)/(4*np.pi*gvdata['Fpi'])**2,\
                (gvdata['mpi']/gvdata['m_omega'])**2, \
                (2*gvdata['mk']**2 - gvdata['mpi']**2)/gvdata['m_omega']**2,\
                x[ens]['mpiL'], (p[(ens,'aw0')] / 2)**2, x[ens]['alphaS']))

    if switches['print_lattice']:
        print(r'ensemble& $am_\O$& $t_{0,\rm orig}/a^2$& $t_{0,\rm imp}/a^2$& $\sqrt{t_{0,\rm orig}}m_\O$& $\sqrt{t_{0,\rm imp}}m_\O$& $w_{0,\rm orig}/a$& $w_{0,\rm imp}/a$& $w_{0,\rm orig} m_\O$& $w_{0,\rm imp} m_\O$ \\')
        print(r'\hline')
        for l in fits_y:
            print(l)
            if any(ens in l for ens in ['a15m135XL','a12m130','a09m135']):
                print("\\hline")
        print('')
        print(r'ensemble& $l_F^2$& $s_F^2$& $l_\Omega^2$& $s_\Omega^2$& $m_\pi L$& $\e_a^2$& $\a_S$\\')
        print(r'\hline')
        for l in fits_x:
            print(l)
            if any(ens in l for ens in ['a15m135XL','a12m130','a09m135']):
                print("\\hline")
        print('')
        data.close()
        sys.exit()

    data.close()
    return {'x':x, 'y_w0':y_w0, 'y_w0_imp':y_w0_imp, 'y_t0':y_t0, 'y_t0_imp':y_t0_imp, 'p':p}
