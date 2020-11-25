# Scale setting with m<sub>&Omega;</sub> and w<sub>0</sub>

This repository performs the chiral, continuum and infinite volume extrapolations of `w_0 m_Omega` to perform a scale setting on the [MDWF on gradient-flowed HISQ](https://arxiv.org/abs/1701.07559) action.  The present results accompany the scale setting publication available at [arXiv:2011.12166](https://arxiv.org/abs/2011.12166).

The analysis was performed by Nolan Miller ([millerb](https://github.com/millernb)) with the `master` branch, and Logan Carpenter ([
loganofcarpenter](https://github.com/orgs/callat-qcd/people/loganofcarpenter)) with cross checks by André Walker-Loud ([walkloud](https://github.com/walkloud)) on the `andre` branch.

The raw correlation functions are contained in the file `data/callat_mdwf_hisq_scale_correlators.h5` and the bootstrap results for the ground state masses and values of `Fpi` are contained in the file `data/omega_pi_k_spec.h5`.


## How to use - `andre` branch

This analysis code uses Peter Lepage's [gvar >= 11.2](https://github.com/gplepage/gvar) and [lsqfit >= 11.5.1](https://github.com/gplepage/lsqfit).  To run the analysis:

```
python fit_mOmega.py
```

Various options to control the analysis are provided in the `input_params.py` file, which controls which extrapolation models are considered, the values of the prior widths etc.
