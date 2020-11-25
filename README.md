# Scale setting with mOmega

These are python analysis scripts which reproduce the results accompanying the publication which can be found on the arXiv
```
add arxiv posting
```

The analysis was performed by Nolan Miller (`master` branch) and Logan Carpenter with cross checks by André Walker-Loud (`andre` branch).

The raw correlation functions are contained in the file `data/callat_mdwf_hisq_scale_correlators.h5` and the bootstrap results for the ground state masses and values of `Fpi` are contained in the file `data/omega_pi_k_spec.h5`.


## How to use

To generate the extrapolation and interpolation results from the paper, run `python scale-setting.py -c [name]`. This will automatically create the folder `/results/[name]/` . A summary of the results is given inside `/results/[name]/README.md`. Extra options can be viewed by running `python scale-setting.py --help`, which is given below for convenience.

```
usage: scale-setting.py [-h] [-c COLLECTION_NAME] [-m MODELS [MODELS ...]] [-ex EXCLUDED_ENSEMBLES [EXCLUDED_ENSEMBLES ...]] [-em {all,order,disc,alphas}] [-df DATA_FILE] [-re] [-mc] [-nf] [-na] [-d]

Perform scale setting

optional arguments:
  -h, --help            show this help message and exit
  -c COLLECTION_NAME, --collection COLLECTION_NAME
                        fit with priors and models specified in /results/[collection]/{prior.yaml,settings.yaml} and save results
  -m MODELS [MODELS ...], --models MODELS [MODELS ...]
                        fit specified models
  -ex EXCLUDED_ENSEMBLES [EXCLUDED_ENSEMBLES ...], --exclude EXCLUDED_ENSEMBLES [EXCLUDED_ENSEMBLES ...]
                        exclude specified ensembles from fit
  -em {all,order,disc,alphas}, --empirical_priors {all,order,disc,alphas}
                        determine empirical priors for models
  -df DATA_FILE, --data_file DATA_FILE
                        fit with specified h5 file
  -re, --reweight       use charm reweightings on a06m310L
  -mc, --milc           use milc's determinations of a/w0
  -nf, --no_fit         do not fit models
  -na, --no_average     do not average models
  -d, --default         use default priors; defaults to using optimized priors if present, otherwise default priors 
```

To fine-tune the results, either re-run the fits using the options above or by modifying `/results/[name]/settings.yaml`. Similarly, the fits can be constructed with different priors by editing `/results/[name]/priors.yaml` and re-running `python scale-setting.py -c [name]`.

In addition to this library, this repo contains Juypyter notebooks. The fit for a single model can be explored in `/notebooks/fit_model.ipynb`. The model average is provided in `/notebooks/average_models.ipynb`. Some miscellaneous drudgery (eg, the paper's sensitivity figure) is available in `/notebooks/bespoke_plots.ipynb`.

## Requirements

This work makes extensive use of Peter Lepage's Python modules [`gvar`](https://github.com/gplepage/gvar) and [`lsqfit`](https://github.com/gplepage/lsqfit), which are used to construct the fits and model average. Further, the settings and priors are primarily tweaked by the accompanying `yaml` files loaded via [`PyYAML`](https://github.com/yaml/pyyaml).