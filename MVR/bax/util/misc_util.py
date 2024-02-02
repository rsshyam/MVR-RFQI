"""
Miscellaneous utilities.
"""

from argparse import Namespace
from pathlib import Path
import os
import logging
import pickle
from collections import defaultdict
from re import X
import numpy as np

def dict_to_namespace(params):
    """
    If params is a dict, convert it to a Namespace, and return it.

    Parameters ----------
    params : Namespace_or_dict
        Namespace or dict.

    Returns
    -------
    params : Namespace
        Namespace of params
    """
    # If params is a dict, convert to Namespace
    if isinstance(params, dict):
        params = Namespace(**params)

    return params


class suppress_stdout_stderr:
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    Source: https://stackoverflow.com/q/11130156
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class Dumper:
    def __init__(self, experiment_name):
        cwd = Path.cwd()
        # while cwd.name != 'bayesian-active-control':
            # cwd = cwd.parent
        # this should be the root of the repo
        self.expdir = cwd
        logging.info(f'Dumper dumping to {cwd}')
        # if self.expdir.exists() and overwrite:
            # shutil.rmtree(self.expdir)
        # self.expdir.mkdir(parents=True)
        self.info = defaultdict(list)
        self.info_path = self.expdir / 'info.pkl'
        # args = vars(args)
        # print('Run with the following args:')
        # pprint(args)
        # args_path = self.expdir / 'args.json'
        # with args_path.open('w') as f:
            # json.dump(args, f, indent=4)

    def add(self, name, val):
        self.info[name].append(val)

    def save(self):
        with self.info_path.open('wb') as f:
            pickle.dump(self.info, f)


def batch_function(f):
    # naively batch a function by calling it on each element separately and making a list of those
    def batched_f(x_list):
        y_list = []
        for x in x_list:
            y_list.append(f(x))
        return y_list
    return batched_f

def make_postmean_fn(model):
    def postmean_fn(x):
        mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)
        mu_tup_for_x = list(zip(*mu_list))
        return mu_tup_for_x
    return postmean_fn
def make_uncertain_postmean_fn(model,Delta,para_t=1,policy_complete_perturb=False,gaus_nois=False):
    def uncertain_postmean_fn(x):
        mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)
        #print(Delta)
        #print(mu_list)
        #print(para_t,'para_t')
        if not policy_complete_perturb and not gaus_nois:
            #print(np.shape(mu_list))
            #print(np.shape(x))
            #Delta=0
            #print(mu_list)
            t=np.random.uniform(-1,1,np.shape(mu_list))
            #print(t)
            mu_list=mu_list+t*Delta
            #print(mu_list)
        elif gaus_nois:
            #t=np.random.normal(0,1,np.shape(mu_list))
            #print(t)
            t=np.random.normal(0,1,size=(np.shape(mu_list)[0],para_t))
            #print(t,'raw_noise')
            t=np.repeat(t,int(np.shape(mu_list)[1]/para_t),axis=0)
            t=t.reshape(np.shape(mu_list))
            #print(t,'repeat_noise')
            #print(mu_list,'before noise')
            #t=t.reshape((-1,1))
            mu_list=mu_list+t*Delta
            #print(mu_list,'after noise')
        else:
            pert_err=(np.random.randint(2, size=np.shape(mu_list))*2)-1
            mu_list=mu_list+pert_err*Delta
        #print(mu_list)
        #mu_list=mu_list+np.random.uniform(-1,1,len(x))*Delta
        mu_tup_for_x = list(zip(*mu_list))
        return mu_tup_for_x
    return uncertain_postmean_fn   
def make_uncertain_gp_postmean_fn(model,model2,Delta,para_t=1,policy_complete_perturb=False,gaus_nois=False):
    def uncertain_postmean_fn(x):
        mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)
        #err_dta=Namespace()
        #err_dta.x=x[0]
        #err_dta.y=np.random.normal(0,1,size=[np.shape(mu_list[0])])
        err_list=model2.call_function_sample_list(x)
        print(err_list)
        print(np.array(err_list).shape)
        print(mu_list)
        print(np.array(mu_list).shape)
        mu_list=mu_list+np.array(err_list)*Delta
        #print(Delta)
        #print(mu_list)
        #print(para_t,'para_t')
        #if not policy_complete_perturb and not gaus_nois:
            #print(np.shape(mu_list))
            #print(np.shape(x))
            #Delta=0
            #print(mu_list)
        #    t=np.random.uniform(-1,1,np.shape(mu_list))
            #print(t)
        #    mu_list=mu_list+t*Delta
            #print(mu_list)
        #elif gaus_nois:
            #t=np.random.normal(0,1,np.shape(mu_list))
            #print(t)
        #    t=np.random.normal(0,1,size=(np.shape(mu_list)[0],para_t))
            #print(t,'raw_noise')
        #    t=np.repeat(t,int(np.shape(mu_list)[1]/para_t),axis=0)
        #    t=t.reshape(np.shape(mu_list))
            #print(t,'repeat_noise')
            #print(mu_list,'before noise')
            #t=t.reshape((-1,1))
        #    mu_list=mu_list+t*Delta
            #print(mu_list,'after noise')
        #else:
        #    pert_err=(np.random.randint(2, size=np.shape(mu_list))*2)-1
        #    mu_list=mu_list+pert_err*Delta
        #print(mu_list)
        #mu_list=mu_list+np.random.uniform(-1,1,len(x))*Delta
        mu_tup_for_x = list(zip(*mu_list))
        return mu_tup_for_x
    return uncertain_postmean_fn   
