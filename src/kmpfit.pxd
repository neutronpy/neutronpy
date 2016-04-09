cdef extern from "mpfit.h":

    ctypedef struct mp_par:
        int fixed
        int limited[2]
        double limits[2]
        char *parname
        double step
        double relstep
        int side
        int deriv_debug
        double deriv_reltol
        double deriv_abstol

    ctypedef struct mp_config:
        double ftol
        double xtol
        double gtol
        double epsfcn
        double stepfactor
        double covtol
        int maxiter
        int maxfev
        int douserscale
        int nofinitecheck

    ctypedef struct mp_result:
        double bestnorm
        double orignorm
        int niter
        int nfev
        int status
        int npar
        int nfree
        int npegged
        int nfunc
        double *resid
        double *xerror
        double *covar
        char version[20]

    ctypedef int (*mp_func)(int *m, int n, double *x, double **fvec, double **dvec, void *private_data)

    cdef int mpfit(mp_func funct, int npar, double *xall, mp_par *pars, mp_config *config, void *private_data, mp_result *result)
