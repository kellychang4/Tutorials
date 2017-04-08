import numpy as np
import scipy.optimize as opt

class fit:
    def __init__(self, func=[], params={}, freeList={}, mask={}):
        self.func = func # function handle
        self.params = params # initialize params dictionary
        self.freeList = freeList # initialize freeList dictionary 
        self.mask = mask # initialize mask dictionary
        
    def __repr__(self):
        return("<class fit func: %s params: %s freeList: %s mask: %s>" % 
               (self.func, self.params, self.freeList, self.mask))
        
    def __str__(self):
        return("\n\tfunc: %s\n\n\tparams: %s\n\n\tfreeList: %s\n\n\tmask: %s" % 
               (self.func.__name__, self.params, self.freeList, self.mask))

    def editcon(self, params, freeList):
        try:
            unconList = {i: [np.tile(-np.inf, np.shape(params[i])), np.tile(np.inf, np.shape(params[i]))] 
                             for i in freeList.keys() if (freeList[i] is 'None')} 
            scaleList = {i: [np.array(freeList[i][0]),np.array(freeList[i][1])] for i in freeList.keys() 
                         if ((freeList[i] is not 'None') and (np.isscalar(params[i])))}
            fixScaleList = {i: [np.tile(freeList[i][0], np.shape(params[i])), np.tile(freeList[i][1], 
                            np.shape(params[i]))] for i in freeList.keys() if ((freeList[i] is not 'None') 
                            and (not np.isscalar(params[i])) and (np.shape(freeList[i]) == (2,)))}

            editList = {**freeList, **{**scaleList, **fixScaleList, **unconList}}
            params = {**params, **{i: np.array(params[i]) for i in params.keys() if not
                                   isinstance(params[i], np.ndarray)}}

            if not all([np.shape(editList[i]) == ((2,) + np.shape(params[i])) for i in editList.keys()]):
                raise ValueError('Must specify lower and upper bound arrays of equivalent shape as '
                                 'parameters values for free parameter(s): %s' % 
                                 [i for i in editList.keys() if not (np.shape(editList[i]) == 
                                  ((2,) + np.shape(params[i])))])
                
            return(params, editList)
        except ValueError as e:
            raise e
            
    def fixcon(self, freeList, mask):
        try:
            mask = {i: np.array(mask[i], dtype=bool) for i in mask.keys()}
            if not all([np.shape(mask[i]) == (np.shape(freeList[i][0])) for i in mask.keys()]):
                raise ValueError('Must specify mask array of equivalent shape as parameters values for ' 
                                 'free parameter(s): %s' % [i for i in mask.keys() if not 
                                  (np.shape(mask[i]) == np.shape(freeList[i][0]))])
            for key in mask.keys():
                freeList[key][0][mask[key]] = np.nan
                freeList[key][1][mask[key]] = np.nan
            return(freeList)        
        except ValueError as e:
            raise e

    def bndcon(self, params, freeList):
        p = {i: params[i].flatten() for i in freeList.keys()}
        f = {i: [freeList[i][0].flatten(), freeList[i][1].flatten()] for i in freeList.keys()}
        m = {i: np.mean(f[i], axis=0) for i in f.keys()}
        return({x: np.reshape(np.array([f[x][0][i] if (np.isposinf(m[x][i]) and (p[x][i] < f[x][0][i]))
                  else f[x][1][i] if (np.isneginf(m[x][i]) and (p[x][i] > f[x][1][i]))
                  else p[x][i] if np.isnan(m[x][i]) else m[x][i] for i in np.arange(np.size(m[x]))]),
                  np.shape(params[x])) for x in f.keys()})

    def params2vals(self, params, freeList, mask):
        [params,freeList] = self.editcon(params, freeList)
        freeList = self.fixcon(freeList, mask)
        params = self.bndcon(params, freeList)
        vals = np.array([x for i in freeList.keys() for x,y in zip(params[i].flatten(), 
                         freeList[i][0].flatten()) if not np.isnan(y)])
        return(vals, params, freeList) 

    def vals2params(self, vals, params, freeList):
        nVals = [np.sum(~np.isnan(freeList[i][0])) for i in freeList.keys()] 
        vals = {z: vals[np.arange(x,y)] for x,y,z in zip(np.cumsum(nVals)-
                    nVals,np.cumsum(nVals),freeList.keys())}
        indx = [np.argwhere(np.isnan(freeList[j][0].flatten())).flatten() for j in vals.keys()]
        fixed = {j: params[j].flatten()[i] for i,j in zip(indx,vals.keys())}
        updates = {j: np.insert(vals[j],i-np.arange(0,np.size(i)),fixed[j]).reshape(np.shape(params[j]))
                  for i,j in zip(indx, vals.keys())}
        return({**params, **updates})

    def nancon(self, params, freeList):
        indx = ~np.isnan(freeList[0])
        return(any(~np.logical_and(params[indx] >= freeList[0][indx],
                                  params[indx] <= freeList[1][indx])))

    def fitFunction(self, vals, func, params, freeList): 
        params = self.vals2params(vals, params, freeList)
        err = func(**params) + np.sum([np.inf if self.nancon(params[i],freeList[i])
                                       else 0 for i in freeList.keys()])
        return(err)

    def fitcon(self, func, params, freeList, mask):
        [vals,params,freeList] = self.params2vals(params, freeList, mask)
        out = opt.fmin(func=self.fitFunction, x0=vals, args=(func, params, freeList),
                       maxfun=1e6, full_output=True)
        params = self.vals2params(out[0], params, freeList)
        return(params, out[1])

    def fit(self, func=[], params={}, freeList={}, mask={}):
        args = {'func': func, 'params': params, 'freeList': freeList, 'mask': mask}
        args = {i: self.__dict__[i] if not args[i] else args[i] for i in args.keys()} 
        return(self.fitcon(**args))