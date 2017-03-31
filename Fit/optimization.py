import re
import numpy as np
import scipy.optimize as opt

class optFit:
    def __init__(self, func=[], params={}, freeList=[]):
        self.func = func # function handle
        self.params = params # initialize dictionary
        self.freeList = freeList # initialize list

    def catch(self, func, val=False, handle=lambda e: e, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return val

    def params2vals(self, params, freeList):
        return([params[i] for i in freeList])

    def vals2params(self, vals, params, freeList):
        return({**params, **{i: j for i,j in zip(freeList,vals)}})

    def fitFunction(self, vals, func, params, freeList):
        params = self.vals2params(vals, params, freeList)
        return(func(**params))

    def fitnon(self, func, params, freeList):
        vals = self.params2vals(params, freeList)
        vals = opt.fmin(func=self.fitFunction, x0=vals, args=(func, params, freeList), 
                        maxfun=1e6, full_output=True, )
        params = self.vals2params(vals[0], params, freeList)
        return(params, vals[1])

    def params2valscon(self, params, freeList):
        freeList = [re.sub('[= ]','',i) for i in freeList] # remove blanks and '='
        tmp = [re.compile('(<|>)').split(i) for i in freeList]
        lb = [self.catch(lambda: float(y), float('-inf')) for y in [float('-inf') if len(x)==1 
              else x[0] if x[1]=="<" else x[2] if (len(x)==3 and x[1]==">") else x[4] 
              if (len(x)==5 and x[3]==">") else float('-inf') for x in tmp]] # lower bound
        ub = [self.catch(lambda: float(y), float('inf')) for y in [float('inf') if len(x)==1 
              else x[0] if x[1]==">" else x[2] if (len(x)==3 and x[1]=="<") 
              else x[4] if (len(x)==5 and x[3]=="<") else float('inf') for x in tmp]] # upper bound
        indx = [[isinstance(self.catch(lambda: float(re.sub('[<>]','42',y)),True),bool) 
                 for y in x] for x in tmp]
        varList = [i[j] for i,j in zip(tmp, [int(y[0]) for y in [np.where(x) for x in indx]])]
        vals = [params[i] for i in varList] # values
        return(vals, {x: [y,z] for x,y,z in zip(varList,lb,ub)}, varList)

    def bndcon(self, vals, bounds, freeList):
        indx = [x > bounds[y][0] and x < bounds[y][1] for x,y in zip(vals,freeList)]
        tmp = [bounds[y] if not x else float('nan') for x,y in zip(indx,freeList)]
        tmp = [min(x) if (np.isinf(np.mean(x)) and np.mean(x) > 0)
               else max(x) if (np.isinf(np.mean(x)) and np.mean(x) < 0) 
               else np.mean(x) for x in tmp]
        return([y if np.isnan(x) else x for x,y in zip(tmp,vals)])

    def fitFunctionCon(self, vals, func, params, bounds, freeList): 
        params = self.vals2params(vals, params, freeList)
        err = func(**params) + sum([0 if params[i] >= bounds[i][0] and params[i] <= bounds[i][1] 
                                    else float('inf') for i in freeList])
        return(err)

    def fitcon(self, func, params, freeList):
        [vals,bounds,varList] = self.params2valscon(params, freeList)
        vals = self.bndcon(vals, bounds, varList)
        vals = opt.fmin(func=self.fitFunctionCon, x0=vals, args=(func, params, bounds, varList),
                        maxfun=1e6, full_output=True)
        params = self.vals2params(vals[0], params, varList)
        return(params, vals[1])

    def fit(self, func=None, params=None, freeList=None):
        args = {**self.__dict__, **{i: j for i,j in 
                zip(self.fit.__code__.co_varnames,[self,func,params,freeList])}}
        args = {i: self.__dict__[i] if args[i]==None else args[i] for i in self.__dict__}
        if any([re.search('[<>]',i)!=None for i in args['freeList']]):
            return(self.fitcon(**args))
        else:
            return(self.fitnon(**args))