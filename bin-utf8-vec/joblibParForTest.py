__author__ = "Laurence Elliott"

from joblib import Parallel, delayed

def returnArg(arg):
    print(arg)
    return arg

argList = [1,2,3,4,5,6,7,8,9,10]

returnArgResults = Parallel(n_jobs=4, verbose=1, backend="threading")(map(delayed(returnArg), argList))
print(returnArgResults)