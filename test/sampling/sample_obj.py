import os
import qaoa
import argparse

if __name__ == '__main__':

    nvert = 12
    ngraph = 0
    nlayers = 3
    ndegree = 3

    parser = argparse.ArgumentParser(description="Sample QAOA Objective")
    parser.add_argument('-p','--layers',dest='nlayers',type=int, default=1,help='Number of QAOA layers')
    parser.add_argument('-v','--vertices',dest='nvert',type=int, default=12,help='Number of graph vertices')
    parser.add_argument('-g','--graph',dest='ngraph',type=int, default=12,help='Graph instance ID number')
    parser.add_argument('-d','--degree',dest='ndegree',type=int, default=3,help='Degree of random regukar graph')
    parser.add_argument('-j','--jobs',dest='njobs',type=int, default=1,help='Specifies the number of jobs to run simultaneously')
    parser.add_argument('-s','--samples',dest='nsamples',type=int, help='Number of sample points')
    
    args = parser.parse_args()

    obj = qaoa.circuit.load_maxcut(degree=args.ndegree,nvert=args.nvert,graph_num=args.ngraph,nlayers=args.nlayers)
    sampler = qaoa.sampling.ObjectiveSampler(obj)
    results = sampler(args.nsamples,args.njobs,"Objective Value","Gradient Norm","Hessian Eigenvalues")

    filename = "uwmc_d{0}_n{1}_g{2}_p{3}.csv".format(ndegree,nvert,ngraph,nlayers)
    cwd = os.getcwd()
    pathfile = os.path.join(cwd,"results",filename)

    print(results)
    results.to_csv(pathfile)

      

