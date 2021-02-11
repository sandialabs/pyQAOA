if __name__ == '__main__':

    import qaoa

    obj = qaoa.circuit.load_maxcut(degree=3,nvert=12,graph_num=0,nlayers=3)
    sampler = qaoa.sampling.ObjectiveSampler(obj)

    results = sampler(10,"Objective Value","Gradient Norm")

    print(results)

      

