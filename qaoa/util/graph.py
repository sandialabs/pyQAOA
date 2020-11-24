def is_graph(G):
    if hasattr(G,'edges'):
        from networkx.classes.reportviews import EdgeView
        if isinstance(G.edges,EdgeView):
            return True
    else:
        return False

def is_weighted(G):
    return all( "weight" in G[e[0]][e[1]] for e in G.edges )

def graph_edges(G):

    if is_graph(G):
        G = [(u,v,G[u,v]) for u,v in G.edges] if is_weighted(G) else [(u,v) for u,v in G.edges]
    return G

def load(degree,nvert,graph_num=0):
    """
    Retrieves a previously generated random regular graph as a networkx.OrderedGraph

    Parameters
    ----------
    degree - The degree of the graph, or the number of edges connected to each vertex. 
             Valid values: 3,4,5,6
    nvert - Number of vertices in the graph. Valid values are 4 through 24. Note that 
            degree must be strictly less than nvert and degree*nvert must be even

    Returns
    -------
    networkx.OrderedGraph
    """ 
    import pkg_resources
    import json
    filename = pkg_resources.resource_filename('qaoa','data/connected_regular_graphs.json')
    with open(filename,"r") as jsonfile:
        graph_dict = json.load(jsonfile)

    key = "({0},{1})".format(degree,nvert)
    if key in graph_dict.keys():
        graphs = graph_dict[key]
        max_gn = len(graphs)
        if graph_num <= max_gn:
            from networkx import OrderedGraph
            G = OrderedGraph()
            G.add_edges_from(graphs[graph_num])
            return G
        else:
            raise ValueError("Graph number {0} exceeds maximum index {1}".format(graph_num,max_gn))
    else:
        raise KeyError("No graph of degree {0} with {1} vertices available".format(degree,nvert))
