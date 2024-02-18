class VirtualLink:

    def __init__(self, bw, onode, tnode, index):
        self.bandwidth = bw
        self.onode = onode
        self.tnode = tnode
        self.index = index
        self.host = None
        self.cost = 0.0

    def map(self, route, network):
        sls = []
        for i in range(len(route) - 1):
            key = route[i] * 100 + route[i+1]
            sls.append(network.link_dict[key])
            network.connection[route[i]][route[i+1]] -= self.bandwidth
            network.connection[route[i+1]][route[i]] -= self.bandwidth
        self.host = sls 
        cost = self.bandwidth * len(sls)
        for link in self.host:
            link.bandwidth -= self.bandwidth
            link.virtual_link_set.add(self)
        self.cost = cost            #TODO:cost return on map

    def unmap(self, network):
        if (self.host == None):
            return
        #self.cost = 0.0
        for link in self.host:
            link.bandwidth += self.bandwidth
            link.virtual_link_set.remove(self)
            network.connection[link.onode.index][link.tnode.index] += self.bandwidth
            network.connection[link.tnode.index][link.onode.index] += self.bandwidth
        self.host = None
