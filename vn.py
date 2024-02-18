class VirtualNode:

    def __init__(self, cpu, x, y, index):
        self.cpu = cpu
        self.x = x
        self.y = y
        self.index = index
        self.host = None
        self.link_set = set()
        self.neighbors = set()
        # self.req = None

    def map(self, node):
        self.host = node
        node.cpu -= self.cpu
        node.vn_set.add(self)

    def unmap(self):
        if (self.host == None):
            return
        else:
            self.host.vn_set.remove(self)
            self.host.cpu += self.cpu
            self.host = None

    def getAvailableResources(self):
        linkSum = 0
        for link in self.link_set:
            linkSum += link.bandwidth
        self.resources = self.cpu * linkSum


def vncmp(n): return -n.cpu
