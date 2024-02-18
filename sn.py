class SubstrateNode:

    def __init__(self, index, cpu, x, y):
        self.index = index
        self.totalCPU = cpu
        self.cpu = cpu
        self.x = x
        self.y = y
        self.link_set = set()
        self.vn_set = set()

    def get_available_resources(self):
        link_sum = 0
        for link in self.link_set:
            link_sum += link.bandwidth
        return self.cpu * link_sum


def sncmp(n): return -n.get_available_resources()
