from sn import sncmp
from topology import bfs
import rl


class VirtualRequest:
    def __init__(self, index, virtual_nodes, virtual_links, t_begin, t_end):
        self.index = index
        self.virtual_nodes = virtual_nodes
        self.virtual_links = virtual_links
        self.t_begin = t_begin
        self.t_end = t_end
        self.alpha = 0.0
        self.revenue = self.get_revenue()
        self.mapped = False
        self.hosts = []

    def get_revenue(self):
        rev = 0.0
        for e in self.virtual_links:
            rev += e.bandwidth
        for n in self.virtual_nodes:
            rev += self.alpha * n.cpu
        return rev * (self.t_end - self.t_begin)

    def node_map(self, network):
        status = True
        for vn in self.virtual_nodes:
            if status:
                status = False
                network.substrate_nodes.sort(key=sncmp)
                for sn in network.substrate_nodes:
                    if sn not in self.hosts and sn.cpu >= vn.cpu:
                        vn.map(sn)
                        status = True
                        self.hosts.append(sn)
                        break
            else:
                # print("node map failed:" + str(self.index))
                return False
        # if not status:
            # print("node map failed:" + str(self.index))
        return status

    def node_map_rl(self, network, model, test=False):
        if not test:
            counter = 0
            for vn in self.virtual_nodes:
                sn = model.choose_node(network, vn, self)
                if sn:
                    vn.map(sn)
                    counter += 1
                    self.hosts.append(sn)
                else:
                    model.del_grad(counter)
                    return False
            return True
        else:
            for vn in self.virtual_nodes:
                sn = model.choose_node_test(network, vn, self)
                if sn:
                    vn.map(sn)
                    self.hosts.append(sn)
                else:
                    return False
            return True

    def link_map(self, network):
        for link in self.virtual_links:
            origin = link.onode.host.index
            to = link.tnode.host.index
            route = bfs(network.connection, origin, to, link.bandwidth)
            if route is not None:
                link.map(route, network)
            else:
                # print("link failed:" + str(self.index))
                return False
        return True

    def unmap(self, network):
        for n in self.virtual_nodes:
            n.unmap()
        for l in self.virtual_links:
            l.unmap(network)
        self.hosts = []


def reqcmp(r): return r.revenue

