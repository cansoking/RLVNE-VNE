from graph import Graph
from vn import vncmp, VirtualNode
from vl import VirtualLink
import numpy as np


def build_virtual_request(index, path):
    from req import VirtualRequest
    file_path = path + '/req' + str(index) + '.txt'
    file = open(file_path)
    lsp = file.readline().split(" ")
    num_nodes = int(lsp[0])
    num_links = int(lsp[1])
    t_begin = int(lsp[3])
    t_end = int(lsp[4]) + t_begin
    vns = []
    vls = []
    for i in range(num_nodes):
        lsp = file.readline().split(" ")
        cpu, x, y = float(lsp[2]) * 2.5, float(lsp[0]), float(lsp[1])
        node = VirtualNode(cpu, x, y, i)
        vns.append(node)
    for i in range(num_links):
        lsp = file.readline().split(" ")
        bw, origin, to = float(lsp[2]), int(lsp[0]), int(lsp[1])
        vns[origin].neighbors.add(vns[to])
        vns[to].neighbors.add(vns[origin])
        link = VirtualLink(bw, vns[origin], vns[to], i)
        vns[origin].link_set.add(link)
        vns[to].link_set.add(link)
        vls.append(link)
    vns.sort(key=vncmp)
    file.close()
    return VirtualRequest(index, vns, vls, t_begin, t_end)

def build_substrate_network(filepath):
    g = Graph(filepath)
    return g


def bfs(connection, origin, to, bandwidth):
    num_nodes = len(connection)
    trace = -np.ones(num_nodes, dtype="int64")
    old = set()
    new = set()
    for index, bw in enumerate(connection[origin]):
        if bw >= bandwidth:
            trace[index] = origin
            if index == to:
                result = []
                while to != origin:
                    result.append(to)
                    to = trace[to]
                result.append(origin)
                return result
            new.add(index)
            old.add(index)
    while len(new) > 0:
        next_new = set()
        for row in new:
            for index, bw in enumerate(connection[row]):
                if index not in old and bw >= bandwidth:
                    trace[index] = row
                    if index == to:
                        result = []
                        while to != origin:
                            result.append(to)
                            to = trace[to]
                        result.append(origin)
                        return result
                    old.add(index)
                    next_new.add(index)
        new = next_new
    return None
