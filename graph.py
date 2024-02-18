from sn import SubstrateNode
from sl import SubstrateLink
import numpy as np


class Graph:
    def __init__(self, path):
        file = open(path)
        lsp = file.readline().split(" ")
        # 读取node和link的数量
        self.num_nodes = int(lsp[0])
        self.num_links = int(lsp[1])
        self.substrate_nodes = []
        self.substrate_links = []
        self.link_dict = {}
        self.connection = np.zeros([self.num_nodes, self.num_nodes])
        self.floyd = []

        # 构建node
        for i in range(self.num_nodes):
            lsp = file.readline().split(" ")
            cpu = float(lsp[2])
            x = int(lsp[0])
            y = int(lsp[1])
            node = SubstrateNode(i, cpu, x, y)
            self.substrate_nodes.append(node)
        # 构建link
        for i in range(self.num_links):
            lsp = file.readline().split(" ")
            bandwidth = float(lsp[2]) / 5 * 3 - 10
            origin = int(lsp[0])
            to = int(lsp[1])
            onode = self.substrate_nodes[origin]
            tnode = self.substrate_nodes[to]
            link = SubstrateLink(i, bandwidth, onode, tnode)
            self.substrate_links.append(link)
            self.connection[origin][to] = bandwidth
            self.connection[to][origin] = bandwidth
            self.link_dict[origin * 100 + to] = link
            self.link_dict[to * 100 + origin] = link
            onode.link_set.add(link)
            tnode.link_set.add(link)
        file.close()
        # 构建floyd矩阵求最短路径
        with open('floyd.txt') as file:
            for i in range(self.num_nodes):
                line = file.readline().strip()
                l = list(map(int, line.split(' ')))
                self.floyd.append(l)
