class SubstrateLink:

    def __init__(self, index, bw, onode, tnode):
        self.bandwidth = bw
        self.totalBandwidth = bw
        self.onode = onode  # from changed to origin
        self.tnode = tnode
        self.index = index
        self.virtual_link_set = set()
