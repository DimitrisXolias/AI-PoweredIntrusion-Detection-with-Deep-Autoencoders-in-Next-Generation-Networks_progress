
#!/usr/bin/python3

from mininet.topo import Topo  # type: ignore
from mininet.net import Mininet  # type: ignore
from mininet.node import RemoteController, OVSSwitch  # type: ignore # use RemoteController!
from mininet.cli import CLI  # type: ignore
from mininet.log import setLogLevel  # type: ignore

class IDSTopology(Topo):
    def build(self):
        # Hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        h3 = self.addHost('h3', ip='10.0.0.3/24')

        # Switch (named r1)
        r1 = self.addSwitch('r1')

        # Links
        self.addLink(h1, r1)
        self.addLink(h2, r1)
        self.addLink(h3, r1)

# Define custom switch that uses OpenFlow13
class MySwitch(OVSSwitch):
    def start(self, controllers):
        super(MySwitch, self).start(controllers)
        self.cmd("ovs-vsctl set bridge %s protocols=OpenFlow13" % self.name)

if __name__ == '__main__':
    setLogLevel('info')
    topo = IDSTopology()

    # Connect to Ryu controller on localhost:6653
    c0 = RemoteController('c0', ip='127.0.0.1', port=6653)

    # Use custom switch that enforces OpenFlow13
    net = Mininet(topo=topo,
                  controller=c0,
                  switch=MySwitch)

    net.start()
    CLI(net)
    net.stop()
