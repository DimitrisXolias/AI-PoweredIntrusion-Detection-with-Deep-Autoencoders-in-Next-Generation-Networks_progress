#!/usr/bin/python3

from mininet.topo import Topo # type: ignore
from mininet.net import Mininet # type: ignore
from mininet.node import RemoteController # type: ignore
from mininet.link import TCLink # type: ignore
from mininet.cli import CLI # type: ignore

class ThesisTopo(Topo):
    def build(self):
        # Switches
        s1 = self.addSwitch('s1')  # core
        s2 = self.addSwitch('s2')  # edge 1
        s3 = self.addSwitch('s3')  # edge 2

        # Hosts
        h1 = self.addHost('h1', ip='10.0.0.1')  # Victim
        h2 = self.addHost('h2', ip='10.0.0.2')  # Attacker
        h3 = self.addHost('h3', ip='10.0.0.3')  # Collector/API
        h4 = self.addHost('h4', ip='10.0.0.4')  # Normal User

        # Links
        self.addLink(s1, s2)
        self.addLink(s1, s3)
        self.addLink(s2, h1)
        self.addLink(s2, h3)
        self.addLink(s3, h2)
        self.addLink(s3, h4)

if __name__ == '__main__':
    topo = ThesisTopo()
    net = Mininet(topo=topo, controller=RemoteController, link=TCLink)

    net.start()
    print("✅ Network started. Testing connectivity...")
    net.pingAll()
    CLI(net)
    net.stop()
