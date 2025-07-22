#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: ./run.sh file.pcap"
  exit 1
fi

java -cp "CICFlowMeter.jar:jnetpcap-1.4.r1425.jar" cic.flowmeter.Main input_output "$1"
