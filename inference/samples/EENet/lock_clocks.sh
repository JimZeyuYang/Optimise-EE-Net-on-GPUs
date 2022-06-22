sudo cpupower frequency-set -g userspace
sudo cpupower frequency-set -f 3.7GHz

sudo nvidia-smi --lock-memory-clocks=9501
sudo nvidia-smi --lock-gpu-clocks=2130

