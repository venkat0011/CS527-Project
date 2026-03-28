# shell script to install lean
sudo apt install git curl
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y
source $HOME/.elan/env
lean --version
lake --version
git config --global user.email "venkats3@illinois.edu"
git config --global user.name "venkat"