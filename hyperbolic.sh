wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

sudo apt-get install libgl1 -y
sudo apt-get install cmake -y

uv venv
source .venv/bin/activate
uv sync

sudo apt install git-lfs -y
sudo apt install htop -y
sudo apt install tmux -y

git config --global user.email "testdummyvt@gmail.com"
git config --global user.name "testdummyvt"

CDIR=$(pwd)
export PYTHONPATH=$CDIR:$PYTHONPATH
