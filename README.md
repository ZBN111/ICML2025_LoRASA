## Create Python env
Create a virtual env, e.g. with conda. Use python 3.8.
```
conda create --name <env_name> python=3.8
conda activate <env_name>
```
Install required packages
```
pip install pip==24.0 setuptools==45.2.0 wheel==0.34.2
pip install -r requirements.txt
```

## Install SC2
Install sc2
```
./install_sc2.sh
```
Set environment variable for SC2PATH. By default SC2PATH is set to \`pwd\`'/StarCraftII'

```
export SC2PATH=<path to StarCraftII>
```

## Install MAMuJoCo
`.mujoco/` is in the folder. Move it to `$HOME`
```
mv .mujoco $HOME
```

Set environment variables
```
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt
```

## Usage
Add `pwd` to `PYTHONPATH`.
```
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```
Run an experiment:
```
./run_scripts/MUJOCO/ant.sh
```
## Troubleshooting
If encounter this error when running an MuJoCo experiment:

`gym.error.DependencyNotInstalled: /home/lovicii/anaconda3/envs/mappo6.0/bin/../lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1). (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)`

This can be solved by (if GLIBCXX_3.4.30 is in `/usr/lib/x86_64-linux-gnu/libstdc++.so.6`):
```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/$USER/anaconda3/envs/<env_name>/bin/../lib/libstdc++.so.6
```