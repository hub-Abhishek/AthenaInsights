bash
conda env create -f environment.yaml

conda activate fools_gold

conda install -n base conda-libmamba-solver

conda config --set solver libmamba

cd src/scripts/lib

pip uninstall misc -y

python setup.py bdist_wheel

pip install dist/misc-0.1.0-py3-none-any.whl --force-reinstall

cd ../../..

python -m ipykernel install --user --name fools_gold --display-name  fools_gold