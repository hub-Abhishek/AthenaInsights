bash
conda env create -f environment.yaml

conda activate fools_gold

cd .\src\scripts/lib
python setup.py bdist_wheel

pip uninstall misc -y

pip install ..\src\scripts\lib\dist\misc-0.1.0-py3-none-any.whl --force-reinstall

python -m ipykernel install --user --name fools_gold --display-name  fools_gold