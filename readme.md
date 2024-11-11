cd .\src\scripts/lib
& C:/Users/abhis/miniconda3/envs/fools_gold/python.exe setup.py bdist_wheel
!pip uninstall misc -y
!pip install ..\src\scripts\lib\dist\misc-0.1.0-py3-none-any.whl --force-reinstall
