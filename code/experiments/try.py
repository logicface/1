import importlib.util

def check_module(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"Module {module_name} is not available.")
    else:
        print(f"Module {module_name} is available.")

modules_to_check = [
    'numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn', 'tensorflow', 'torch'
]

for module in modules_to_check:
    check_module(module)