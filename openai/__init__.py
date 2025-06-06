from importlib import import_module
module = import_module('vendor.openai')
for attr in module.__dict__:
    if not attr.startswith('__'):
        globals()[attr] = module.__dict__[attr]
