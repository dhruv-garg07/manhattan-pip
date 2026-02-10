
import pkgutil
import mcp
import mcp.types
import inspect

def list_submodules(module):
    for loader, module_name, is_pkg in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
        print(module_name)
        try:
            mod = __import__(module_name, fromlist=['*'])
            for name in dir(mod):
                if "Message" in name:
                    print(f"  Found {name} in {module_name}")
        except Exception as e:
            print(f"  Error importing {module_name}: {e}")

list_submodules(mcp)
