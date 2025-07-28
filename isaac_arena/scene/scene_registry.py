import importlib
import inspect
import pkgutil
import random
from typing import Any

from isaac_arena.scene.asset import Asset


class SingletonMeta(type):
    """
    Metaclass that overrides __call__ so that only one instance
    of any class using it is ever created.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # first time: actually create the instance
            cls._instances[cls] = super().__call__(*args, **kwargs)
        # afterwards: always return the same object
        return cls._instances[cls]


class ObjectRegistry(metaclass=SingletonMeta):

    def __init__(self, base_packages=None):
        self.base_packages = base_packages or ["isaac_arena.scene"]
        self.registry = {}
        self._auto_register()

    def _auto_register(self):
        """
        Walk through all submodules of each base package,
        import them, and register any Asset subclasses found.
        """
        for pkg_name in self.base_packages:
            pkg = importlib.import_module(pkg_name)
            # pkg.__path__ is only present if pkg is a package
            if not hasattr(pkg, "__path__"):
                continue

            # Go through all submodules of the base package
            for finder, submod_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
                try:
                    module = importlib.import_module(submod_name)
                except ImportError:
                    # Ignore modules that fail to import
                    continue
                self._register_from_module(module)

    def _register_from_module(self, module):
        """
        Inspect a module for any Asset subclasses (except Asset itself),
        instantiate them, and register under their `get_name()`.
        """
        for _, cls in inspect.getmembers(module, inspect.isclass):
            # must be in this module, subclass Asset, but not Asset itself
            if issubclass(cls, Asset) and cls is not Asset and cls.__module__ == module.__name__:
                # skip if __init__ has required (non-default) args beyond self
                # This is to avoid registering classes that have required arguments
                # beyond self, which are not needed for instantiation
                # All our assets have no required arguments beyond self for now.
                sig = inspect.signature(cls.__init__)
                # drop 'self'
                params = list(sig.parameters.values())[1:]
                # check if any param is positional/keyword without a default
                needs_arg = any(
                    p.default == inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for p in params
                )
                if needs_arg:
                    continue

                instance = cls()
                self.register_object(instance.get_name(), instance)

    def register_object(self, name: str, object):
        assert name not in self.registry, f"Object {name} already registered"
        self.registry[name] = object

    def get_object_by_name(self, name) -> Any:
        return self.registry[name]

    def get_objects_by_tag(self, tag) -> list[Any]:
        return [object for object in self.registry.values() if tag in object.tags]

    def get_random_object_by_tag(self, tag) -> Any:
        return random.choice(self.get_objects_by_tag(tag))
