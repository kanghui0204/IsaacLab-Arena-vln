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
