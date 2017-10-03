class Registry:
    def __init__(self):
        self.listing = {}

    def __getitem__(self, name):
        return self.listing[name]

    def all(self):
        return self.listing

    def register_to(self, obj, name=None):
        '''Register `obj` under `name`.'''
        name = name or obj.__name__
        assert name not in self.listing, '%s is already registered by %r' % (name, self.listing[name])
        # Add to listing
        self.listing[name] = obj
        return obj

    def register(self, name):
        '''Use as a decorator'''
        # Handle if decorator was used without parentheses
        if callable(name):
            obj = name
            return self.register_to(obj, name=None)

        return lambda obj: self.register_to(obj, name)

# Global registries
env_registry = Registry()
optimizer_registry = Registry()
# model_registry = Registry()
