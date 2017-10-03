from registry import Registry

reg = Registry()

def test_decorator_without_arguments():
    @reg.register
    class Env1:
        pass
    assert reg[Env1.__name__] == Env1, 'The entry should be registered under its default name %s' % Env1.__name__

def test_decorator_with_arguments():
    custom_name = 'some_name'
    @reg.register(custom_name)
    class Env2:
        pass
    assert reg[custom_name] == Env2, 'The entry should be registered under the specified %s' % custom_name
