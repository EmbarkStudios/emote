import pytest

from emote.memory.memory import MemoryProxyWrapper


class DummyMemoryProxy:
    def __init__(self):
        self.my_attribute = "hello!"

    def say_hello(self):
        return "hello world"

    @property
    def a_property(self):
        return "a property"


class EmptyMemoryProxyWrapper(MemoryProxyWrapper):
    pass


class SayHelloMemoryProxyWrapper(MemoryProxyWrapper):
    def __init__(self, inner):
        super().__init__(inner)

    def say_hello(self):
        return "not hello world"


class SayGoodbyeProxyWrapper(MemoryProxyWrapper):
    def __init__(self, inner):
        super().__init__(inner)

    def say_goodbye(self):
        return "goodbye"


def test_call_nonexisting_method_on_wrapper_calls_inner():
    dummy = DummyMemoryProxy()
    wrapper = EmptyMemoryProxyWrapper(dummy)

    assert (
        wrapper.say_hello == dummy.say_hello
    ), "Expected wrapper to forward non-existing method to inner."


def test_call_existing_method_on_wrapper_calls_existing():
    dummy = DummyMemoryProxy()
    wrapper = SayHelloMemoryProxyWrapper(dummy)

    assert (
        wrapper.say_hello == wrapper.say_hello
    ), "Expected wrapper to always use existing method if it exist."


def test_chained_wrappers():
    dummy = DummyMemoryProxy()
    wrapper1 = SayHelloMemoryProxyWrapper(dummy)
    wrapper2 = SayGoodbyeProxyWrapper(wrapper1)
    wrapper3 = EmptyMemoryProxyWrapper(wrapper2)

    assert (
        wrapper3.say_hello == wrapper1.say_hello
    ), "Expected wrapper to be able to chain inner forwards."
    assert (
        wrapper3.say_goodbye == wrapper2.say_goodbye
    ), "Expected wrapper to be able to chain inner forwards."


def test_wrapper_disallows_accessing_non_method():
    dummy = DummyMemoryProxy()
    wrapper = EmptyMemoryProxyWrapper(dummy)

    with pytest.raises(AttributeError):
        wrapper.my_attribute


def test_wrapper_disallows_accessing_non_existing_attribute():
    dummy = DummyMemoryProxy()
    wrapper = EmptyMemoryProxyWrapper(dummy)

    with pytest.raises(AttributeError):
        wrapper.i_do_not_exist


def test_wrapper_allows_accessing_property():
    dummy = DummyMemoryProxy()
    wrapper = EmptyMemoryProxyWrapper(dummy)

    wrapper.a_property


def test_wrapper_allows_accessing_property_nested():
    dummy = DummyMemoryProxy()
    wrapper = EmptyMemoryProxyWrapper(dummy)
    wrapper = SayGoodbyeProxyWrapper(wrapper)

    wrapper.a_property
