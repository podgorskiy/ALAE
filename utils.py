from torch import nn
import torch
import threading


def save_model(x, name):
    if isinstance(x, nn.DataParallel):
        torch.save(x.module.state_dict(), name)
    else:
        torch.save(x.state_dict(), name)


class AsyncCall(object):
    def __init__(self, fnc, callback=None):
        self.Callable = fnc
        self.Callback = callback
        self.result = None

    def __call__(self, *args, **kwargs):
        self.Thread = threading.Thread(target=self.run, name=self.Callable.__name__, args=args, kwargs=kwargs)
        self.Thread.start()
        return self

    def wait(self, timeout=None):
        self.Thread.join(timeout)
        if self.Thread.isAlive():
            raise TimeoutError
        else:
            return self.result

    def run(self, *args, **kwargs):
        self.result = self.Callable(*args, **kwargs)
        if self.Callback:
            self.Callback(self.result)


class AsyncMethod(object):
    def __init__(self, fnc, callback=None):
        self.Callable = fnc
        self.Callback = callback

    def __call__(self, *args, **kwargs):
        return AsyncCall(self.Callable, self.Callback)(*args, **kwargs)


def async_func(fnc=None, callback=None):
    if fnc is None:
        def add_async_callback(f):
            return AsyncMethod(f, callback)
        return add_async_callback
    else:
        return AsyncMethod(fnc, callback)
