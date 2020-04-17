# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from torch import nn
import torch
import threading
import hashlib
import pickle
import os


class cache:
    def __init__(self, function):
        self.function = function
        self.pickle_name = self.function.__name__

    def __call__(self, *args, **kwargs):
        m = hashlib.sha256()
        m.update(pickle.dumps((self.function.__name__, args, frozenset(kwargs.items()))))
        output_path = os.path.join('.cache', "%s_%s" % (m.hexdigest(), self.pickle_name))
        try:
            with open(output_path, 'rb') as f:
                data = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError):
            data = self.function(*args, **kwargs)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        return data


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


class Registry(dict):
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name):
        def register_fn(module):
            assert module_name not in self
            self[module_name] = module
            return module
        return register_fn
