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

import numpy as np
import sklearn.svm
import multiprocessing as mp

indices = [0, 1, 2, 3, 4, 10, 11, 17, 19]


def run(attrib_idx):
    results = np.load("principal_directions/wspace_att_%d.npy" % attrib_idx).item()

    pruned_indices = list(range(results['latents'].shape[0]))
    # pruned_indices = sorted(pruned_indices, key=lambda i: -np.max(results[attrib_idx][i]))
    # keep = int(results['latents'].shape[0] * 0.95)
    # print('Keeping: %d' % keep)
    # pruned_indices = pruned_indices[:keep]

    # Fit SVM to the remaining samples.
    svm_targets = np.argmax(results[attrib_idx][pruned_indices], axis=1)
    space = 'dlatents'

    svm_inputs = results[space][pruned_indices]

    svm = sklearn.svm.LinearSVC(C=1.0, dual=False, max_iter=10000)
    svm.fit(svm_inputs, svm_targets)
    svm.score(svm_inputs, svm_targets)
    svm_outputs = svm.predict(svm_inputs)

    w = svm.coef_[0]

    np.save("principal_directions/direction_%d" % attrib_idx, w)


p = mp.Pool(processes=4)
p.map(run, indices)
