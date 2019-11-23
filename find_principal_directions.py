import numpy as np
import sklearn.svm


method = {
    0:False,
    1:True,
    2:True,
    3:True,
    4:True,
    10:True,
    11:True,
    17:True,
    19:True,
}

for attrib_idx in range(19, 20):
    try:
        results = np.load("wspace_att_%d.npy" % attrib_idx).item()

        pruned_indices = list(range(results['latents'].shape[0]))
        pruned_indices = sorted(pruned_indices, key=lambda i: -np.max(results[attrib_idx][i]))
        keep = int(results['latents'].shape[0] * 0.5)
        print('Keeping: %d' % keep)
        pruned_indices = pruned_indices[:keep]

        # Fit SVM to the remaining samples.
        svm_targets = np.argmax(results[attrib_idx][pruned_indices], axis=1)
        space = 'dlatents'

        svm_inputs = results[space][pruned_indices]

        if method[attrib_idx]:
            svm = sklearn.svm.LinearSVC(C=0.1)
            svm.fit(svm_inputs, svm_targets)
            svm.score(svm_inputs, svm_targets)
            svm_outputs = svm.predict(svm_inputs)

            w = svm.coef_[0]

            np.save("direction_%d" % attrib_idx, w)

        else:
            c1 = (svm_inputs * svm_targets[..., None]).mean(axis=0)
            c2 = (svm_inputs * (1.0 - svm_targets[..., None])).mean(axis=0)

            w = c1 - c2
            w = w / np.sqrt((w*w).sum())

            np.save("direction_%d" % attrib_idx, w)
    except:
        pass
