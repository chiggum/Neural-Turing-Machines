import theano.tensor as T
import numpy as np
import theano


class rmsprop(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.running_square_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.running_avg_ = [theano.shared(np.zeros_like(p.get_value()))
                             for p in params]
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum, minimum_grad = 1e-4, rescale=5.):
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.95
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates
    
    def graves_rmsprop_updates(self, params, grads, learning_rate=1e-4, alpha=0.9, epsilon=1e-4, chi=0.95):
        """
        Alex Graves' RMSProp [1]_.
        .. math ::
            n_{i} &= \chi * n_i-1 + (1 - \chi) * grad^{2}\\
            g_{i} &= \chi * g_i-1 + (1 - \chi) * grad\\
            \Delta_{i} &= \alpha * Delta_{i-1} - learning_rate * grad /
                    sqrt(n_{i} - g_{i}^{2} + \epsilon)\\
            w_{i} &= w_{i-1} + \Delta_{i}
        References
        ----------
        .. [1] Graves, Alex.
            "Generating Sequences With Recurrent Neural Networks", p.23
            arXiv:1308.0850
        """
        updates = []
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = T.switch(not_finite, 0.1 * param, grad)
            old_square = self.running_square_[n]
            old_avg = self.running_avg_[n]
            old_memory = self.memory_[n]
            new_square = chi * old_square + (1. - chi) * grad ** 2
            new_avg = chi * old_avg + (1. - chi) * grad
            new_memory = alpha * old_memory - learning_rate * grad / T.sqrt(new_square - \
                        new_avg ** 2 + epsilon)
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((old_memory, new_memory))
            updates.append((param, param + new_memory))
        return updates
    
    def updates_v2(self, params, grads, learning_rate, momentum):
        combination_coeff = 0.95
        minimum_grad = 1E-4
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = T.sqrt(new_square - new_avg ** 2 + minimum_grad)
            #rms_grad = T.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates