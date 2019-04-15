import numpy as np
import cupy as cp
from chainer import optimizer
from chainer import backend
from chainer.backends import cuda

_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.epsilon = 1
_default_hyperparam.epsilon = 0.6
_default_hyperparam.approx = 'second'

class HamiltonianExplicitRule(optimizer.UpdateRule):
    _kernel_x = None
    _kernel_p = None
    _kernel_r = None
    
    def __init__(
        self, parent_hyperparam=None,epsilon = None, delta=None, approx=None, body=None, tail=None, expon=None):
        super(HamiltonianExplicitRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if epsilon is not None:
            self.hyperparam.epsilon = epsilon
        if delta is not None:
            self.hyperparam.delta = delta
        if approx is not None:
            self.hyperparam.approx = approx
            
        def init_state(self, param):
            xp = backend.get_array_module(param.data)
            with cuda.get_device_from_array(param.data):
                self.state['p'] = xp.zeros_like(param.data)
        
        def update_core_cpu(self, param):
            grad = param.grad
            if grad is None:
                return
            hp = self.hyperparam
            p = self.state['p']
            
            if hp.approx == 'first':
                p = hp.delta * p - hp.epsilon * hp.delta * grad
            else:
                if _kernel_r is None:
                    _kernel_r = (2.0 - (1.0/delta))
                else:
                    p = _kernel_r * p - hp.epsilon * grad
                
            sqsum = np.vdot(p,p)
            param.data += hp.epsilon / np.sqrt(1.0 + sqsum) * p
            
        def update_core_gpu(self, param):
            grad = param.grad
            if grad is None:
                return
            hp = self.hyperparam
            p = self.state['p']
            
            if HamiltonianExplicitRule._kernel_x is None:
                HamiltonianExplicitRule._kernel_x = cp.ElementwiseKernel(
                    'T epsilon, T p, T denomp, T param',
                    'T x',
                    'x = param + epsilon * p / denomp',
                    'Hamiltonian_x')
                
            if HamiltonianExplicitRule._kernel_r is None:
                HamiltonianExplicitRule._kernel_r = cp.ReductionKernel(
                    'T p',
                    'T denomp',
                    'p * p',
                    'a + b',
                    'denomp = sqrt(a)',
                    '1',
                    'relativistic')
                    
            if hp.approx == 'first':
                # p
                if HamiltonianExplicitRule._kernel_p is None:
                    HamiltonianExplicitRule._kernel_p = cp.ElementwiseKernel(
                        'T delta, T epsilon, T grad, T p0',
                        'T p1',
                        'p1 = p0 * delta - epsilon * delta * grad',
                        'Hamiltonian_p')
                p = HamiltonianExplicitRule._kernel_p(hp.delta, hp.epsilon, grad, p)
                # x
                denomp = HamiltonianExplicitRule._kernel_r(p)
                param.data = HamiltonianExplicitRule._kernel_x(hp.epsilon, p, denomp, param.data)
            else:
                if HamiltonianExplicitRule._kernel_p is None:
                    HamiltonianExplicitRule._kernel_p = cp.ElementwiseKernel(
                        'T delta, T epsilon, T grad, T p0',
                        'T p1',
                        'p1 = p0 * (2.0 - (1.0 / delta)) - epsilon * grad',
                        'Hamiltonian_p')
                else:
                    # p
                    p = HamiltonianExplicitRule._kernel_p(hp.delta, hp.epsilon, grad, p)
                # x
                denomp = HamiltonianExplicitRule._kernel_r(p)
                param.data = HamiltonianExplicitRule._kernel_x(hp.epsilon, p, denomp, param.data)

class Hamiltonian(optimizer.GradientMethod):
    def __init__(
        self, epsilon=_default_hyperparam.epsilon, delta=_default_hyperparam.delta, approx=_default_hyperparam.approx):
        super(Hamiltonian, self).__init__()
        self.hyperparam.epsilon = epsilon
        self.hyperparam.delta = delta
        self.hyperparam.approx = approx
    
    epsilon = optimizer.HyperparameterProxy('epsilon')
    delta = optimizer.HyperparameterProxy('delta')
    approx = optimizer.HyperparameterProxy('approx')
    
    def create_update_rule(self):
        return HamiltonianExplicitRule(self.hyperparam)