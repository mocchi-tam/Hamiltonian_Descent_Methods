import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from network import MLP, NNet
import Hamiltonian

def main():
    parser = argparse.ArgumentParser(description='Hamiltonian Descent Methods')
    parser.add_argument('--batchsize', '-b', type=int, default=100)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--gpu', '-g', type=int, default=-1, choices=[-1,0,1,2,3])
    parser.add_argument('--out', '-o', type=str, default='verification/')
    parser.add_argument('--data', '-d', type=str, default='mnist', choices=['mnist','cifar10'])
    parser.add_argument('--method', '-m', type=str, default='sem', choices=['adam','sgd','fem','sem'])
    args = parser.parse_args()
    
    # Experiment setup
    if args.data == 'mnist':
        model = MLP(n_units=500, n_out=10)
        train, test = chainer.datasets.get_mnist()
    elif args.data == 'cifar10':
        model = NNet(n_out=10)
        train, test = chainer.datasets.get_cifar10()
    
    model = L.Classifier(model)
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
    
    # Optimizer
    if args.method == 'adam':
        optimizer = chainer.optimizers.Adam()
    elif args.method == 'sgd':
        optimizer = chainer.optimizers.MomentumSGD(lr=0.01)
    elif args.method == 'fem':
        optimizer = Hamiltonian.Hamiltonian(approx='first')
    elif args.method == 'sem':
        optimizer = Hamiltonian.Hamiltonian(approx='second')
   
    optimier.setup(model)
    
    # iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
    
    # Setup a trainer
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    if args.method == 'sgd':
        trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=(50, 'epoch'))
    
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    
    trainer.extend(extensions.ProgressBar())
    trainer.run()

if __name__ == '__main__':
    main()