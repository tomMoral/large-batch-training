'''
Train CNNs on the CIFAR10/CIFAR100
Plots a parametric plot between SB and LB
minimizers demonstrating the relative sharpness
of the two minima.

Requirements:
- Keras (with Theano)
- Matplotlib
- Numpy

GPU run command:
    KERAS_BACKEND=theano python plot_parametric_plot.py --network C[1-4]
'''

from __future__ import print_function
from keras.datasets import cifar10, cifar100
from keras.utils import np_utils
import numpy
import matplotlib.pyplot as plt
import argparse
import network_zoo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This code first trains the user-specific network (C[1-4])'
                    ' using small-batch ADAM and large-batch ADAM, and then '
                    'plots the parametric plot connecting the two minimizers '
                    'illustrating the sharpness difference.')
    parser.add_argument('-n', '--network', help='''Selects which network
                        to plot the parametric plots for.
                        Choices are C1, C2, C3 and C4.''', required=True)
    network_choice = vars(parser.parse_args())['network']

    nb_epoch = 30

    # the data, shuffled and split between train and test sets
    if network_choice in ['C1', 'C2']:
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        nb_classes = 10
    elif network_choice in ['C3', 'C4']:
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        nb_classes = 100
    else:
        raise ValueError('Invalid choice of network. Please choose one of C1, '
                         'C2, C3 or C4. Refer to the paper for details '
                         'regarding these networks')

    X_train = X_train.astype('float32')
    X_train = X_train.swapaxes(-1, -2).swapaxes(-3, -2)
    X_test = X_test.astype('float32')
    X_test = X_test.swapaxes(-1, -2).swapaxes(-3, -2)
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # build the network
    if network_choice in ['C1', 'C3']:
        model = network_zoo.shallownet(nb_classes)
        model_LK = network_zoo.shallownet(nb_classes, lastKernel=True)
    elif network_choice in ['C2', 'C4']:
        model = network_zoo.deepnet(nb_classes)
        model_LK = network_zoo.deepnet(nb_classes, lastKernel=True)

    # let's train the model using Adam
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.save_weights('x0.h5')

    # let's first find the small-batch solution
    model.fit(X_train, Y_train,
              batch_size=256,
              epochs=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
    sb_solution = [l.get_weights() for l in model.layers]

    # re-compiling to reset the optimizer accumulators
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # setting the initial (starting) point
    model.load_weights('x0.h5')

    # now, let's train the large-batch solution
    model.fit(X_train, Y_train,
              batch_size=5000,
              epochs=nb_epoch,
              validation_data=(X_test, Y_test))
    lb_solution = [l.get_weights() for l in model.layers]
    model.save_weights('xf_lb.h5')

    # Use last kernel for the large batch classifier
    model_LK.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
    model_LK.load_weights('xf_lb.h5')
    model_LK.fit(X_train, Y_train,
                 batch_size=5000,
                 epochs=1,
                 validation_data=(X_test, Y_test))
    lb_solution[-1] = model_LK.layers[-1].get_weights()

    # parametric plot data collection
    # we discretize the interval [-1,2] into 25 pieces
    alpha_range = numpy.linspace(-1, 2, 25)
    data_for_plotting = numpy.zeros((25, 4))

    i = 0
    for alpha in alpha_range:
        print("Computing value for alpha={}".format(alpha))
        for p in range(len(model.layers)):
            weights = [lb_w * alpha + sb_w * (1 - alpha)
                       for lb_w, sb_w in zip(lb_solution[p], sb_solution[p])]
            model.layers[p].set_weights(weights)

        train_xent, train_acc = model.evaluate(X_train, Y_train,
                                               batch_size=5000, verbose=0)
        test_xent, test_acc = model.evaluate(X_test, Y_test,
                                             batch_size=5000, verbose=0)
        data_for_plotting[i, :] = [train_xent, train_acc, test_xent, test_acc]
        i += 1

    import IPython
    IPython.embed()

    # finally, let's plot the data
    # we plot the XENT loss on the left Y-axis
    # and accuracy on the right Y-axis
    # if you don't have Matplotlib, simply print
    # data_for_plotting to file and use a different plotter

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(alpha_range, data_for_plotting[:, 0], 'b-')
    ax1.plot(alpha_range, data_for_plotting[:, 2], 'b--')

    ax2.plot(alpha_range, data_for_plotting[:, 1] * 100., 'r-')
    ax2.plot(alpha_range, data_for_plotting[:, 3] * 100., 'r--')

    ax1.set_xlabel('alpha')
    ax1.set_ylabel('Cross Entropy', color='b')
    ax2.set_ylabel('Accuracy', color='r')
    ax1.legend(('Train', 'Test'), loc=0)

    ax1.grid(b=True, which='both')
    plt.savefig('Figures/' + network_choice + '.pdf')
    print('Plot save as ' + network_choice + '.pdf in the Figures/ folder')
