import tensorflow as tf
from typing import Callable

slim = tf.contrib.slim

__leaky_relu_alpha__ = 0.2


def __leaky_relu__(x, alpha=__leaky_relu_alpha__, name='Leaky_ReLU'):
    return tf.maximum(x, alpha*x, name=name)


class Model(object):
    def __init__(self,
                 input_tensor: tf.Variable,
                 variable_scope_name: str,
                 n_hidden_neurons: int,
                 n_hidden_layers: int,
                 n_out_dim: int,
                 activation_fn: Callable,
                 reuse: bool):
        self.input = input_tensor
        self.variable_scope_name = variable_scope_name
        self.n_hidden_neurons = n_hidden_neurons
        self.n_hidden_layers = n_hidden_layers
        self.n_out_dim = n_out_dim
        self.activation_fn = activation_fn
        self.reuse = reuse
        self.output_tensor = None
        self.define_model()

    def define_model(self):
        with tf.variable_scope(self.variable_scope_name, reuse=self.reuse):
            x = self.input
            with slim.variable_scope([slim.fully_connected],
                                     num_outputs=self.n_hidden_neurons,
                                     activation_fn=self.activation_fn):
                for i in range(self.n_hidden_layers):
                    x = slim.fully_connected(inputs=x)
            self.output_tensor = slim.fully_connected(inputs=x,
                                                      num_outputs=self.n_out_dim,
                                                      activation_fn=None)


class Generator(Model):
    def __init__(self,
                 input_tensor: tf.Variable,
                 variable_scope_name: str='Generator',
                 n_hidden_neurons: int=512,
                 n_hidden_layers: int=3,
                 n_out_dim: int=2,
                 activation_fn: Callable=__leaky_relu__,
                 reuse: bool=False):
        super(Generator, self).__init__(input_tensor,
                                        variable_scope_name,
                                        n_hidden_neurons,
                                        n_hidden_layers,
                                        n_out_dim,
                                        activation_fn,
                                        reuse)


class Critic(Model):
    def __init__(self,
                 input_tensor: tf.Variable,
                 variable_scope_name: str='Critic',
                 n_hidden_neurons: int=512,
                 n_hidden_layers: int=3,
                 n_out_dim: int=1,
                 activation_fn: Callable=__leaky_relu__,
                 reuse: bool=False):
        super(Critic, self).__init__(input_tensor,
                                     variable_scope_name,
                                     n_hidden_neurons,
                                     n_hidden_layers,
                                     n_out_dim,
                                     activation_fn,
                                     reuse)
