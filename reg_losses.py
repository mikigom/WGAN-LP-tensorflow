import tensorflow as tf
from model import Critic
slim = tf.contrib.slim


def get_perbatuation_samples(training_samples, generated_samples, per_type,
                             dragan_parameter_C):
    x_hat = None

    if per_type == 'no_purf':
        x_hat = training_samples

    # Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." arXiv preprint arXiv:1704.00028 (2017).
    elif per_type == 'wgan_gp':
        epsilon = tf.random_uniform(
            shape=[tf.shape(training_samples)[0], 1],
            minval=0.,
            maxval=1.
        )
        x_hat = epsilon * training_samples + (1 - epsilon) * generated_samples

    # Kodali, Naveen, et al. "How to Train Your DRAGAN." arXiv preprint arXiv:1705.07215 (2017).
    elif per_type == 'dragan_only_training':
        u = tf.random_uniform(
            shape=[tf.shape(training_samples)[0], 1],
            minval=0.,
            maxval=1.
        )
        _, batch_std = tf.nn.moments(tf.reshape(training_samples, [-1]), axes=[0])

        delta = dragan_parameter_C * batch_std * u

        alpha = tf.random_uniform(
            shape=[tf.shape(training_samples)[0], 1],
            minval=0.,
            maxval=1.
        )

        x_hat = training_samples + (1 - alpha) * delta

    elif per_type == 'dragan_both':
        samples = tf.concat([training_samples, generated_samples], axis=0)

        u = tf.random_uniform(
            shape=[tf.shape(samples)[0], 1],
            minval=0.,
            maxval=1.
        )
        _, batch_std = tf.nn.moments(tf.reshape(samples, [-1]), axes=[0])

        delta = dragan_parameter_C * batch_std * u

        alpha = tf.random_uniform(
            shape=[tf.shape(samples)[0], 1],
            minval=0.,
            maxval=1.
        )

        x_hat = samples + (1 - alpha) * delta

    else:
        NotImplementedError('arg per_type is not injected correctly.')

    return x_hat


def get_regularization_term(training_samples, generated_samples,
                            reg_type, per_type,
                            critic_variable_scope_name,
                            dragan_parameter_C=0.5):
    x_hat = get_perbatuation_samples(training_samples, generated_samples, per_type,
                                     dragan_parameter_C)

    critic = Critic(x_hat, variable_scope_name=critic_variable_scope_name, reuse=True)
    gradients = tf.gradients(critic.output_tensor, [x_hat])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

    gradient_penalty = None

    # Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." arXiv preprint arXiv:1704.00028 (2017).
    if reg_type == 'GP':
        gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)

    # Henning Petzka, Asja Fischer, and Denis Lukovnicov. "On the regularization of Wasserstein GANs."
    # arXiv preprint arXiv:1709.08894 (2017).
    elif reg_type == 'LP':
        gradient_penalty = tf.reduce_mean((tf.maximum(0., slopes - 1)) ** 2)

    else:
        NotImplementedError('arg reg_type is not injected correctly.')

    return gradient_penalty, x_hat
