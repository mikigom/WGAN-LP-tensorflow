import sys

import tensorflow as tf

import data_generator
from model import Generator, Critic

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_integer("n_epoch", 5000, "Epoch to train [5000]")
flags.DEFINE_integer("n_batch_size", 256, "Batch size to train [256]")
flags.DEFINE_integer("latent_dimensionality", 2, "Dimensionality of the latent variables [2]")

"""
During training, 10 critic updates are performed for every generator update,
except for the first 25 generator updates,
where the critic is updated 100 times for each generator update
in order to get closer to the optimal critic in the beginning of training.
"""

flags.DEFINE_integer("begining_init_step", 25, "[25]")
flags.DEFINE_integer("n_c_iters_under_begining_init_step", 100, "[100]")
flags.DEFINE_integer("n_c_iters_over_begining_init_step", 10, "[10]")

flags.DEFINE_float("learning_rate", 5e-5, "Learning rate of optimizer [5e-5]")
flags.DEFINE_float("lambda", 5., "Weights for critics' regularization term [5]")
flags.DEFINE_string("DRAGAN_purturbation", "no_purf", "[no_purf, purf_only_generated, perf_both]")
flags.DEFINE_string("dataset", 'GeneratorSwissRoll',
                    "Which dataset is used? [GeneratorGaussians8, GeneratorGaussians25, GeneratorSwissRoll]")
FLAGS = flags.FLAGS


class Trainer(object):
    def __init__(self):
        self.dataset_generator = None
        self.real_input = None

        self.z = None

        self.generator = None
        self.critic_x = None
        self.critic_gz = None

        self.g_loss = None
        self.c_negative_loss = None
        self.c_regularization_loss = None
        self.c_loss = None

        self.step = None

        self.sess = None
        self.step_inc = None
        self.g_opt = None
        self.c_opt = None

        self.g_update_fetch_dict = None
        self.c_update_fetch_dict = None
        self.c_feed_dict = None

        self.coord = None
        self.threads = None

        self.define_dataset()
        self.define_latent()
        self.define_model()
        self.define_loss()
        self.define_optim()
        self.define_writer_and_summary()
        self.define_saver()
        self.initialize_session_and_etc()
        self.define_feed_and_fetch()

    def __del__(self):
        self.sess.close()

    def define_dataset(self):
        self.dataset_generator = iter(getattr(data_generator, FLAGS.dataset)(FLAGS.n_batch_size))
        self.real_input = tf.placeholder(tf.float32, shape=(FLAGS.n_batch_size, None))

    def define_latent(self):
        self.z = tf.random_uniform(FLAGS.latent_dimensionality, minval=-1., maxval=1., name='z')

    def define_model(self):
        self.generator = Generator(self.z)
        self.critic_x = Critic(self.real_input)
        self.critic_gz = Critic(self.generator.output_tensor)

    def define_loss(self):
        # TODO
        self.g_loss = ???
        self.c_negative_loss = ???
        self.c_regularization_loss = ???
        self.c_loss = self.c_negative_loss + self.c_regularization_loss

    def define_optim(self):
        self.step = tf.Variable(0, name='step', trainable=False)
        self.step_inc = tf.assign(self.step, self.step + 1)

        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)

        self.g_opt = optimizer.minimize(self.g_loss, var_list=self.generator.var_list)
        self.c_opt = optimizer.minimize(self.c_loss, var_list=self.critic_x.var_list)

    def define_writer_and_summary(self):
        # TODO
        pass

    def define_saver(self):
        # TODO
        pass

    def initialize_session_and_etc(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def define_feed_and_fetch(self):
        self.g_update_fetch_dict = {
            "opt": self.g_opt,
            "z": self.z,
            "G_z": self.generator.output_tensor,
            "loss": self.g_loss,
            "step": self.step,
        }

        self.c_update_fetch_dict = {
            "opt": self.c_opt,
            "x": self.real_input,
            "G_z": self.generator.output_tensor,
            "negative_loss": self.c_negative_loss,
            "regularization_loss": self.c_regularization_loss,
            "step": self.step,
        }

        self.c_feed_dict = {
            self.real_input: None
        }

    def train(self):
        try:
            print("[.] Learning Start...")
            step = 0
            while not self.coord.should_stop():
                if step >= FLAGS.n_epoch:
                    raise tf.errors.OutOfRangeError

                self.feed_dict[self.real_input] = next(self.dataset_generator)
                step = self.sess.run(self.step)

                n_c_iters = (FLAGS.n_c_iters_under_begining_init_step
                             if step < FLAGS.begining_init_step
                             else FLAGS.n_c_iters_over_begining_init_step)
                for _ in range(n_c_iters):
                    c_fetch_dict = self.sess.run(self.c_update_fetch_dict,
                                                 feed_dict=self.c_feed_dict)

                g_fetch_dict = self.sess.run(self.g_update_fetch_dict)

                # TODO

                self.sess.run(self.step_inc)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            self.coord.request_stop()
        except KeyboardInterrupt:
            print("Interrupted")
            self.coord.request_stop()
        except Exception as e:
            print(e)
            print("@ line {}".format(sys.exc_info()[-1].tb_lineno))
        finally:
            print('Stop')
            self.coord.request_stop()
            self.coord.join(self.threads)
