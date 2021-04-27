import time

from utils.resnet_ops import *
from utils.resnet_utils import *
from keras.utils import to_categorical

class ResNet_DEAL(object):
    def __init__(self, sess, args):
        self.model_name = 'ResNet'
        self.sess = sess
        self.dataset_name = args.dataset


        if self.dataset_name == 'cifar10_keras':
            self.img_size = [32, 32]
            self.c_dim = 3
            self.label_dim = 10

        if self.dataset_name == 'cifar100_keras':
            self.img_size = [32, 32]
            self.c_dim = 3
            self.label_dim = 100

        if self.dataset_name == 'mnist_keras':
            self.img_size = [28, 28]
            self.c_dim = 1
            self.label_dim = 10

        if self.dataset_name == 'svhn':
            self.img_size = [32,32]
            self.c_dim = 1
            self.label_dim = 10
        if self.dataset_name == 'medical':
            self.img_size = [128, 128]
            self.c_dim = 1
            self.label_dim = 2


        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir

        self.res_n = args.res_n

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.iteration = None


        self.init_lr = args.lr

        #DEAL Modifications###########################################################
        self.W3 = None
        self.W4 = None
        self.annealing_step = None
        ##############################################################################


    ##################################################################################
    # Generator
    ##################################################################################

    def network(self, x, is_training=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):

            print('RESNET_DEAL')

            annealing_step = tf.placeholder(dtype=tf.int32)


            if self.res_n < 50 :
                residual_block = resblock
            else :
                residual_block = bottle_resblock

            residual_list = get_residual_layer(self.res_n)

            ch = 32 # paper is 64
            x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

            for i in range(residual_list[0]) :
                x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

            for i in range(1, residual_list[1]) :
                x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

            for i in range(1, residual_list[2]) :
                x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

            ########################################################################################################

            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

            for i in range(1, residual_list[3]) :
                x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

            ########################################################################################################


            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            x = global_avg_pooling(x)
            #x = fully_conneted(x, units=self.label_dim, scope='logit')

            #DEAL Modifications##############################################################################
            x = flatten(x)
            logits, self.W3, self.W4 = DEAL_dense_layer(x, units=self.label_dim, scope='logit')

            print('LOGITS')
            print(logits.shape)

            evidence = softsign_evidence(logits)

            print('EVIDENCE')
            print(evidence.shape)

            alpha = evidence + 1
            #u = K / tf.reduce_sum(alpha, axis=1, keepdims=True)
            #prob = alpha / tf.reduce_sum(alpha, 1, keepdims=True)

            ################################################################################################


            return alpha, annealing_step

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self, val_x, val_y, test_x, test_y):

        remove_checkpoints()

        if self.sess._closed == False:
            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))


        """ Graph Input """
        self.train_inptus = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.c_dim], name='train_inputs')
        self.train_labels = tf.placeholder(tf.float32, [self.batch_size, self.label_dim], name='train_labels')

        self.val_inputs = tf.placeholder(tf.float32, [len(val_x), self.img_size[0], self.img_size[1], self.c_dim], name='val_inputs')
        self.val_labels = tf.placeholder(tf.float32, [len(val_y), self.label_dim], name='val_labels')

        self.test_inptus = tf.placeholder(tf.float32, [len(test_x), self.img_size[0], self.img_size[1], self.c_dim], name='test_inputs')
        self.test_labels = tf.placeholder(tf.float32, [len(test_y), self.label_dim], name='test_labels')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """

        self.train_logits, annealing_step = self.network(self.train_inptus)
        self.val_logits, annealing_step = self.network(self.val_inputs, is_training=False, reuse=True)
        self.test_logits, annealing_step = self.network(self.test_inptus, is_training=False, reuse=True)

        #self.train_loss, self.train_accuracy = classification_loss(logit=self.train_logits, label=self.train_labels)
        #self.test_loss, self.test_accuracy = classification_loss(logit=self.test_logits, label=self.test_labels)
        
        #reg_loss = tf.losses.get_regularization_loss()
        #self.train_loss += reg_loss
        #self.test_loss += reg_loss

        #DEAL Modifications###################################################################################

        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        self.annealing_step = annealing_step

        self.train_loss, self.train_accuracy = deal_loss(logits=self.train_logits, label=self.train_labels,
                                                         units=self.label_dim, W3=self.W3, W4=self.W4,
                                                         lmb=0.005,
                                                         global_step=global_step,
                                                         annealing_step=annealing_step)
        self.val_loss, self.val_accuracy = deal_loss(logits=self.val_logits, label=self.val_labels,
                                                       units=self.label_dim, W3=self.W3, W4=self.W4,
                                                       lmb=0.005,
                                                       global_step=global_step,
                                                       annealing_step=annealing_step)
        self.test_loss, self.test_accuracy = deal_loss(logits=self.test_logits, label=self.test_labels,
                                                       units=self.label_dim, W3=self.W3, W4=self.W4,
                                                       lmb=0.005,
                                                       global_step=global_step,
                                                       annealing_step=annealing_step)


        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.train_loss, global_step=global_step)

        ######################################################################################################


        """ Training """
        #self.optim = tf.train.MomentumOptimizer(self.lr, momentum=0.9).minimize(self.train_loss)

        """" Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_accuracy = tf.summary.scalar("train_accuracy", self.train_accuracy)

        self.summary_val_loss = tf.summary.scalar("val_loss", self.val_loss)
        self.summary_val_accuracy = tf.summary.scalar("val_accuracy", self.val_accuracy)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_accuracy])
        self.val_summary = tf.summary.merge([self.summary_val_loss, self.summary_val_accuracy])

    ##################################################################################
    # Train
    ##################################################################################

    def fit(self, train_x, train_y, val_x, val_y, FLAGS):

        # transform labels to categorial values
        if FLAGS.dataset == 'cifar10_keras':
            train_y = to_categorical(train_y, 10)
            val_y = to_categorical(val_y, 10)
        if FLAGS.dataset == 'mnist_keras':
            train_y = to_categorical(train_y, 10)
            val_y = to_categorical(val_y, 10)
        if FLAGS.dataset == 'cifar100_keras':
            train_y = to_categorical(train_y, 100)
            val_y = to_categorical(val_y, 100)
        if FLAGS.dataset == 'svhn':
            train_y = to_categorical(train_y, 10)
            val_y = to_categorical(val_y, 10)
        if FLAGS.dataset == 'medical':
            train_y = to_categorical(train_y, 2)
            val_y = to_categorical(val_y, 2)

        # determine number of iterations
        self.iteration = len(train_x) // self.batch_size

        # initialize all variables
        tf.global_variables_initializer().run(session=self.sess)

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            epoch_lr = self.init_lr
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter

            if start_epoch >= int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.01
            elif start_epoch >= int(self.epoch * 0.5) and start_epoch < int(self.epoch * 0.75) :
                epoch_lr = epoch_lr * 0.1
            print(" [*] Load SUCCESS")
        else:
            epoch_lr = self.init_lr
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # Learning rate decrease
            #if epoch == int(self.epoch * 0.5) or epoch == int(self.epoch * 0.75) :
            #    epoch_lr = epoch_lr * 0.1

            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_x = train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_y = train_y[idx*self.batch_size:(idx+1)*self.batch_size]

                batch_x = data_augmentation(batch_x, self.img_size, self.dataset_name)

                train_feed_dict = {
                    self.train_inptus : batch_x,
                    self.train_labels : batch_y,
                    self.lr : epoch_lr,
                    #DEAL Modifications ############################################################
                    self.annealing_step : 50 * self.iteration
                    ################################################################################
                }

                val_feed_dict = {
                    self.val_inputs : val_x,
                    self.val_labels : val_y,
                    # DEAL Modifications ############################################################
                    self.annealing_step: 50 * self.iteration
                    ################################################################################
                }


                # update network
                _, summary_str, train_loss, train_accuracy = self.sess.run(
                    [self.optim, self.train_summary, self.train_loss, self.train_accuracy], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # test
                summary_str, val_loss, val_accuracy = self.sess.run(
                    [self.val_summary, self.val_loss, self.val_accuracy], feed_dict=val_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, train_accuracy: %.2f, val_accuracy: %.2f, learning_rate : %.4f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, train_accuracy, val_accuracy, epoch_lr))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def decision_function(self, X):
        return self.predict(X)

    def predict(self, X):

        train_x = X
        train_x_length = train_x.shape[0]

        if self.FLAGS.dataset == 'cifar10_keras':
            pred = np.zeros(shape=(0, 10))
        if self.FLAGS.dataset == 'mnist_keras':
            pred = np.zeros(shape=(0, 10))
        if self.FLAGS.dataset == 'cifar100_keras':
            pred = np.zeros(shape=(0, 100))
        if self.FLAGS.dataset == 'svhn':
            pred = np.zeros(shape=(0, 10))
        if self.FLAGS.dataset == 'medical':
            pred = np.zeros(shape=(0, 2))

        if self.FLAGS.dataset == 'cifar10_keras' or self.FLAGS.dataset == 'mnist_keras' or self.FLAGS.dataset == 'cifar100_keras' or self.FLAGS.dataset == 'svhn':
            start=0
            end=2000
            for idx in range(0, int(train_x_length/2000)):
                X_cache = train_x[start:end]


                predict_feed_dict = {
                    self.val_inputs: X_cache
                }

                prediction = self.sess.run(
                    [self.val_logits], feed_dict=predict_feed_dict
                )

                p_pred = np.array(prediction)
                p_pred = p_pred.reshape([p_pred.shape[1],p_pred.shape[2]])
                pred = np.concatenate([pred, p_pred])

                start += 2000
                end += 2000

        if self.FLAGS.dataset == 'medical':
            start = 0
            end = 200
            for idx in range(0, int(train_x_length / 200)):
                X_cache = train_x[start:end]

                predict_feed_dict = {
                    self.val_inputs: X_cache
                }

                prediction = self.sess.run(
                    [self.val_logits], feed_dict=predict_feed_dict
                )

                p_pred = np.array(prediction)
                p_pred = p_pred.reshape([p_pred.shape[1], p_pred.shape[2]])
                pred = np.concatenate([pred, p_pred])

                start += 200
                end += 200

        return pred

    @property
    def model_dir(self):
        return "{}{}_{}_{}_{}".format(self.model_name, self.res_n, self.dataset_name, self.batch_size, self.init_lr)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self, test_x, test_y, FLAGS):

        # transform labels to categorial values
        self.FLAGS = FLAGS
        if FLAGS.dataset == 'cifar10_keras':
            test_y = to_categorical(test_y, 10)
        if FLAGS.dataset == 'mnist_keras':
            test_y = to_categorical(test_y, 10)
        if FLAGS.dataset == 'cifar100_keras':
            test_y = to_categorical(test_y, 100)
        if FLAGS.dataset == 'svhn':
            test_y = to_categorical(test_y, 10)
        if FLAGS.dataset == 'medical':
            test_y = to_categorical(test_y, 2)

        tf.global_variables_initializer().run(session=self.sess)

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_feed_dict = {
            self.test_inptus: test_x,
            self.test_labels: test_y
        }


        test_accuracy = self.sess.run(self.test_accuracy, feed_dict=test_feed_dict)
        print("test_accuracy: {}".format(test_accuracy))

        return test_accuracy


    def score(self, test_x, test_y, FLAGS):
        test_accuracy = self.test(test_x, test_y, FLAGS)

        return test_accuracy
