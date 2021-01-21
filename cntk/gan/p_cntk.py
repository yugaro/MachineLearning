import cntk as C
import numpy as np
import os
import matplotlib.pyplot as plt
import time


def create_reader(path, is_training, input_dim, num_label_classes):
    """
    input:
        path (str): path to data,
        is_training (bool): true (when learning)
        input_dim (int): input number
        num_label_classes (int): class number
    output:
        C.io.MinibatchSource
    """
    labelStream = C.io.StreamDef(
        field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = C.io.StreamDef(
        field='features', shape=input_dim, is_sparse=False)

    deserailizer = C.io.CTFDeserializer(path, C.io.StreamDefs(
        labels=labelStream, features=featureStream))

    return C.io.MinibatchSource(deserailizer,
                                randomize=is_training,
                                max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)


# directory of learning and test data
data_dir = os.path.join("9000_cntk_data", "DataSets", "MNIST")

# path to directory of learning and test data
train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")

# check the exsitsence of test data
if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
    raise ValueError("Data not Found")
print("Data directory is {0}".format(data_dir))


# random seed
np.random.seed(123)

# input and output order of model
g_input_dim = 100
g_hidden_dim = 128
g_output_dim = d_input_dim = 784
d_hidden_dim = 128
d_output_dim = 1

minibatch_size = 1024  # batch size
num_minibatches = 40  # mini batch number
lr = 0.00005  # learning rate


def noise_sample(num_samples):
    """
    uniformly sampling from [-1, 1]
    input:
        num_samples (int): noise sample number
    output:
        np.random.uniform:
        noise to Generator(shape=(num_samples, g_input_dim))
    """
    return np.random.uniform(
        low=-1.0,
        high=1.0,
        size=[num_samples, g_input_dim]
    ).astype(np.float32)


def generator(z):
    """
    Generator
    input:
        z (ndarray): noise
    output:
        C.layers.Dense: output layer of 784 vector image
    """
    with C.layers.default_options(init=C.xavier()):
        h1 = C.layers.Dense(g_hidden_dim, activation=C.relu)(z)
        return C.layers.Dense(g_output_dim, activation=C.tanh)(h1)


def discriminator(x):
    """
    Discriminator
    input:
        x (C.input_variable): image of dataset or from generater
    output:
        C.layers.Dense:
        layer to output [0, 1] (probability of true data)
    """
    with C.layers.default_options(init=C.xavier()):
        h1 = C.layers.Dense(d_hidden_dim, activation=C.relu)(x)
        return C.layers.Dense(d_output_dim, activation=C.sigmoid)(h1)


def build_graph(noise_shape, image_shape, G_progress_printer, D_progress_printer):
    """
    modeling
    input:
        noise_shape (int): shape of noise to Generator
        image_shape (int): shape of input image
        G_progress_printer　(C.logging.ProgressPrinter): print process of Generator
        D_progress_printer(C.logging.ProgressPrinter): print process of Discriminator
    output:
        X_real (C.input_variable):container to receive real image [-1, 1]
        X_fake (C.layers.Dense): create image
        Z (C.input_variable): container to receive noise to Generator
        G_trainer (C.Trainer): trainer class of Generator
        D_trainer (C.Trainer): trainer class of Discriminator
    """
    input_dynamic_axes = [C.Axis.default_batch_axis(
    )]  # set dynamic axis (defaultBatchAxis)

    # container to receive image
    # noise into Generator
    Z = C.input_variable(noise_shape, dynamic_axes=input_dynamic_axes)
    X_real = C.input_variable(
        image_shape, dynamic_axes=input_dynamic_axes)  # real image

    X_real_scaled = 2 * (X_real / 255.0) - 1.0  # scale according to the output of Generator
    X_fake = generator(Z)  # image from Generato

    # define Discriminator
    # D_real: receiver of real image
    # D_fake: receiver of fake image
    # share weight
    D_real = discriminator(X_real_scaled)
    D_fake = D_real.clone(
        method='share',
        substitutions={X_real_scaled.output: X_fake.output}
    )

    # setting loss function of Generator, Discriminator
    G_loss = 1.0 - C.log(D_fake)
    D_loss = -(C.log(D_real) + C.log(1.0 - D_fake))

    # setting learning rate and optimazation method
    # use Adagrand
    G_learner = C.fsadagrad(
        parameters=X_fake.parameters,
        lr=C.learning_parameter_schedule_per_sample(lr),
        momentum=C.momentum_schedule_per_sample(0.9985724484938566)
    )

    D_learner = C.fsadagrad(
        parameters=D_real.parameters,
        lr=C.learning_parameter_schedule_per_sample(lr),
        momentum=C.momentum_schedule_per_sample(0.9985724484938566)
    )

    # define of Trainer of Generator
    G_trainer = C.Trainer(
        X_fake,
        (G_loss, None),
        G_learner,
        G_progress_printer
    )

    # define Trainer of Discriminator
    D_trainer = C.Trainer(
        D_real,
        (D_loss, None),
        D_learner,
        D_progress_printer
    )

    return X_real, X_fake, Z, G_trainer, D_trainer


def train(reader_train):
    """
    input:
        reader_train (C.io.MinibatchSource): reader
    output:
         Z (C.input_variable): container to receive noise into Generator
         X_fake (C.layers.Dense):leyer of output of images
    """
    k = 2  # learning ratio of Generator and Discriminator (1: k)

    # setting learning process
    print_frequency_mbsize = num_minibatches // 5
    pp_G = C.logging.ProgressPrinter(print_frequency_mbsize)
    pp_D = C.logging.ProgressPrinter(print_frequency_mbsize * k)

    X_real, X_fake, Z, G_trainer, D_trainer = \
        build_graph(g_input_dim, d_input_dim, pp_G, pp_D)

    input_map = {X_real: reader_train.streams.features}  # labeling is not necessary.

    # learning
    for train_step in range(num_minibatches):

        # leaning Generator for one time vs leaning Discriminator for k times
        # Learning Discriminator
        for gen_train_step in range(k):
            Z_data = noise_sample(minibatch_size)
            X_data = reader_train.next_minibatch(minibatch_size, input_map)
            # sumple number of mini batch == No
            if X_data[X_real].num_samples == Z_data.shape[0]:
                batch_inputs = {X_real: X_data[X_real].data, Z: Z_data}
                D_trainer.train_minibatch(batch_inputs)

        # learning Generator
        Z_data = noise_sample(minibatch_size)
        batch_inputs = {Z: Z_data}
        G_trainer.train_minibatch(batch_inputs)

    return Z, X_fake


def plot_images(images, subplot_shape):
    """
    plot MNIST image
    imput:
        images (ndarray): images (784px)
        subplot_shape (list): image row, row [index(int), row(int)]
    """
    plt.style.use('ggplot')
    fig, axes = plt.subplots(*subplot_shape)
    for image, ax in zip(images, axes.flatten()):
        ax.imshow(image.reshape(28, 28), vmin=0, vmax=1.0, cmap='gray')
        ax.axis('off')
    plt.show()


# create image
start = time.time()
reader_train = create_reader(
    train_file, True, d_input_dim, num_label_classes=10)

# learn
G_input, G_output = train(reader_train)
print("excution time: {}".format(time.time() - start))

# create images by adding noise to learning Generator
noise = noise_sample(36)
images = G_output.eval({G_input: noise})

plot_images(images, subplot_shape=[6, 6])
