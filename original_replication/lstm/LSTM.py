# -*- coding: utf-8 -*-

"""
LSTM training and prediction on MIDI dataset to generate new music.

Based directly on the supplied original implementation.

Minimal execution changes:
1. music21 .flat -> .flatten()
2. float32 used for large arrays to reduce RAM
3. Training uses short sequence windows instead of materialising the
   complete padded song sequence as dense one-hot matrices
4. Windows containing only padding are skipped

The original LSTM equations, loss calculation, manual backpropagation,
manual Adam optimizer, embedding update, hyperparameters and generation
approach are retained.
"""

from music21 import *
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


# =============================================================================
# PART 1
# Process data and get it ready for LSTM
# =============================================================================


# -----------------------------------------------------------------------------
# Optional visualisation of the compatibility MIDI.
#
# The original code executed these immediately:
#
# midi_data = converter.parse('data/ct.38-40 no.1.mid')
# midi_data.write('text')
# midi_data.plot()
# midi_data.plot('histogram', 'pitchclass')
#
# They are disabled here because they are visualisation only and can block
# automated training. They do not affect the model.
# -----------------------------------------------------------------------------


# =============================================================================
# PREPARE MUSIC DATA
# =============================================================================

def process_data(songs):

    whole_data = []

    for song_number, song in enumerate(songs, start=1):

        print(
            "Processing MIDI {}/{}: {}".format(
                song_number,
                len(songs),
                os.path.basename(song)
            )
        )

        # Compatibility change:
        # original used converter.parse(song).flat
        midi_data = converter.parse(song).flatten()

        song_data = []

        prev_offset = -1

        for element in midi_data:

            if isinstance(element, note.Note):

                if element.offset != prev_offset:

                    song_data.append(
                        [
                            element.pitch.nameWithOctave,
                            element.quarterLength
                        ]
                    )

                else:

                    if len(song_data[-1]) < 4:

                        song_data[-1].append(
                            element.pitch.nameWithOctave
                        )

                        song_data[-1].append(
                            element.quarterLength
                        )

                prev_offset = element.offset

            elif isinstance(element, chord.Chord):

                pitch_names = '.'.join(
                    n.nameWithOctave
                    for n in element.pitches
                )

                if element.offset != prev_offset:

                    song_data.append(
                        [
                            pitch_names,
                            element.quarterLength
                        ]
                    )

                else:

                    if len(song_data[-1]) < 4:

                        song_data[-1].append(
                            pitch_names
                        )

                        song_data[-1].append(
                            element.quarterLength
                        )

                prev_offset = element.offset

        for item in song_data:

            if len(item) < 4:

                item.append(None)
                item.append(None)

        whole_data.append(
            song_data
        )

    return whole_data


# =============================================================================
# TRANSFORM DATA
# =============================================================================

def transform_data(songs):

    max_len = 0

    for song in songs:

        max_len = max(
            max_len,
            len(song)
        )

    for song in songs:

        for i in range(
            max_len - len(song)
        ):

            song.append(
                [
                    None,
                    None,
                    None,
                    None
                ]
            )

    transformed_data = []

    for song in songs:

        t_song_data = []

        for item in song:

            t_song_data.append(
                tuple(item)
            )

        transformed_data.append(
            t_song_data
        )

    return transformed_data


# =============================================================================
# DICTIONARY
# =============================================================================

def get_dictionary(songs):

    possible_combs = set(
        item
        for song in songs
        for item in song
    )

    data_to_int = dict(
        (v, i)
        for i, v in enumerate(possible_combs)
    )

    int_to_data = dict(
        (i, v)
        for i, v in enumerate(possible_combs)
    )

    return (
        data_to_int,
        int_to_data
    )


# =============================================================================
# LOAD MIDI
# =============================================================================

songs = glob.glob(
    'data/*.mid'
)

if len(songs) == 0:

    raise RuntimeError(
        "No MIDI files found in data/"
    )


songs_data = process_data(
    songs
)

songs_data = transform_data(
    songs_data
)


print(
    "Whole music data size:",
    np.array(
        songs_data,
        dtype=object
    ).shape
)


data_to_int, int_to_data = get_dictionary(
    songs_data
)


print(
    "Number of unique notes:",
    len(data_to_int)
)


# =============================================================================
# PART 2
# Train data
# =============================================================================


# =============================================================================
# ORIGINAL HYPERPARAMETERS
# =============================================================================

INPUT = 100

HIDDEN = 256

vocab_size = len(
    data_to_int
)

OUTPUT = vocab_size

LEARNING_RATE = 0.005

BETA1 = 0.9

BETA2 = 0.999

BATCH_SIZE = 50


# =============================================================================
# MINIMAL RAM ADAPTATION
# =============================================================================

# The supplied code tries to retain all ~6763 timesteps as dense one-hot
# matrices simultaneously.
#
# We process only this many consecutive timesteps at once.
#
# This is truncated BPTT and is the one substantive training adaptation
# required to fit the supplied implementation into available RAM.

SEQUENCE_LENGTH = 16


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def sigmoid(X):

    return 1. / (
        1 + np.exp(-X)
    )


def softmax(X):

    # Original formula retained.
    # Subtracting max only prevents exp overflow and does not change
    # the mathematical softmax probabilities.

    X = X - np.max(
        X,
        axis=1,
        keepdims=True
    )

    exp_X = np.exp(X)

    return (
        exp_X
        /
        np.sum(
            exp_X,
            axis=1
        ).reshape(
            -1,
            1
        )
    )


def tanh(X):

    return np.tanh(X)


def dtanh(X):

    return 1 - X ** 2


# =============================================================================
# LSTM
# =============================================================================

def initialize_parameters():

    parameters = {}

    parameters['fgw'] = np.random.normal(
        0,
        0.01,
        (
            INPUT + HIDDEN,
            HIDDEN
        )
    ).astype(
        np.float32
    )

    parameters['igw'] = np.random.normal(
        0,
        0.01,
        (
            INPUT + HIDDEN,
            HIDDEN
        )
    ).astype(
        np.float32
    )

    parameters['ogw'] = np.random.normal(
        0,
        0.01,
        (
            INPUT + HIDDEN,
            HIDDEN
        )
    ).astype(
        np.float32
    )

    parameters['ggw'] = np.random.normal(
        0,
        0.01,
        (
            INPUT + HIDDEN,
            HIDDEN
        )
    ).astype(
        np.float32
    )

    parameters['how'] = np.random.normal(
        0,
        0.01,
        (
            HIDDEN,
            OUTPUT
        )
    ).astype(
        np.float32
    )

    return parameters


# =============================================================================
# ADAM STATE
# =============================================================================

def initialize_V(parameters):

    V = {}

    V['vfgw'] = np.zeros_like(
        parameters['fgw'],
        dtype=np.float32
    )

    V['vigw'] = np.zeros_like(
        parameters['igw'],
        dtype=np.float32
    )

    V['vogw'] = np.zeros_like(
        parameters['ogw'],
        dtype=np.float32
    )

    V['vggw'] = np.zeros_like(
        parameters['ggw'],
        dtype=np.float32
    )

    V['vhow'] = np.zeros_like(
        parameters['how'],
        dtype=np.float32
    )

    return V


def initialize_S(parameters):

    S = {}

    S['sfgw'] = np.zeros_like(
        parameters['fgw'],
        dtype=np.float32
    )

    S['sigw'] = np.zeros_like(
        parameters['igw'],
        dtype=np.float32
    )

    S['sogw'] = np.zeros_like(
        parameters['ogw'],
        dtype=np.float32
    )

    S['sggw'] = np.zeros_like(
        parameters['ggw'],
        dtype=np.float32
    )

    S['show'] = np.zeros_like(
        parameters['how'],
        dtype=np.float32
    )

    return S


# =============================================================================
# EMBEDDINGS
# =============================================================================

def get_embeddings(
    batch_dataset,
    embeddings
):

    embedding_dataset = np.matmul(
        batch_dataset,
        embeddings
    )

    return embedding_dataset


# =============================================================================
# LSTM CELL
# =============================================================================

def lstm_cell(
    batch_dataset,
    prev_activation_matrix,
    prev_cell_matrix,
    parameters
):

    concat_dataset = np.concatenate(
        (
            batch_dataset,
            prev_activation_matrix
        ),
        axis=1
    )

    fa = sigmoid(
        np.matmul(
            concat_dataset,
            parameters['fgw']
        )
    )

    ia = sigmoid(
        np.matmul(
            concat_dataset,
            parameters['igw']
        )
    )

    oa = sigmoid(
        np.matmul(
            concat_dataset,
            parameters['ogw']
        )
    )

    ga = tanh(
        np.matmul(
            concat_dataset,
            parameters['ggw']
        )
    )

    cell_memory_matrix = (
        np.multiply(
            fa,
            prev_cell_matrix
        )
        +
        np.multiply(
            ia,
            ga
        )
    )

    activation_matrix = np.multiply(
        oa,
        tanh(
            cell_memory_matrix
        )
    )

    lstm_activations = {}

    lstm_activations['fa'] = fa
    lstm_activations['ia'] = ia
    lstm_activations['oa'] = oa
    lstm_activations['ga'] = ga

    return (
        lstm_activations,
        cell_memory_matrix,
        activation_matrix
    )


# =============================================================================
# OUTPUT CELL
# =============================================================================

def output_cell(
    activation_matrix,
    parameters
):

    output_matrix = softmax(
        np.matmul(
            activation_matrix,
            parameters['how']
        )
    )

    return output_matrix


# =============================================================================
# ORIGINAL LOSS / ACCURACY
# =============================================================================

def cal_loss_accuracy(
    batch_labels,
    output_cache
):

    loss = 0

    accuracy = 0

    prob = 1

    batch_size = batch_labels[0].shape[0]

    for i in range(
        1,
        len(output_cache) + 1
    ):

        labels = batch_labels[i]

        pred = output_cache[
            'o' + str(i)
        ]

        # Avoid log(0) numerical failure.
        pred_safe = np.clip(
            pred,
            1e-8,
            1 - 1e-7
        )

        prob = np.multiply(
            prob,
            np.sum(
                np.multiply(
                    labels,
                    pred_safe
                ),
                axis=1
            ).reshape(
                -1,
                1
            )
        )

        loss += np.sum(
            (
                np.multiply(
                    labels,
                    np.log(
                        pred_safe
                    )
                )
                +
                np.multiply(
                    1 - labels,
                    np.log(
                        1 - pred_safe
                    )
                )
            ),
            axis=1
        ).reshape(
            -1,
            1
        )

        accuracy += np.array(
            np.argmax(
                labels,
                1
            )
            ==
            np.argmax(
                pred,
                1
            ),
            dtype=np.float32
        ).reshape(
            -1,
            1
        )

    # Keep original perplexity equation.
    #
    # Clamp probability only to prevent division by literal zero.
    prob = np.maximum(
        prob,
        1e-30
    )

    perplexity = np.sum(
        (
            1 / prob
        )
        **
        (
            1 / len(
                output_cache
            )
        )
    ) / batch_size

    loss = np.sum(
        loss
    ) * (
        -1 / batch_size
    )

    accuracy = (
        np.sum(
            accuracy
        )
        /
        batch_size
    ) / len(
        output_cache
    )

    return (
        perplexity,
        loss,
        accuracy
    )


# =============================================================================
# FORWARD PROPAGATION
# =============================================================================

def forward_propagation(
    batches,
    parameters,
    embeddings
):

    batch_size = batches[0].shape[0]

    lstm_cache = {}

    activation_cache = {}

    cell_cache = {}

    output_cache = {}

    embedding_cache = {}

    a0 = np.zeros(
        [
            batch_size,
            HIDDEN
        ],
        dtype=np.float32
    )

    c0 = np.zeros(
        [
            batch_size,
            HIDDEN
        ],
        dtype=np.float32
    )

    activation_cache['a0'] = a0

    cell_cache['c0'] = c0

    for i in range(
        len(batches) - 1
    ):

        batch_dataset = batches[i]

        batch_dataset = get_embeddings(
            batch_dataset,
            embeddings
        )

        embedding_cache[
            'emb' + str(i)
        ] = batch_dataset

        (
            lstm_activations,
            ct,
            at
        ) = lstm_cell(
            batch_dataset,
            a0,
            c0,
            parameters
        )

        ot = output_cell(
            at,
            parameters
        )

        lstm_cache[
            'lstm' + str(i + 1)
        ] = lstm_activations

        activation_cache[
            'a' + str(i + 1)
        ] = at

        cell_cache[
            'c' + str(i + 1)
        ] = ct

        output_cache[
            'o' + str(i + 1)
        ] = ot

        a0 = at

        c0 = ct

    return (
        embedding_cache,
        lstm_cache,
        activation_cache,
        cell_cache,
        output_cache
    )


# =============================================================================
# OUTPUT CELL ERROR
# =============================================================================

def calculate_output_cell_error(
    batch_labels,
    output_cache,
    parameters
):

    output_error_cache = {}

    activation_error_cache = {}

    for i in range(
        1,
        len(output_cache) + 1
    ):

        error_output = (
            output_cache[
                'o' + str(i)
            ]
            -
            batch_labels[i]
        )

        error_activation = np.matmul(
            error_output,
            parameters['how'].T
        )

        output_error_cache[
            'eo' + str(i)
        ] = error_output

        activation_error_cache[
            'ea' + str(i)
        ] = error_activation

    return (
        output_error_cache,
        activation_error_cache
    )


# =============================================================================
# SINGLE LSTM CELL ERROR
# =============================================================================

def calculate_single_lstm_cell_error(
    activation_output_error,
    next_activation_error,
    next_cell_error,
    parameters,
    lstm_activation,
    cell_activation,
    prev_cell_activation
):

    activation_error = (
        activation_output_error
        +
        next_activation_error
    )

    oa = lstm_activation['oa']

    ia = lstm_activation['ia']

    ga = lstm_activation['ga']

    fa = lstm_activation['fa']

    eo = np.multiply(
        np.multiply(
            np.multiply(
                activation_error,
                tanh(
                    cell_activation
                )
            ),
            oa
        ),
        1 - oa
    )

    cell_error = np.multiply(
        np.multiply(
            activation_error,
            oa
        ),
        dtanh(
            tanh(
                cell_activation
            )
        )
    )

    cell_error += next_cell_error

    ei = np.multiply(
        np.multiply(
            np.multiply(
                cell_error,
                ga
            ),
            ia
        ),
        1 - ia
    )

    eg = np.multiply(
        np.multiply(
            cell_error,
            ia
        ),
        dtanh(
            ga
        )
    )

    ef = np.multiply(
        np.multiply(
            np.multiply(
                cell_error,
                prev_cell_activation
            ),
            fa
        ),
        1 - fa
    )

    prev_cell_error = np.multiply(
        cell_error,
        fa
    )

    # Retained exactly as supplied.
    embed_activation_error = np.matmul(
        ef,
        parameters['fgw'].T
    )

    embed_activation_error += np.matmul(
        ei,
        parameters['igw'].T
    )

    embed_activation_error += np.matmul(
        eo,
        parameters['ggw'].T
    )

    embed_activation_error += np.matmul(
        eg,
        parameters['ogw'].T
    )

    input_units = (
        parameters['fgw'].shape[0]
        -
        parameters['fgw'].shape[1]
    )

    prev_activation_error = (
        embed_activation_error[
            :,
            input_units:
        ]
    )

    embed_error = (
        embed_activation_error[
            :,
            :input_units
        ]
    )

    lstm_error = {}

    lstm_error['ef'] = ef
    lstm_error['ei'] = ei
    lstm_error['eo'] = eo
    lstm_error['eg'] = eg

    return (
        prev_activation_error,
        prev_cell_error,
        embed_error,
        lstm_error
    )


# =============================================================================
# BACKWARD PROPAGATION
# =============================================================================

def backward_propagation(
    batch_labels,
    embedding_cache,
    lstm_cache,
    activation_cache,
    cell_cache,
    output_cache,
    parameters
):

    (
        output_error_cache,
        activation_error_cache
    ) = calculate_output_cell_error(
        batch_labels,
        output_cache,
        parameters
    )

    lstm_error_cache = {}

    embedding_error_cache = {}

    eat = np.zeros(
        activation_error_cache[
            'ea1'
        ].shape,
        dtype=np.float32
    )

    ect = np.zeros(
        activation_error_cache[
            'ea1'
        ].shape,
        dtype=np.float32
    )

    for i in range(
        len(lstm_cache),
        0,
        -1
    ):

        (
            pae,
            pce,
            ee,
            le
        ) = calculate_single_lstm_cell_error(

            activation_error_cache[
                'ea' + str(i)
            ],

            eat,

            ect,

            parameters,

            lstm_cache[
                'lstm' + str(i)
            ],

            cell_cache[
                'c' + str(i)
            ],

            cell_cache[
                'c' + str(i - 1)
            ]
        )

        lstm_error_cache[
            'elstm' + str(i)
        ] = le

        embedding_error_cache[
            'eemb' + str(i - 1)
        ] = ee

        eat = pae

        ect = pce

    derivatives = {}

    derivatives[
        'dhow'
    ] = calculate_output_cell_derivatives(
        output_error_cache,
        activation_cache,
        parameters
    )

    lstm_derivatives = {}

    for i in range(
        1,
        len(lstm_error_cache) + 1
    ):

        lstm_derivatives[
            'dlstm' + str(i)
        ] = calculate_single_lstm_cell_derivatives(

            lstm_error_cache[
                'elstm' + str(i)
            ],

            embedding_cache[
                'emb' + str(i - 1)
            ],

            activation_cache[
                'a' + str(i - 1)
            ]
        )

    derivatives['dfgw'] = np.zeros_like(
        parameters['fgw'],
        dtype=np.float32
    )

    derivatives['digw'] = np.zeros_like(
        parameters['igw'],
        dtype=np.float32
    )

    derivatives['dogw'] = np.zeros_like(
        parameters['ogw'],
        dtype=np.float32
    )

    derivatives['dggw'] = np.zeros_like(
        parameters['ggw'],
        dtype=np.float32
    )

    for i in range(
        1,
        len(lstm_error_cache) + 1
    ):

        derivatives['dfgw'] += (
            lstm_derivatives[
                'dlstm' + str(i)
            ]['dfgw']
        )

        derivatives['digw'] += (
            lstm_derivatives[
                'dlstm' + str(i)
            ]['digw']
        )

        derivatives['dogw'] += (
            lstm_derivatives[
                'dlstm' + str(i)
            ]['dogw']
        )

        derivatives['dggw'] += (
            lstm_derivatives[
                'dlstm' + str(i)
            ]['dggw']
        )

    return (
        derivatives,
        embedding_error_cache
    )


# =============================================================================
# OUTPUT CELL DERIVATIVES
# =============================================================================

def calculate_output_cell_derivatives(
    output_error_cache,
    activation_cache,
    parameters
):

    dhow = np.zeros_like(
        parameters['how'],
        dtype=np.float32
    )

    batch_size = activation_cache[
        'a1'
    ].shape[0]

    for i in range(
        1,
        len(output_error_cache) + 1
    ):

        output_error = (
            output_error_cache[
                'eo' + str(i)
            ]
        )

        activation = (
            activation_cache[
                'a' + str(i)
            ]
        )

        dhow += (
            np.matmul(
                activation.T,
                output_error
            )
            /
            batch_size
        )

    return dhow


# =============================================================================
# SINGLE LSTM CELL DERIVATIVES
# =============================================================================

def calculate_single_lstm_cell_derivatives(
    lstm_error,
    embedding_matrix,
    activation_matrix
):

    concat_matrix = np.concatenate(
        (
            embedding_matrix,
            activation_matrix
        ),
        axis=1
    )

    batch_size = embedding_matrix.shape[0]

    derivatives = {}

    derivatives['dfgw'] = (
        np.matmul(
            concat_matrix.T,
            lstm_error['ef']
        )
        /
        batch_size
    )

    derivatives['digw'] = (
        np.matmul(
            concat_matrix.T,
            lstm_error['ei']
        )
        /
        batch_size
    )

    derivatives['dogw'] = (
        np.matmul(
            concat_matrix.T,
            lstm_error['eo']
        )
        /
        batch_size
    )

    derivatives['dggw'] = (
        np.matmul(
            concat_matrix.T,
            lstm_error['eg']
        )
        /
        batch_size
    )

    return derivatives


# =============================================================================
# UPDATE PARAMETERS
# =============================================================================

def update_parameters(
    parameters,
    derivatives,
    V,
    S
):

    vfgw = (
        BETA1
        *
        V['vfgw']
        +
        (
            1 - BETA1
        )
        *
        derivatives['dfgw']
    )

    vigw = (
        BETA1
        *
        V['vigw']
        +
        (
            1 - BETA1
        )
        *
        derivatives['digw']
    )

    vogw = (
        BETA1
        *
        V['vogw']
        +
        (
            1 - BETA1
        )
        *
        derivatives['dogw']
    )

    vggw = (
        BETA1
        *
        V['vggw']
        +
        (
            1 - BETA1
        )
        *
        derivatives['dggw']
    )

    vhow = (
        BETA1
        *
        V['vhow']
        +
        (
            1 - BETA1
        )
        *
        derivatives['dhow']
    )


    sfgw = (
        BETA2
        *
        S['sfgw']
        +
        (
            1 - BETA2
        )
        *
        derivatives['dfgw'] ** 2
    )

    sigw = (
        BETA2
        *
        S['sigw']
        +
        (
            1 - BETA2
        )
        *
        derivatives['digw'] ** 2
    )

    sogw = (
        BETA2
        *
        S['sogw']
        +
        (
            1 - BETA2
        )
        *
        derivatives['dogw'] ** 2
    )

    sggw = (
        BETA2
        *
        S['sggw']
        +
        (
            1 - BETA2
        )
        *
        derivatives['dggw'] ** 2
    )

    show = (
        BETA2
        *
        S['show']
        +
        (
            1 - BETA2
        )
        *
        derivatives['dhow'] ** 2
    )


    parameters['fgw'] -= (
        LEARNING_RATE
        *
        (
            vfgw
            /
            (
                np.sqrt(
                    sfgw
                )
                +
                1e-6
            )
        )
    )

    parameters['igw'] -= (
        LEARNING_RATE
        *
        (
            vigw
            /
            (
                np.sqrt(
                    sigw
                )
                +
                1e-6
            )
        )
    )

    parameters['ogw'] -= (
        LEARNING_RATE
        *
        (
            vogw
            /
            (
                np.sqrt(
                    sogw
                )
                +
                1e-6
            )
        )
    )

    parameters['ggw'] -= (
        LEARNING_RATE
        *
        (
            vggw
            /
            (
                np.sqrt(
                    sggw
                )
                +
                1e-6
            )
        )
    )

    parameters['how'] -= (
        LEARNING_RATE
        *
        (
            vhow
            /
            (
                np.sqrt(
                    show
                )
                +
                1e-6
            )
        )
    )


    V['vfgw'] = vfgw
    V['vigw'] = vigw
    V['vogw'] = vogw
    V['vggw'] = vggw
    V['vhow'] = vhow

    S['sfgw'] = sfgw
    S['sigw'] = sigw
    S['sogw'] = sogw
    S['sggw'] = sggw
    S['show'] = show

    return (
        parameters,
        V,
        S
    )


# =============================================================================
# UPDATE EMBEDDINGS
# =============================================================================

def update_embeddings(
    embeddings,
    embedding_error_cache,
    batch_labels
):

    embedding_derivatives = np.zeros_like(
        embeddings,
        dtype=np.float32
    )

    batch_size = batch_labels[
        0
    ].shape[0]

    for i in range(
        len(
            embedding_error_cache
        )
    ):

        embedding_derivatives += (
            np.matmul(
                batch_labels[i].T,
                embedding_error_cache[
                    'eemb' + str(i)
                ]
            )
            /
            batch_size
        )

    embeddings = (
        embeddings
        -
        LEARNING_RATE
        *
        embedding_derivatives
    )

    return embeddings


# =============================================================================
# MEMORY-SAFE BATCH CREATION
# =============================================================================

PAD_EVENT = (
    None,
    None,
    None,
    None
)


def make_window_batch(
    songs,
    data_int,
    song_start,
    time_start
):

    """
    Create only ONE short dense one-hot sequence.

    Original get_batches() created every dense timestep for every possible
    batch and retained all of them.

    This function creates only SEQUENCE_LENGTH + 1 timesteps for the
    current iteration.
    """

    batch_data = songs[
        song_start:
        song_start + BATCH_SIZE
    ]

    if len(
        batch_data
    ) != BATCH_SIZE:

        return None

    note_list = []

    for j in range(
        time_start,
        time_start + SEQUENCE_LENGTH + 1
    ):

        batch_dataset = np.zeros(
            [
                BATCH_SIZE,
                len(data_int)
            ],
            dtype=np.float32
        )

        for k in range(
            BATCH_SIZE
        ):

            current_note = (
                batch_data[k][j]
            )

            idx = data_int[
                current_note
            ]

            batch_dataset[
                k,
                idx
            ] = 1.0

        note_list.append(
            batch_dataset
        )

    return note_list


def window_has_real_targets(
    songs,
    song_start,
    time_start
):

    """
    Prevent the bogus:
        Loss = 0
        Perplexity = inf

    observed earlier when a training window fell entirely inside padded
    portions of the songs.
    """

    batch_data = songs[
        song_start:
        song_start + BATCH_SIZE
    ]

    for k in range(
        len(batch_data)
    ):

        for j in range(
            time_start + 1,
            time_start + SEQUENCE_LENGTH + 1
        ):

            if (
                batch_data[k][j]
                !=
                PAD_EVENT
            ):

                return True

    return False


# =============================================================================
# BUILD LIST OF VALID WINDOW POSITIONS
# =============================================================================

def get_valid_windows(
    songs
):

    valid_windows = []

    if len(songs) < BATCH_SIZE:

        return valid_windows

    max_time = len(
        songs[0]
    )

    # Use non-overlapping sequence windows.
    # This keeps preprocessing fast and avoids creating unnecessary
    # duplicate windows.

    for song_start in range(
        0,
        len(songs) - BATCH_SIZE + 1,
        BATCH_SIZE
    ):

        for time_start in range(
            0,
            max_time - SEQUENCE_LENGTH - 1,
            SEQUENCE_LENGTH
        ):

            if window_has_real_targets(
                songs,
                song_start,
                time_start
            ):

                valid_windows.append(
                    (
                        song_start,
                        time_start
                    )
                )

    return valid_windows


# =============================================================================
# TRAIN
# =============================================================================

def train(
    songs,
    data_int,
    iters
):

    parameters = initialize_parameters()

    V = initialize_V(
        parameters
    )

    S = initialize_S(
        parameters
    )

    embeddings = np.random.normal(
        0,
        0.01,
        (
            vocab_size,
            INPUT
        )
    ).astype(
        np.float32
    )

    L = []

    P = []

    A = []


    print()
    print(
        "Finding valid training windows..."
    )

    valid_windows = get_valid_windows(
        songs
    )

    print(
        "Valid training windows:",
        len(
            valid_windows
        )
    )


    if len(
        valid_windows
    ) == 0:

        raise RuntimeError(
            "No valid training windows were found."
        )


    print()
    print(
        "=" * 70
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        "Sequence length:",
        SEQUENCE_LENGTH
    )

    print(
        "Iterations:",
        iters
    )

    print()


    for step in range(
        iters
    ):

        window_index = (
            step
            %
            len(
                valid_windows
            )
        )

        (
            song_start,
            time_start
        ) = valid_windows[
            window_index
        ]

        batches = make_window_batch(
            songs,
            data_int,
            song_start,
            time_start
        )


        # ================================================================
        # FORWARD PROPAGATION
        # ================================================================

        (
            embedding_cache,
            lstm_cache,
            activation_cache,
            cell_cache,
            output_cache
        ) = forward_propagation(
            batches,
            parameters,
            embeddings
        )


        # ================================================================
        # ORIGINAL LOSS / PERPLEXITY / ACCURACY
        # ================================================================

        (
            perplexity,
            loss,
            acc
        ) = cal_loss_accuracy(
            batches,
            output_cache
        )


        # ================================================================
        # ORIGINAL BACKPROPAGATION
        # ================================================================

        (
            derivatives,
            embedding_error_cache
        ) = backward_propagation(
            batches,
            embedding_cache,
            lstm_cache,
            activation_cache,
            cell_cache,
            output_cache,
            parameters
        )


        # ================================================================
        # ORIGINAL ADAM UPDATE
        # ================================================================

        (
            parameters,
            V,
            S
        ) = update_parameters(
            parameters,
            derivatives,
            V,
            S
        )


        # ================================================================
        # ORIGINAL EMBEDDING UPDATE
        # ================================================================

        embeddings = update_embeddings(
            embeddings,
            embedding_error_cache,
            batches
        )


        L.append(
            loss
        )

        P.append(
            perplexity
        )

        A.append(
            acc
        )


        # Print more often so we can see whether it is alive.
        if step % 10 == 0:

            print(
                "For Single Batch :"
            )

            print(
                "Step       = {}".format(
                    step
                )
            )

            print(
                "Loss       = {}".format(
                    round(
                        float(
                            loss
                        ),
                        2
                    )
                )
            )

            print(
                "Perplexity = {}".format(
                    round(
                        float(
                            perplexity
                        ),
                        2
                    )
                )
            )

            print(
                "Accuracy   = {}".format(
                    round(
                        float(
                            acc
                        )
                        *
                        100,
                        2
                    )
                )
            )

            print()


        # Explicitly release the largest temporary arrays.
        del batches

        del embedding_cache

        del lstm_cache

        del activation_cache

        del cell_cache

        del output_cache

        del derivatives

        del embedding_error_cache


    return (
        embeddings,
        parameters,
        L,
        P,
        A
    )


# =============================================================================
# TRAIN MODEL
# =============================================================================

embeddings, parameters, L, P, A = train(
    songs_data,
    data_to_int,
    1000
)


# =============================================================================
# PLOTS
# =============================================================================

avg_loss = []

avg_acc = []

avg_perp = []

i = 0


while i < len(L):

    avg_loss.append(
        np.mean(
            L[
                i:
                i + 50
            ]
        )
    )

    avg_acc.append(
        np.mean(
            A[
                i:
                i + 50
            ]
        )
    )

    avg_perp.append(
        np.mean(
            P[
                i:
                i + 50
            ]
        )
    )

    i += 50


plt.plot(
    list(
        range(
            len(
                avg_loss
            )
        )
    ),
    avg_loss
)

plt.xlabel(
    "Iteration of 50 batches"
)

plt.ylabel(
    "Average Loss"
)

plt.title(
    "Average Loss of Each 50 Batches"
)

plt.show()


plt.plot(
    list(
        range(
            len(
                avg_perp
            )
        )
    ),
    avg_perp
)

plt.xlabel(
    "Iteration of 50 batches"
)

plt.ylabel(
    "Average Perplexity"
)

plt.title(
    "Average Perplexity of Each 50 Batches"
)

plt.show()


plt.plot(
    list(
        range(
            len(
                avg_acc
            )
        )
    ),
    avg_acc
)

plt.xlabel(
    "Iteration of 50 batches"
)

plt.ylabel(
    "Average Accuracy"
)

plt.title(
    "Average Accuracy of Each 50 Batches"
)

plt.show()


# =============================================================================
# PART 3
# PREDICT
# =============================================================================

def predict(
    parameters,
    embeddings,
    idx2note,
    vocab_size
):

    out_notes = []

    # Produce 10 pieces of music.
    for i in range(
        10
    ):

        print(
            "Generating piece {}/10...".format(
                i + 1
            )
        )

        a0 = np.zeros(
            [
                1,
                HIDDEN
            ],
            dtype=np.float32
        )

        c0 = np.zeros(
            [
                1,
                HIDDEN
            ],
            dtype=np.float32
        )

        notes = []

        batch_dataset = np.zeros(
            [
                1,
                vocab_size
            ],
            dtype=np.float32
        )

        # Original random start note.
        index = np.random.randint(
            0,
            vocab_size,
            1
        )[0]

        batch_dataset[
            0,
            index
        ] = 1.0

        notes.append(
            idx2note[
                index
            ]
        )

        note_value = idx2note[
            index
        ]

        # Original termination criterion retained.
        #
        # A safety limit is added only to prevent an infinite generation
        # loop if PAD is never sampled.

        generated_steps = 0

        MAX_GENERATED_STEPS = 1000

        while (
            note_value
            !=
            (
                None,
                None,
                None,
                None
            )
            and
            generated_steps
            <
            MAX_GENERATED_STEPS
        ):

            batch_dataset = get_embeddings(
                batch_dataset,
                embeddings
            )

            (
                lstm_activations,
                ct,
                at
            ) = lstm_cell(
                batch_dataset,
                a0,
                c0,
                parameters
            )

            ot = output_cell(
                at,
                parameters
            )

            pred = np.random.choice(
                vocab_size,
                1,
                p=ot[0]
            )[0]

            notes.append(
                idx2note[
                    pred
                ]
            )

            note_value = idx2note[
                pred
            ]

            batch_dataset = np.zeros(
                [
                    1,
                    vocab_size
                ],
                dtype=np.float32
            )

            batch_dataset[
                0,
                pred
            ] = 1.0

            a0 = at

            c0 = ct

            generated_steps += 1


        out_notes.append(
            notes
        )


    return out_notes


# =============================================================================
# GENERATE MUSIC
# =============================================================================

pred_notes = predict(
    parameters,
    embeddings,
    int_to_data,
    vocab_size
)


# =============================================================================
# WRITE GENERATED SONGS AS MIDI
# =============================================================================

count = 0


for piece in pred_notes:

    p1 = stream.Part()

    p1.insert(
        0,
        instrument.Piano()
    )

    p2 = stream.Part()

    p2.insert(
        0,
        instrument.Piano()
    )


    for item in piece:

        if item != (
            None,
            None,
            None,
            None
        ):

            if (
                item[0] != None
                and
                item[1] != None
            ):

                if '.' in item[0]:

                    chord_pitches = (
                        item[0].split(
                            '.'
                        )
                    )

                    p1.append(
                        chord.Chord(
                            chord_pitches,
                            quarterLength=item[1]
                        )
                    )

                else:

                    p1.append(
                        note.Note(
                            item[0],
                            quarterLength=item[1]
                        )
                    )


            if (
                item[2] != None
                and
                item[3] != None
            ):

                if '.' in item[2]:

                    chord_pitches = (
                        item[2].split(
                            '.'
                        )
                    )

                    p2.append(
                        chord.Chord(
                            chord_pitches,
                            quarterLength=item[3]
                        )
                    )

                else:

                    p2.append(
                        note.Note(
                            item[2],
                            quarterLength=item[3]
                        )
                    )


    s = stream.Stream(
        [
            p1,
            p2
        ]
    )


    mid = midi.translate.streamToMidiFile(
        s
    )


    mid.open(
        'out'
        +
        str(
            count
        )
        +
        '.mid',
        'wb'
    )


    mid.write()

    mid.close()


    print(
        "saved generated song! check your directory."
    )


    count += 1


print()
print(
    "=" * 70
)

print(
    "LSTM RUN COMPLETE"
)

print(
    "=" * 70
)

print(
    "Generated MIDI files: out0.mid to out9.mid"
)