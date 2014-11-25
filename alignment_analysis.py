import matplotlib.pyplot as plt
import librosa
import numpy as np


def plot_best_path(similarity_matrix, p, q):
    '''
    Plots the similarity matrix with the best path on the current axis.

    :parameters:
        - similarity_matrix : np.ndarray
            Pairwise similarity of two signals
        - p : np.ndarray
            first-axis indices of best path through matrix
        - q : np.ndarray
            second-axis indices of best path through matrix
    '''
    plt.imshow(similarity_matrix.T,
               aspect='auto',
               interpolation='nearest',
               cmap=plt.cm.gray)
    tight = plt.axis()
    plt.plot(p, q, 'r.', ms=.2)
    plt.xlabel('MIDI beat')
    plt.ylabel('Audio beat')
    plt.title('Similarity matrix and best cost path')
    plt.axis(tight)


def plot_gram(gram):
    '''
    Plots a *gram (cqt-gram, pian roll-gram).

    :parameters:
        - gram : np.ndarray
            A 2-d representation of time/frequency, with frequencies being the
            notes between MIDI note 36 and 96.
    '''
    librosa.display.specshow(gram,
                             x_axis='frames',
                             y_axis='cqt_note',
                             fmin=librosa.midi_to_hz(36),
                             fmax=librosa.midi_to_hz(96))


def plot_path_diff(p, q):
    '''
    Plot the beat index shift from p to q.

    :parameters:
        - p : np.ndarray
            first-axis indices of best path through matrix
        - q : np.ndarray
            second-axis indices of best path through matrix
    '''
    plt.plot(p - q)
    plt.xlabel('Index')
    plt.ylabel('Index shift')
    plt.title('Path difference')


def plot_path_cost(similarity_matrix, p, q):
    '''
    Plot the cost along the best-cost path.

    :parameters:
        - similarity_matrix : np.ndarray
            Pairwise similarity of two signals
        - p : np.ndarray
            first-axis indices of best path through matrix
        - q : np.ndarray
            second-axis indices of best path through matrix
    '''
    plt.plot([similarity_matrix[p_v, q_v] for p_v, q_v in zip(p, q)])
    plt.xlabel('Index')
    plt.ylabel('Cost')
    plt.title('Cost along best cost path')


def plot_diagnostics(audio_gram, midi_gram, similarity_matrix, p, q):
    '''
    Plots all diagnostic plots.

    :parameters:
        - audio_gram : np.ndarray
            CQT of the audio
        - midi_gram : np.ndarray
            MIDI piano roll
        - similarity_matrix : np.ndarray
            Pairwise similarity of two signals
        - p : np.ndarray
            first-axis indices of best path through matrix
        - q : np.ndarray
            second-axis indices of best path through matrix
    '''
    plt.subplot2grid((4, 3), (0, 0), colspan=3)
    plot_gram(audio_gram)
    plt.title('Audio CQT')
    plt.subplot2grid((4, 3), (1, 0), colspan=3)
    plot_gram(midi_gram)
    plt.title('MIDI piano roll')
    plt.subplot2grid((4, 3), (2, 0), rowspan=2)
    plot_path_cost(similarity_matrix, p, q)
    plt.subplot2grid((4, 3), (2, 1), rowspan=2)
    plot_best_path(similarity_matrix, p, q)
    plt.subplot2grid((4, 3), (2, 2), rowspan=2)
    plot_path_diff(p, q)


def synthesize_aligned_midi(audio, fs, m_aligned):
    '''
    Synthesize aligned MIDI data and return it in one channel with the audio
    in the other channel.

    :parameters:
        - audio : np.ndarray
            Audio data array.
        - fs : int
            Sampling rate of the audio data, and to use for synthesis
        - m_aligned : pretty_midi.PrettyMIDI
            Aligned midi data to synthesize.
    :returns:
        - midi_and_audio : np.ndarray
            Stacked stereo wav array of synthesized midi + audio
    '''
    midi_audio_aligned = m_aligned.fluidsynth(fs)
    # Adjust to the same size as audio
    if midi_audio_aligned.shape[0] > audio.shape[0]:
        midi_audio_aligned = midi_audio_aligned[:audio.shape[0]]
    else:
        trim_amount = audio.shape[0] - midi_audio_aligned.shape[0]
        midi_audio_aligned = np.append(midi_audio_aligned,
                                       np.zeros(trim_amount))
    # Stack one in each channel
    return [midi_audio_aligned, audio]


def get_scores(similarity_matrix, p, q, score):
    '''
    Computes alignment scores for judging the quality of the alignment.

    :parameters:
        - similarity_matrix : np.ndarray
            Pairwise similarity of two signals
        - p : np.ndarray
            first-axis indices of best path through matrix
        - q : np.ndarray
            second-axis indices of best path through matrix
        - score : float
            Score of best-cost path using additive penalty

    :returns:
        - score_with_penalty : float
            Best-path score with additive penalty for non-diagonal moves
        - score_no_penalty : float
            Raw best-path score with no penalties
        - normalized_score_with_penalty : float
            Best-path score, normalized by average distance
        - normalized_score_no_penalty : float
            No-penalty best-path score, normalized by average distance
        - percent_non_diagonal : float
            Percentage of non-diagonal steps taken
    '''
    score_no_penalty = np.mean(similarity_matrix[p, q])
    percent_non_diagonal = np.abs(np.diff(p - q)).mean()
    norm = similarity_matrix.mean()
    return (score, score_no_penalty, score/norm, score_no_penalty/norm,
            percent_non_diagonal)
