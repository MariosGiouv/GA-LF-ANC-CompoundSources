"""
This script is integrated with a proposed ANC system for low frequencies in closed spaces, using the ‘PyGAD’package.
The secondary, or control, source used is a compound source, a quadrupole, consisting of two dipoles, which consist of
small subwoofers.
The target of this strategy is to adapt the compound source radiation and coupling to the modal field through its
driving parameters to attenuate the primary field at a selected point; the minimization of the primary field using
the norm  of sound pressure at a discrete point of measurement by utilizing the partial sound destructive interference.
The process of selecting the optimal driving parameters is followed each time for a given configuration. The fitness of
each individual is assessed as the difference between the existing noise level, and the controlled level after applying
the generated i-th driving parameters for the compound source.
"""

import pygad
from scipy import signal
import sounddevice as sd
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy.io.wavfile import write, read
import pylab


def generate_init_ampl_phase(d_amplitude, d_phase_range, phase):
    """
    TODO Generates the initial population, init_ampl_phase array, with random values
    :param d_amplitude:
    :param d_phase_range:
    :param phase:
    :return:
    """
    # random_ints = np.random.randint(1, d_amplitude, size=(d_amplitude - 1, 2))
    # random_phase = np.random.randint(0, d_phase_range, size=d_amplitude - 1) * phase
    random_ints = np.random.randint(1, d_amplitude, size=(40, 2))
    random_phase = np.random.randint(0, d_phase_range, size=40) * phase
    return np.column_stack((random_ints, random_phase)).astype(float)


def play_rec_fun(sol):
    """
    TODO Plays and records audio with the given amplitude and phase values
    :param sol:
    :return:
    """
    print('---->> Recording Audio ampl, ampl2, phase = {}, {}, {}'.format(sol[0], sol[1], sol[2] * 180 / np.pi))
    Lchannel = 0.2 * (10 ** (sol[0] / 20)) * np.sin(2 * np.pi * L_freq * time + 0)
    Rchannel = 0.2 * (10 ** (sol[1] / 20)) * np.sin(2 * np.pi * R_freq * time + sol[2])
    stereo_sine = np.vstack((Lchannel, Rchannel))
    stereo_array = stereo_sine.transpose()
    my_recording = sd.playrec(stereo_array, fs, blocking=True)
    my_recording = np.squeeze(my_recording)
    my_resampl_filt_rec = signal.decimate(my_recording, down_sample, n=None, ftype='fir', axis=-1, zero_phase=True)
    nyq_rate = fs_new / 2.0
    width = 5.0 / nyq_rate
    ripple_db = 60.0
    N, beta = kaiserord(ripple_db, width)
    cutoff_hz = [frequencies[0] / nyq_rate, frequencies[2] / nyq_rate]
    taps = firwin(N, cutoff_hz, window=('kaiser', beta), pass_zero=False)
    my_filt_recording = lfilter(taps, 1.0, my_resampl_filt_rec)
    return_val_range = my_filt_recording[int(fs_new / 4):]
    return return_val_range


def get_db(pressures):
    """
    TODO Calculates the sound level in decibels
    :param pressures:
    :return:
    """
    sq_pressures = np.square(pressures)
    aver_sqrd = (sum(sq_pressures) / len(pressures))
    level = 10.0 * np.log10(aver_sqrd)
    print('Level of efficient recorded range:', "{:.1f}".format(level))
    return level


def fitness_func(sol, solution_idx):
    """
    TODO Calculates the fitness of the given solution
    :param sol:
    :param solution_idx:
    :return:
    """
    if abs(sol[0] - sol[1] <= 9):
        fitness = noise_level - get_db((play_rec_fun(sol)))
        print('fitness is:', "{:.1f}".format(fitness))
    else:
        fitness = -1000
        print('The two amplitudes differ more than 9.5 dB! : {}, {}'.format(sol[0], sol[1]))
        print('fitness is:', "{:.1f}".format(fitness))
    return fitness


def on_generation(ga_instance):
    """
    :param ga_instance:
    :return:
    """
    global last_fitness
    print("Generation = ", "{:.1f}".format(ga_instance.generations_completed))
    print('Fitness  =', "{:.1f}".format(ga_instance.best_solution()[1]))
    print("last_gen_fitness = {last_gen_fitness}".format(last_gen_fitness=ga_instance.last_generation_fitness))
    print("Change = ", "{:.1f}".format(ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]


if __name__ == '__main__':

    # TODO
    fs = 44100
    down_sample = 10
    fs_new = int(fs / down_sample)
    seconds = 1
    duration = seconds * fs
    time = np.linspace(0, seconds, seconds * fs, False)
    L_freq = 100
    R_freq = 100
    frequencies = [L_freq - 20, L_freq, L_freq + 20]
    pha = np.pi/4
    dipole_amplitude = 9
    dipole_phase_range = 7
    init_ampl_phase = generate_init_ampl_phase(dipole_amplitude, dipole_phase_range, pha)

    popsize = len(init_ampl_phase)
    print(popsize)
    print(sd.query_devices())
    sd.default.device = 1, 3  # Insert audio I/O tags from query_devices()
    sd.default.dtype = [None, 'float64']
    sd.default.channels = 1, 2

    h, w = 2, 1
    pylab.figure(figsize=(12, 9))
    pylab.subplots_adjust(hspace=.7)

    try:
        print('---->> Recording Noise....')     # noise SPL measurement
        noise_recording = sd.rec((1 * duration))
        sd.wait()
        noise_recording = np.squeeze(noise_recording)

        # setting filtering params
        nyq_rate = fs / 2.0
        width = 5.0 / nyq_rate
        ripple_db = 60.0
        N, beta = kaiserord(ripple_db, width)
        cutoff_hz = [frequencies[0] / nyq_rate, frequencies[2] / nyq_rate]
        taps = firwin(N, cutoff_hz, window=('kaiser', beta), pass_zero=False)

        # filtering mic input noise
        filtered_noise = lfilter(taps, 1.0, noise_recording)
        noise_range = filtered_noise[int(fs_new / 4):]
        noise_level = get_db(noise_range)
        print('Noise level:', "{:.1f}".format(noise_level))

        # TODO .......................... GA ................................................................
        fitness_function = fitness_func
        num_generations = 40
        num_parents_mating = 28
        gene_space = np.arange(1, 9.5, 0.5), np.arange(1, 9.5, 0.5), (0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4,
                                                                      6*np.pi/4, 7*np.pi/4)
        parent_selection_type = "rank"
        keep_parents = 4
        crossover_type = "single_point"
        crossover_probability = 0.75
        mutation_type = "random"
        mutation_probability = 0.15
        # mutation_type = "adaptive"
        # mutation_probability = [0.3, 0.1]
        last_fitness = 0

        # TODO
        ga_instance = pygad.GA(initial_population=init_ampl_phase,
                               num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               gene_space=gene_space,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_probability=mutation_probability,
                               mutation_type=mutation_type,
                               on_generation=on_generation,
                               save_best_solutions=True)

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()
        ga_instance.plot_result()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution :", "{:.1f}".format(solution_fitness))
        print("Index of the best solution :", "{:.1f}".format(solution_idx))

        if ga_instance.best_solution_generation != -1:
            print("Best fitness value reached after generations ",
                  "{:.1f}".format(ga_instance.best_solution_generation))

        # Saving the GA instance.
        filename = 'genetic'
        ga_instance.save(filename=filename)

        # Loading the saved GA instance.
        loaded_ga_instance = pygad.load(filename=filename)
        loaded_ga_instance.plot_result()

    except Exception as e:
        print(e)
        input()