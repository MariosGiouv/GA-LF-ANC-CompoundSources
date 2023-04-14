# GA LowFrequency ANC CompoundSources
GA-based ANC system with compound low-frequency sources

This script is integrated with a proposed ANC system for low frequencies in closed spaces [], using the ‘PyGAD’
package []. The secondary or control source used is a compound source, consisting of two dipoles, which consist of
low-volume sub-woofer loudspeakers with low-Bl(force factor) drivers. Therefore, the control source is a quadrupole.
Each individual/chromosome is a set of positive real numbers, which are the three system parameters, two for the driving
monopole amplitudes of each dipole (d_amplitude), and one for their phase difference (phase).

The two amplitudes are levels in dB, relative to the initial driving amplitude of the compound source. This is set so
that the secondary sound level at the measurement point is equal to the primary level, as first recorded in
(Recording Noise)
The gene_space is {(1, 9.5, 0.5), (1, 9.5, 0.5), (0, 3*pi/2, pi/4)}, with 90deg step for phase and difference of the
two amplitudes more than 9.5 dB is not acceptable as it will degenerate the quadrupole radiation features.

The target of this strategy is to adapt the compound source radiation and coupling to the modal field through its
driving parameters to attenuate the primary field at a selected point; the minimization of the primary field using
the norm  of sound pressure at a discrete point of measurement by utilizing the partial sound destructive interference.
The process of selecting the optimal driving parameters is followed each time for a given configuration. The fitness
of each individual is assessed as the difference between the existing noise level,L_n, and the controlled level
after applying the generated i-th driving parameters for the compound source,L_{i,g}: f_{i,g}=L_n - L_{i,g}

The first 2 channels are dedicated the control signals, which play simultaneously with a subwoofer simulating the noise
source, the third channel. A 1/3-octave band filter is applied to the microphone input signal, to not disorientate the
evolution convergence in the presence of a possible event of external temporary sound disturbance.

Note1: There is no need for a reference signal or correlation with the error signal. The proposed ANC system is
independent of transfer functions between sources and error sensors.

Note2: If more control sources are added, then the corresponding control signals should be given in play_rec_fun().
