### GA_LowFrequency_ANC_CompoundSources

#### GA-based ANC system with compound low-frequency sources

This script is integrated with a proposed ANC system for low frequencies in closed spaces, using the *‘PyGAD’*
package [1]. The secondary or control source used is a compound source, a quadrupole, consisting of two dipoles, which 
consist of small sub-woofer loudspeakers with *low-Bl* (force factor) drivers [2]-[4].

The target of this strategy is to adapt the compound source radiation and coupling to the modal field through its
driving parameters to attenuate the primary field at a selected point; the minimization of the primary field using
the norm  of sound pressure at a discrete point of measurement by utilizing the partial sound destructive interference
[5]-[6]. The process of selecting the optimal driving parameters is followed each time for a given configuration. 
The fitness of each individual is assessed as the difference between the existing noise level, and the controlled level
after applying the generated i-th driving parameters for the compound source.

Each chromosome is a set of positive real numbers, which are the three system parameters, two for the driving 
monopole amplitudes of each dipole (*d_amplitude*), and one for their phase difference (*phase*). The two 
amplitudes are levels in dB, relative to the initial driving amplitude of the compound source. This is set so that 
the secondary sound level at the measurement point is equal to the primary level, as first recorded in 
(*Recording Noise*). The *gene_space* provides a 90deg step for phase. Difference of the two amplitudes >9.5 dB is
not acceptable as it will degenerate the quadrupole radiation features.

The first 2 channels are dedicated to control signals, which play simultaneously with a subwoofer simulating the
noise source, the third channel. A 1/3-octave band filter is applied to the microphone input signal, to not 
disorientate the evolution convergence in the presence of a possible event of external sound disturbance.

*Note 1:* There is no need for a reference signal or correlation with the error signal. The proposed ANC system is
independent of transfer functions between sources and error sensors.

*Note 2:* If more control sources are added, then the corresponding control signals should be given in
*play_rec_fun()*.

*Note 3:* You can use conventional sub-woofers in multipole/compound configurations.

*Note 4:* The *print(sd.query_devices())* shows the audio [I, O] tags that should be put in *sd.default.device*.  

*Note 5:* The *ampl* and *ampl_noise* are set so that the recording level at the measurement point is equal, at the
 beginning.

*Note 6:* From query_devices(), insert the proper audio stream I/O tags of your system. The two tags are the same for
 ASIO drivers


Special thanks to Sotirios Tsakalidis (linkedin.com/in/sotirios-tsakalidis)for his most valuable contribution.


[1] *Gad, A.F. PyGAD: An Intuitive Genetic Algorithm Python Library. arXiv 2021, arXiv:2106.06158.
     doi: 553 org/10.48550/arXiv.2106.06158*
 
[2] *Aarts, R.O.M. High-Efficiency Low-Bl Loudspeakers. J. Audio Eng. Soc. 2005, 53, 7/8, 579–592.*

[3] *Giouvanakis, M.; Kasidakis, K.; Sevastiadis, C.; Papanikolaou, G. Design and Construction of Loudspeakers with 
     Low-Bl Driv-519 ers for Low-Frequency Active Noise Control Applications. In Proceedings of the 23rd ICA, Aachen, 
     Germany, 9-13 Sept. 2019. 520*

[4] *Giouvanakis, M.; Sevastiadis, C.; Papanikolaou, G. Measurement of Compound Sound Sources with Adaptive Spatial
     Radiation 521 for Low-Frequency Active Noise Control Applications. Arch. Acoust. 2021, 46, 2, 205–212.
     doi: 10.24425/aoa.2021.136576. 522*
 
[5] *Giouvanakis, M.; Sevastiadis, C.; Vrysis, L.; Papanikolaou, G. Control of Resonant Low-Frequency Noise Simulations
     in Differ-515 ent Areas of Small Spaces Using Compound Source. In Proceedings of the Euronoise Conference, Crete, 
     Greece, 27-31 May 2018. 516.*
  
[6] *Giouvanakis, M.; Sevastiadis, C.; Papanikolaou, G. Low-Frequency Noise Attenuation in a Closed Space using 
     Adaptive Di-517 rectivity Control Sources of a Quadrupole Type. Arch. Acoust. 2019, 44, 1, 71–78. 
     doi: 10.24425/aoa.2019.126353.*
