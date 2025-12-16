## Signal Analysis and Visualization Results

### 1. Data Loading and Verification

The first step of the analysis was to verify that the software correctly ingests the experimental EEG data.  
The system reads CSV files containing electrode recordings from the channels F3, F4, C3, C4, P3, P4, Cz, and Pz together with their corresponding timestamps.

The first visualization shows the raw data view of the software interface. The table confirms that numerical amplitude values for each channel were successfully parsed and organized according to time in milliseconds. This structured representation prepares the data for subsequent processing stages.

---

### 2. Signal Preprocessing

To isolate neural activity from environmental noise, digital filtering was applied. A 50 Hz notch filter was used to suppress power line interference, followed by a band-pass filter between 5 and 35 Hz.

The second visualization illustrates the filtered time-domain signals for the C3 and C4 motor cortex channels. A short high-amplitude transient is visible at the beginning of the recording; however, the signal stabilizes rapidly. This behavior indicates that baseline drift and external noise were effectively removed, resulting in a clean signal suitable for further analysis.

---

### 3. Band Power Comparison

To evaluate the distribution of signal energy across frequency bands, the Power Spectral Density was computed. The analysis covers the Delta, Theta, Alpha/Mu, Beta, and Gamma bands.

The third visualization presents a comparison of total band power values for the C3 and C4 electrodes. A clear difference in absolute power is observed between the left and right hemispheres, particularly in the Alpha/Mu band. This difference provides a reference point for assessing hemispheric lateralization.

---

### 4. Temporal Evolution of the Mu Rhythm

While total band power provides a summary, observing how neural rhythms evolve over time is critical for motor-related tasks. The fourth visualization tracks the Mu band power between 8 and 12 Hz across the duration of the experiment.

After the initial filter stabilization, the C3 channel consistently exhibits lower Mu power compared to C4. This sustained suppression of Mu activity in the left hemisphere aligns with expected neural patterns associated with right-hand motor control.

---

### 5. Event-Related Desynchronization Analysis

To quantify task-related changes in brain activity, the software calculates percentage power changes relative to a resting baseline. A baseline window from 1.0 to 2.0 seconds and an active task window from 2.0 to 3.0 seconds were defined.

The fifth visualization shows a strong negative power change for the C4 electrode, indicating event-related desynchronization during the task period. In contrast, the C3 electrode shows a slight positive change, corresponding to event-related synchronization. This comparison allows raw signal variations to be expressed as interpretable percentage values.

---

### 6. Time-Frequency Spectral Analysis

To analyze signal behavior in both time and frequency domains simultaneously, short-time Fourier transform spectrograms were generated.

The sixth visualization presents heatmaps for channels C3 and C4. Higher energy levels are concentrated in the lower frequency ranges, while power decreases at higher frequencies. This confirms the spectral consistency of the filtered EEG signals.

---

### 7. Topographic Distribution of Brain Activity

Finally, the spatial distribution of band power across the scalp was visualized using two-dimensional topographic mapping.

The seventh visualization displays the Delta band distribution. Higher activity is concentrated around the central and posterior regions, including Cz, P3, and P4, while frontal and lateral areas show lower activation levels. This spatial representation provides a holistic view of brain activity beyond single-channel analysis.
