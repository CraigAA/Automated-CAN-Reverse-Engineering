# Automated CAN Reverse Engineering

**A machine learning framework for bit-level signal classification of CAN bus payloads, using LSTM and Transformer architectures.**

Author: Craig Atkinson (a1669436) · Supervisor: Dr. Marian Mihailescu · University of Adelaide · May 2026

---

## Overview

The CAN bus is the standard serial protocol used by Electronic Control Units (ECUs) in modern vehicles. The structure and meaning of the 8-byte payloads are proprietary, so researchers who need to diagnose faults or study vehicle security have to reverse engineer them by hand. That process is slow: a single payload can mix fast sensor readings, slow-changing bit flags, rolling counters and checksums.

This project automates part of that process. Given a log of CAN traffic from a driving vehicle, the framework classifies **every bit of the 64-bit payload** into one of five functional classes using only the temporal behaviour of each bit (no hand-crafted features). The result is a predicted layout of each CAN ID that locates signal boundaries (**tokenisation**) and assigns each signal a data type (a partial **translation**).

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Static | Unused / spare / constant bits |
| 1 | Bit Flag | Boolean state flags (e.g. indicator on/off) |
| 2 | Counter | Rolling counters used for synchronisation |
| 3 | Sensor | Continuous sensor values (e.g. speed, RPM) |
| 4 | Checksum | Integrity-check bits |

## Key Features

- **Bit-level classification** of 64-bit CAN payloads into five signal types
- **Four architectures compared:** Full Sequence LSTM, Full Sequence Transformer, Independent Bit LSTM, Independent Bit Transformer
- **Majority-vote aggregation** of sliding-window predictions across a whole 20-30 minute driving log, with per-bit vote share displayed as transparency in the output visualisation
- **No manual feature engineering:** models learn directly from raw bit time series
- **Cross-vehicle testing** on Subaru, Mazda and Toyota CAN IDs, and a real reverse engineering demo on a Holden Astra 2018 (no public DBC available)

## How It Works

### Data

- **Source:** the CANdid dataset (Howson et al., 2025), roughly 60 million CAN messages across 10 vehicles, collected at the University of Adelaide. The continuous driving logs are used because they cover the widest range of driving conditions. Frames contain only a timestamp, CAN ID and payload.
- **Ground truth:** CANdid has no signal labels, so definitions were taken from the [comma.ai openDBC](https://github.com/commaai/opendbc) project. These community-sourced files often didn't match the CANdid logs (missing CAN IDs, shifted bit positions), so labels were **manually decoded and corrected**. The training set uses 13 Subaru Impreza CAN IDs; the Mazda CX-3 and Toyota Yaris sets are much smaller.

### Pre-processing

1. Parse raw logs into a pandas DataFrame (CAN ID, payload, timestamp)
2. Filter to the manually decoded and labelled messages
3. Convert 8-byte hex payloads into 64-bit binary arrays
4. Build sliding windows per CAN ID. All experiments use **window size 100 and stride 100** (non-overlapping), since larger windows hit GPU RAM limits and overlapping windows overfit quickly.

### Two input strategies

- **Full Sequence:** the whole 64-bit frame is one feature vector per time step, so the model can learn spatial layout and temporal behaviour together.
- **Independent Bit:** the payload is flattened so each bit position is its own 1D time series. The model ignores spatial layout and focuses on temporal signatures, such as a counter's predictable toggling or the low-entropy changes of a flag.

Each strategy is implemented with both an LSTM and a Transformer encoder.

### Hyperparameters

Models were trained on 11 decoded Subaru Impreza CAN IDs and validated on 1.

| Model | Learning rate | Dropout | Model depth | Validation Macro F1 | Early stopping epoch |
|-------|--------------:|--------:|------------:|--------------------:|---------------------:|
| LSTM Full Sequence | 0.001 | 20% | 64 | 0.22 | 20 |
| LSTM Independent Bits | 0.0005 | 20% | 64 | 0.36 | 15 |
| Transformer Full Sequence | 0.0001 | 50% | 128 | 0.35 | 40 |
| Transformer Independent Bits | 0.0001 | 50% | 128 | 0.37 | 40 |

The Transformers were less stable and needed a lower learning rate, higher dropout and greater depth to reach a stable validation loss. They also trained more slowly: about 30 minutes for 40 epochs, versus 5-10 minutes for the LSTMs. Inference is fast for all models.

### Window aggregation

A majority-vote step combines predictions from every window of a driving log into one class per bit. This handles signals that look different depending on context: a slow sensor on a stationary car can resemble a static bit flag, but shows clear continuous values when driving. The vote share for each bit is rendered as an alpha value, so contested bits are easy to spot and act as a rough confidence indicator.

### Evaluation

Macro F1 is the primary metric rather than accuracy. Signal sizes vary widely (a sensor may span 16 bits, a flag just one), so accuracy would reward a model that only learns the large sensor blocks. Macro F1 weights Flags, Counters, Sensors and Checksums equally. Static bits are excluded with a bit mask.

## Results

### Architecture comparison (unseen Subaru CAN ID)

| Model | Macro F1 | F1 Sensor | F1 Counter | F1 Flag | F1 Checksum |
|-------|---------:|----------:|-----------:|--------:|------------:|
| Transformer Full Sequence | 0.09 | 0.00 | 0.00 | 0.44 | 0.00 |
| LSTM Full Sequence | 0.09 | 0.00 | 0.00 | 0.48 | 0.00 |
| **LSTM Independent Bits** | **0.39** | **0.40** | **1.00** | 0.17 | 0.00 |
| Transformer Independent Bits | 0.27 | 0.29 | 0.67 | 0.11 | 0.00 |

- **Independent-bit models generalise; full-sequence models don't.** Full-sequence models scored near zero on most classes, likely because they memorise payload layouts rather than learning signal behaviour.
- **The Independent Bit LSTM was the best model**, and was especially strong on counters (F1 1.00). The report suggests recurrent layers suit isolated bit time series better than attention.
- **No model detected checksums** (F1 = 0.00). See [Limitations](#limitations).

### Cross-vehicle generalisation (Macro F1)

| Model | Subaru | Mazda | Toyota |
|-------|-------:|------:|-------:|
| LSTM Independent Bits, single-vehicle training | 0.39 | 0.62 | 0.44 |
| LSTM Independent Bits, cross-vehicle training | 0.39 | 0.52 | 0.45 |
| Transformer Independent Bits, single-vehicle training | 0.27 | 0.74 | 0.24 |
| Transformer Independent Bits, cross-vehicle training | 0.32 | 0.74 | 0.20 |

- Models trained **only on Subaru data** still generalised to Mazda and Toyota payloads. The single-vehicle LSTM scored 0.62 on Mazda and 0.44 on Toyota, compared with 0.39 on Subaru.
- **Adding cross-vehicle training data did not reliably help.** It slightly improved the Transformer on Subaru (0.27 to 0.32) but hurt it on Toyota (0.24 to 0.20), and lowered the LSTM on Mazda (0.62 to 0.52).
- The Transformer reached the single highest score (0.74 on Mazda), but the **LSTM was more consistent** across all three manufacturers.
- The report concludes that the temporal behaviour of basic data types such as counters and sensors is similar across manufacturers. Note that the Mazda and Toyota test sets contain only a limited number of CAN IDs.

### Real-world use: Holden Astra 2018

The Holden Astra is common in Australia and included in CANdid, but absent from openDBC, so it makes a realistic reverse engineering target with no public ground truth. The framework outputs a predicted layout with per-bit confidence that a researcher can use to decide where to focus manual analysis.

## Limitations

- **Checksums are not detected.** A single bit's time series, viewed in isolation, is indistinguishable from a counter, sensor or noise. Subaru checksums behaved like simple combinations of other signals, while Mazda used more complex algorithms. Detecting them needs context from neighbouring bits, which the independent-bit approach discards.
- **Data types only.** The framework predicts what *kind* of signal a bit is, not what it means (speed, RPM, indicator state, etc.).
- **Small, imperfect labelled data.** Labels come from community DBC files that needed manual correction, and most labelled data is from Subaru.
- **False positives** occur, particularly for bit flags.

## Future Work

- **Checksum detection** by combining bit-level temporal behaviour with payload-level context
- **Full translation:** assign meaning (speed, RPM, etc.) to detected signals
- **Physical value decoding:** once a sensor signal is isolated (e.g. bits 8-23), fit supervised regression against ground truth such as GPS or OBD-II logs to recover scale factors and offsets, with the LSTM handling tokenisation upstream

## Getting Started

```bash
git clone https://github.com/CraigAA/Automated-CAN-Reverse-Engineering.git
cd Automated-CAN-Reverse-Engineering
```

## References

- Buscemi, A., Turcanu, I., Castignani, G., Panchenko, A., Engel, T., and Shin, K. G. (2023). A survey on controller area network reverse engineering. *IEEE Communications Surveys & Tutorials*, 25(3):1445-1481.
- Buscemi, A., Castignani, G., Engel, T., and Turcanu, I. (2020). A data-driven minimal approach for CAN bus reverse engineering. *IEEE CAVS*, pp. 1-5.
- Lin, X. et al. (2024). ByCAN: Reverse engineering controller area network (CAN) messages from bit to byte level. *IEEE Internet of Things Journal*, 11:35477-35491.
- Ngo, P., Sprinkle, J., and Bhadani, R. (2022). CANClassify: Automated decoding and labeling of CAN bus signals. Master's thesis, UC Berkeley.
- Alkhatib, N. et al. (2022). CAN-BERT do it? Controller area network intrusion detection system based on BERT language model. *IEEE/ACS AICCSA*, pp. 1-8.
- Jaynes, M. et al. (2016). Automating ECU identification for vehicle security. *IEEE ICMLA*, pp. 632-635.
- Howson, T., Karlsen, J., Chothia, T., and Moore, T. (2025). CANdid: An open-access annotated dataset of vehicle CAN bus traffic. *USENIX VehicleSec*.
- comma.ai (2026). [opendbc: a Python API for your car](https://github.com/commaai/opendbc).

## Acknowledgements

Supervised by Dr. Marian Mihailescu, University of Adelaide.
