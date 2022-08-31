# Supplementary Code for the Paper _An Assessment of Differential-Neural Distinguishers_.

**Note:** This Git repository uses git submodules to have access to the neural networks from literature. In order to clone them, run `git submodule init` and `git submodule update` (or copy the two git repositories in the corresponding directories if you have not cloned this repository).

## Overview

| File/Directory&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  | Explanation                                                                                                       |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| cipher                                                    | Contains the implementation of all ciphers we use                                                                 |
| CSYY22                                                    | https://github.com/AI-Lab-Y/ND_mc as a submodule                                                                  |
| ddt                                                       | Code for DDT calculation                                                                                          |
| exception                                                 | Custom exceptions                                                                                                 |
| freshly_trained_nets                                      | Directory where networks trained by the train_* scripts will be saved                                             |
| Goh19                                                     | https://github.com/agohr/deep_speck as a submodule                                                                |
| lib                                                       | External libraries                                                                                                |
| nets                                                      | Previously trained networks used for most experiments in the paper                                                |
| differential_linear_bias.py                               | Empirically evaluates the differential-linear biases for all one bit masks and all ciphers                        |
| eval.py                                                   | Provides functions for distinguisher evaluations                                                                  |
| eval_(cipher).py                                          | Evaluates the performance of the distinguishers for 'cipher' in various settings (default, real difference, etc.) |
| make_train_data.py                                        | Provides functions for data generation                                                                            |
| present_biases.py                                         | Statistical evaluation of the PRESENT distinguisher on more rounds                                                |
| requirements.txt                                          | Python packages needed to run the python scripts                                                                  |
| train_(cipher).py                                         | Trains a neural network for 'cipher' using the same hyper-parameter as in the paper                               |
| train_nets.py                                             | Main code for training a neural network                                                                           |
| verify_test_vectors.py                                    | Checks that the ciphers in cipher/ are correctly implemented                                                      |

## Requirements
The python code was tested using Python 3.9.12 and the packages from requirements.txt.
Some experiments require up to 32GB of RAM. This can be prevented by lowering the n_samples* values, but comes at the expense of less precise results.
