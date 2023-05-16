# Supplementary Code for the Paper _An Assessment of Differential-Neural Distinguishers_.

**Note:** This Git repository uses git submodules to have access to the neural networks from literature. In order to clone them, run `git submodule init` and `git submodule update` (or copy the two git repositories in the corresponding directories if you have not cloned this repository).

## Overview

| File/Directory&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Explanation                                                                                                                                                  |
|----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| cipher                                                   | Contains the implementation of all ciphers we use                                                                                                            |
| CSYY22                                                   | https://github.com/AI-Lab-Y/ND_mc as a submodule                                                                                                             |
| ddt                                                      | Code for DDT calculation                                                                                                                                     |
| exception                                                | Custom exceptions                                                                                                                                            |
| freshly_trained_nets                                     | Directory where networks trained by the train_* scripts will be saved                                                                                        |
| Goh19                                                    | https://github.com/agohr/deep_speck as a submodule                                                                                                           |
| lib                                                      | External libraries                                                                                                                                           |
| LLS+23                                                   | https://github.com/JIN-smile/Improved-Related-key-Differential-based-Neural-Distinguishers as a submodule                                                    |
| nets                                                     | Previously trained networks used for most experiments in the paper                                                                                           |
| differential_linear_bias.py                              | Empirically evaluates the differential-linear biases for all one bit masks and all ciphers                                                                   |
| eval.py                                                  | Provides functions for distinguisher evaluations                                                                                                             |
| eval_(cipher).py                                         | Evaluates the performance of the distinguishers for 'cipher' in various settings (default, real difference, etc.)                                            |
| make_train_data.py                                       | Provides functions for data generation                                                                                                                       |
| multi_pair_advantage.py                                  | Evaluates the multi-pair advantage                                                                                                                           |
| present_biases.py                                        | Statistical evaluation of the PRESENT distinguisher on more rounds                                                                                           |
| requirements.txt                                         | Python packages needed to run the python scripts                                                                                                             |
| train_(cipher).py                                        | Trains a neural network for 'cipher' using the same hyper-parameter as in the paper                                                                          |
| train_nets.py                                            | Main code for training a neural network                                                                                                                      |
| verify_test_vectors.py                                   | Checks that the ciphers in cipher/ are correctly implemented                                                                                                 |
| ZWWW22                                                   | Neural Networks from the URL provided in https://github.com/CryptAnalystDesigner/NeuralDistingsuisherWithInception/blob/main/model_file_download_address.txt |

## Requirements
The python code was tested using Python 3.9.12 and the packages from requirements.txt.
Some experiments require up to 32GB of RAM. This can be prevented by lowering the n_samples* values, but comes at the expense of less precise results.

For experiments using the neural networks from ZWWW22, those network first need to be downloaded from the URL provided in https://github.com/CryptAnalystDesigner/NeuralDistingsuisherWithInception/blob/main/model_file_download_address.txt and then be put into the directory ZWWW22 (the directory ZWWW22 should then contain the two subdirectories Speck3264 and Simon3264).

## References
| Identifier | Work                                                                                                                                                                                                                                                                   |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [CSYY22]   | Chen, Y., Shen, Y., Yu, H., Yuan, S.: A New Neural Distinguisher Considering Features Derived From Multiple Ciphertext Pairs. The Computer Journal (03 2022). https://doi.org/10.1093/comjnl/bxac019, https://doi.org/10.1093/comjnl/bxac019, bxac019                  |
| [Goh19]    | Gohr, A.: Improving attacks on round-reduced Speck32/64 using deep learning. In: Boldyreva, A., Micciancio, D. (eds.) CRYPTO 2019, Part II. LNCS, vol. 11693, pp. 150–179. Springer, Heidelberg (Aug 2019). https://doi.org/10.1007/978-3-030-26951-7_6                |
| [LLS+23]   | Lu, J., Liu, G., Sun, B., Li, C., Liu, L.: Improved (Related-Key) Differential-Based Neural Distinguishers for SIMON and SIMECK Block Ciphers. The Computer Journal (01 2023). https://doi.org/10.1093/comjnl/bxac195, https://doi.org/10.1093/comjnl/bxac195, bxac195 |
| [ZWWW22]   | Zhang, L., Wang, Z., Wang, B.: Improving differential-neural cryptanalysis with inception blocks. Cryptology ePrint Archive, Report 2022/183 (2022), https://eprint.iacr.org/2022/183                                                                                  |
