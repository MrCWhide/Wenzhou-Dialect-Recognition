# Wenzhou Speech Recognition Project  、
NO COPY！！！！！！！！ 2024年10月27日声明
不允许复制、发行、商用，软著、专利注册  2024年10月27日声明
## Project Overview  
This project aims to build a speech recognition system for Wenzhou dialect,coded by CWhide,26,Oct,2024,using MFCC for feature extraction and applying both HMM and neural network models.  

coding environment：Ubuntu，CWhide‘s computer

## Folder Structure
- **data/**: Contains audio samples and labels
  - `train_sample/`: Training audio files
  - `test_sample/`: Test audio files
- **src/**: Source code files
  - `features_extraction.py`: MFCC extraction module.Several grguments should be adjusted(PS:没有学过)
  - `hmm_model.py`: HMM model implementation
  - `dnn_model.py`: DNN model for more complex feature learning（留）
  - `main.py`: Main file for running the full training and evaluation pipeline（主程序）
- **utils/**: Utility files for data loading

## Usage
1. Place audio samples in `data/train_sample/` and `data/test_sample/`.
2. Update `main.py` with appropriate labels and run.

## if you have any questions,please page C.whide.
## Please forgive my errs in this project.If they're pointed out,I would be extremely obliged
