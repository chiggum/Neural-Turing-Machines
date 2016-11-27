# Neural-Turing-Machines
An attempt at replicating Deepmind's Neural Turing Machines in Theano as a part of my bachelor's thesis.

Advisor: [Prof. Amit Sethi](http://www.iitg.ernet.in/amitsethi/)

Here is the link to the paper: http://arxiv.org/abs/1410.5401

## Results
- Following are the results on COPY task of the NTM I implemented from scratch.
- Training is done on sequences of length varying from 1 to 20 and width 8 (bits per element in a sequence).
- With version 1 of NTM which has a single read-write head.

![Alt ntm-v1-on-test-seq-of-len-66](https://chiggum.github.io/Neural-Turing-Machines/plots/ntm_learning_curve-COPY-20-2016-03-03-15-25-49/ntm-info-COPY-10-120-66-2016-03-04-16-18-50.png)

- Learning Curve corresponding to version 1 of NTM (Sorry I din't compute the sliding window average in version 1).

![Alt ntm-v1-learning-curve](https://chiggum.github.io/Neural-Turing-Machines/plots/ntm_learning_curve-COPY-20-2016-03-03-15-25-49/ntm_learning_curve-COPY-20-2016-03-03-15-25-49.txt.png)

- With version 2 of NTM which has separate heads for reading and writing.

![Alt ntm-v2-on-test-seq-of-len-34](https://chiggum.github.io/Neural-Turing-Machines/plots/ntm2_learning_curve-COPY-5-2016-03-12-22-14-27/ntm2-info-COPY-10-120-34-2016-03-13-07-01-52.png)

- Learning Curve corresponding to version 2 of NTM.

![Alt ntm-v2-learning-curve](https://chiggum.github.io/Neural-Turing-Machines/plots/ntm2_learning_curve-COPY-5-2016-03-12-22-14-27/ntm2_learning_curve-COPY-5-2016-03-12-22-14-27_run_avg.png)

## Usage
For training: In `ntm_v*.py` set
```
to_test = False
```
To run your trained model or a pre-trained model from `model` directory,

In `ntm_v*.py` set
```
to_test = True
test_path = path_to_npz_model_parameters_file
```

## Thesis Report
Please visit this [link](https://chiggum.github.io/Neural-Turing-Machines/Report/Report_MA499.pdf) for my bachelor's thesis report.

## Presentation
Please visit this [link](https://chiggum.github.io/Neural-Turing-Machines/presentation/ram_pres_with_notes.pdf) for a presentation with comments, of my thesis.

## Reading material
Check out the `reading material` directory of this project on github for some relevant papers related to RAM based models.

## Other NTM Implementations
- fumin very nicely explained NTM with a working implementation in GO. Check out: https://github.com/fumin/ntm
- Another NTM implementation in Theano by shawntawn. Check out: https://github.com/shawntan/neural-turing-machines

## Future works
- Making NTM to work on other tasks described in the paper.
- Using NTM to make agents for playing games (Deep reinforcement learning).
