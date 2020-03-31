# Solving a sudoku with backstrapping and deep learning

## Table of contents:

* Grid Detection
* Data generation
* CNN training
* Solving the grid with backstrapping


## Dependencies 

* opencv-Python
* Numpy
* Pillow
* Keras
* Scipy

## Grid detection:

*(details and code for this section can be found in the notebook: grid_detection.ipynb)*

First, some preprocessing:

![](plots/img_preprocessing.png)

Detect the grid contour and corners using opencv:

![](plots/grid_detection.png)

Extraction of digit area:

![](plots/grid.png)

## Data Generation

*(details and code for this section can be found in the notebook: 'digit_generation&training.ipynb')*

First, generate random digit image with random field around the digit area:

![](plots/cell_selection.png)


Produce random noise and elastic deformation for model robustness:

![](plots/noisy_images.png)


Some exemples of Generated images:

![](plots/digits.png)

## Train the model with Keras (Tensorflow backend):

*(details and code in the notebook: 'digit_generation&training.ipynb')*

![](plots/training_stats.png)

Evaluation on validation set composed of "real life" sudoku images:
4 grids, 324 images (sudoku cells) for a total acuracy of 1 (100%).

![](plots/conf_matrix.png)

## Solving the grid with backstrapping

see [this noteook](solve_grid.ipynb) for details.
