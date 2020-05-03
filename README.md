# Solving a sudoku with backtracking and deep learning

![](app.gif)

## Table of contents:

* Grid Detection
* Data generation
* CNN training
* Solving the grid with backstrapping
* App creation


## Dependencies 

* opencv-Python
* Numpy
* Pillow
* Keras
* Scipy

## Grid detection:

(details and code for this section can be found in [this notebook](grid_detection.ipynb))

First, some preprocessing:

![](plots/img_preprocessing.png)

Detect the grid contour and corners using opencv:

![](plots/grid_detection.png)

Extraction of digit area:

![](plots/grid.png)

## Data Generation

*(details and code for this section can be found in [this notebook](digit_generation&training.ipynb))

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
28 grids, 2268 images (sudoku cells) for a total accuracy of 99.51%.

![](plots/conf_matrix3.png)

## Solving the grid with backtracking

see [this notebook](solve_grid.ipynb) for details.

![](plots/final_display.png)

## Putting everything in a desktop app.

For this secton we will use Kivy, an opensource python library for application developpement on multiple OS.$
The code is available in the kivy_app folder.




