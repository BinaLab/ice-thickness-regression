# ice-thickness-regression

#### Overview
This code estimates thickness of each ice layer present in a given Snow Radar image. The algorithm is based on Fully Convolutional Regression Networks, and assumes a maximum of 27 ice layers to be present in an image. 

The code requires `keras` library.

#### Python files

Training: `thickness_regression.py`

Inference or prediction: `inference.py`

#### Output

Plots and models will be saved in the `model_out` folder. Outputs will be generated in the `out` folder.

After running `inference.py`, thickness estimates for each input image will be generated as `.txt` files inside the `out` folder. Use `out/create_df.py` to generate a `.csv` which would be easier to comprehend.
