## Code used in publication ['Length, Width, and Relative Age Analysis of Lineaments in the Galileo Regional Maps with LineaMapper', PSJ, 2025](https://iopscience.iop.org/article/10.3847/PSJ/add349)

### Demos
Test the interactive LineaMapper v2.0 and get full georeferenced predictions with all versions of LineaMapper

 Bounding Box Demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/javirk/europa_surface/blob/revert_fixed/DEMO_draw_box_to_mask.ipynb)
 
 LineaMapper georeferenced predictions: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/javirk/europa_surface/blob/revert_fixed/DEMO_apply_LineaMapper.ipynb)


Code and data are published on Mendeley Data: https://data.mendeley.com/datasets/rjhsjrnxgv/1

You need to install GDAL on your machine (not only in the python environment), for example from here https://trac.osgeo.org/osgeo4w/. Otherwise, the requirements_full.txt should work 

Steps:
- Download the jupyter notebook called *LineaMapper_v2_bbox_prompt.ipynb*a s well as the test image *TO BE NAMED.png* to one folder on your local machine.
- Download the model weights from Mendeley Data *LINK TO BE INSERTED*
- Store the model weights in the same folder as the jupyter notebook.
- Either activate your local python environment (conda, pip) or generate a new one with the *requirements.txt* files
- Test first to run the jupyter notebook as it is.
- If you can successfully draw multiple bounding boxes inside the image, you can now try with an image of your liking by changing the variable *IMAGE_PATH*. Note that the image should not be greater than 200x200 pixels for a flawless prediction.

To run the jupyter notebook locally, you might need to run the following command in your local environment first (if your kernel crashes):

> jupyter nbextension enable --py widgetsnbextension --sys-prefix

Instead of executing the jupyter notebook on your local machine, you can also use it directly with Google Colab. Just make sure that the model weights are uploaded to your Google Drive.

If you would like to use the stitching tool, ...

Use python 3.12 or 3.13 for the installation. For example, use:
> virtualenv -p your/path/to/python3.12/python.exe vgeosam312

Then, install the required packages with pip:
> pip install -r requirements.txt

Or use:
> pip install -r requirements_full.txt

Install pycocotools with
> git clone https://github.com/CarolineHaslebacher/cocoapi.git
> 
> git checkout cocoeval_for_multiple_categories

> cd /cocoapi
> 
> pip install ./PythonAPI
> 
(works on Windows *only* with /common/ folder copied into PythonAPI)

if on Colab, try:
> !git clone -b cocoeval_for_multiple_categories https://github.com/CarolineHaslebacher/cocoapi.git


