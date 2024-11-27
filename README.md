[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/javirk/europa_surface/master?labpath=DEMO_draw_box_to_mask.ipynb)

revert_fixed: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/javirk/europa_surface/revert_fixed?labpath=DEMO_draw_box_to_mask.ipynb)

#### Code used for publication XX (subm. to PSJ)

## Demo: test the interactive LineaMapper v2.0
 **Try it directly online: [Demo on Binder](https://mybinder.org/v2/gh/javirk/europa_surface/master?labpath=DEMO_draw_box_to_mask.ipynb)** (takes a few minutes to start up).

Steps:
- Download the jupyter notebook called *LineaMapper_v2_bbox_prompt.ipynb*a s well as the test image *TO BE NAMED.png* to one folder on your local machine.
- Download the model weights from zenodo *LINK TO BE INSERTED*
- Store the model weights in the same folder as the jupyter notebook.
- Either activate your local python environment (conda, pip) or generate a new one with the *requirements.txt* files (TO BE UPLOADED and STEP-BY-STEP GUIDE ADDED)
- Test first to run the jupyter notebook as it is.
- If you can successfully draw multiple bounding boxes inside the image, you can now try with an image of your liking by changing the variable *IMAGE_PATH*. Note that the image should not be greater than 200x200 pixels for a flawless prediction.

To run the jupyter notebook locally, you might need to run the following command in your local environment first (if your kernel crashes):

> jupyter nbextension enable --py widgetsnbextension --sys-prefix

Instead of executing the jupyter notebook on your local machine, you can also use it directly with Google Colab. Just make sure that the model weights are uploaded to your Google Drive.

If you would like to use the stitching tool, ...

Use python 3.12 for the installation. For example, use:
> virtualenv -p your/path/to/python3.12/python.exe vgeosam312

Then, install the required packages with pip:
> pip install -r requirements.txt

Install pycocotools with
> git clone https://github.com/CarolineHaslebacher/cocoapi.git
> 
> git checkout cocoeval_for_multiple_categories

> cd /cocoapi
> 
> pip install ./PythonAPI
> 
(works on Windows *only* with /common/ folder copied into PythonAPI)


Private note: The requirements.txt come from venv 'vgeosam312'.


