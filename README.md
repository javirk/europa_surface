[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/javirk/europa_surface/master?labpath=DEMO_bbox_prompt.ipynb)

#### Code used for publication XX (subm. to PSJ)

## Demo: test the interactive LineaMapper v2.0
 **Try it directly online: [live demo on Binder](https://mybinder.org/v2/gh/javirk/europa_surface/master?labpath=DEMO_bbox_prompt.ipynb)**.

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


