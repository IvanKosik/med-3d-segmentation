med-3d-segmentation
===================

Code to train neural network models to segmentate 3D medical tomographic images.


Installation
~~~~~~~~~~~~

**Create conda env with Python 3.7**

.. code:: bash

    $ conda create -n my-conda-env-name python=3.7

**Activate new conda env**

.. code:: bash

    $ conda activate my-conda-env-name

**Install dependencies**

.. code:: bash

    $ pip install tensorflow-gpu==1.14 albumentations opencv-python pandas scikit-image segmentation-models keras nibabel
