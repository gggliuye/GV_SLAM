SuperPixel
===================================


* They carry more information than pixels.
* Superpixels have a perceptual meaning since pixels belonging to a given superpixel share similar visual properties.
* They provide a convenient and compact representation of images that can be very useful for computationally demanding problems.

`medium post <https://medium.com/@darshita1405/superpixels-and-slic-6b2d8a6e4f08>`_

.. image:: images/dog.png
   :align: center
   :width: 60%


* Graph Based method
[Mine](https://github.com/gggliuye/graph_based_image_segmentation)

* SLIC (Simple Linear Iterative Clustering)
This algorithm generates superpixels by clustering pixels based on their color similarity and proximity in the image plane. This is done in the five-dimensional [labxy] space, where [lab] is the pixel color vector in CIELAB color space and xy is the pixel position.

* TUC
[Superpixel Technische Universitat Chemnitz](https://www.tu-chemnitz.de/etit/proaut/en/research/superpixel.html) 2015.
[Application](https://www.tu-chemnitz.de/etit/proaut/en/research/changeprediction.html) in a winter-summer localization task.

* SEEDS
[SEEDS: Superpixels Extracted via Energy-Driven Sampling](https://arxiv.org/abs/1309.3848).

.. toctree::
   :maxdepth: 3
   :caption: Contents: