Gardient Ascent
==================

Starting from an inital rough clustering, during each iteration gradient ascnet mehtods refine
the clusters from the previous iteration to obatin better segmentation until convergence.

SLIC
==================

**SLIC (Simple Linear Iterative Clustering)** (2010) :
This algorithm generates superpixels by clustering pixels based on their color similarity and
proximity in the image plane. This is done in the five-dimensional 'labxy' space, where 'lab'
is the pixel color vector in CIELAB color space and xy is the pixel position.

This `paper <https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/SLIC_Superpixels.pdf>`_ made a comparsion of different super pixel methods.

.. image:: images/sp_compare.PNG
   :align: center
   :width: 80%

1.1 Distance Measure
---------------

.. math::
  \begin{align*}
  &d_{lab} = \| v_{lab,1} - v_{lab,2} \|_{2} \\
  &d_{xy} = \| v_{xy,1} - v_{xy,2} \|_{2} \\
  &D_{s} = d_{lab} + m \frac{d_{xy}}{\sqrt{N/K}}
  \end{align*}

.. math::
  d_{xy} = \| v_{xy,1} - v_{xy,2} \|_{2}

.. math::
  D_{s} = d_{lab} + m \frac{d_{xy}}{\sqrt{N/K}}

where N is the number of pixels in the image, and K the number of desired clusters. The N K term serves as a normalization for pixel distance.

1.2 Algorithm
-----------------

Image gardient computed as :

.. math::
  G(x,y) = \|I(x+1, y) - I(x-1,y)\|^{2} + \|I(x, y+1) - I(x,y-1)\|^{2}

The algorithm is a sepcial case of K-means adapted to the task.

.. image:: images/SLIC.PNG
   :align: center
   :width: 80%
