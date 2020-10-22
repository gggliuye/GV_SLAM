SLIC
==================


**SLIC (Simple Linear Iterative Clustering)** (2010) :
This algorithm generates superpixels by clustering pixels based on their color similarity and
proximity in the image plane. This is done in the five-dimensional 'labxy' space, where 'lab'
is the pixel color vector in CIELAB color space and xy is the pixel position.

This `paper <https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/SLIC_Superpixels.pdf>`_ made a comparsion of different super pixel methods.

.. image:: images/sp_compare.png
   :align: center
   :width: 80%
