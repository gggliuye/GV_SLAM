Graph Based
=====================

Each pixel is treated as a node in a graph, and edge weight between nodes are set proportional ti the similarity
between pixels (normally, pixel distances and color differences.). Graph could be built based on grid sample, or KNN, etc.
Finally, the superpixel segments are extracted by effectively minimizing a cost function defined on the graph.

`My implementation <https://github.com/gggliuye/graph_based_image_segmentation>`_ based on
*Efficient graph-based image segmentation 2004*

`Link <https://vio.readthedocs.io/zh_CN/latest/Other/PGM.html>`_ to my documentation about PGM segmentation.

.. image:: images/allresults.jpg
   :align: center
   :width: 60%
