# RMSNorm

## Forward
$$y_{ij} = \frac{x_{ij}}{ \sqrt{ \frac{1}{H} \sum_{h=1}^H x_{ih}^2 + \epsilon } }$$

## Backward
$$\frac{\partial L}{\partial x_{ij}} = \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{\partial y_{bh}}{\partial x_{ij}}$$
$$\frac{\partial L}{\partial x_{ij}} = \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{\sqrt{ \frac{1}{H} \sum_{h=1}^H x_{ih}^2 + \epsilon } \frac{\partial x_{bh}}{\partial x_{ij}} - x_{bh} }{\partial x_{ij}}$$
