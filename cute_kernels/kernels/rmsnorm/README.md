# RMSNorm

## Forward
$$y_{ij} = \frac{w_j x_{ij}}{r_i}$$
where
$$r_i = \sqrt{ \frac{1}{H} \sum_{h=1}^H x_{ih}^2 + \epsilon }$$

## Backward
$$\frac{\partial L}{\partial w_j} = \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{\partial y_{bh}}{\partial w_j}$$
$$= \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{x_{bh}}{r_b} \frac{\partial w_h}{\partial w_j}$$
$$= \sum_{b=1}^{B} \frac{\partial L}{\partial y_{bj}} \frac{x_{bj}}{r_b}$$

$$\frac{\partial L}{\partial x_{ij}} = \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{\partial y_{bh}}{\partial x_{ij}}$$
$$= \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{w_h}{r_i^2} \left[ r_b \frac{\partial x_{bh}}{\partial x_{ij}} - x_{bh} \frac{1}{2H} 2 x_{bj} \right] $$
$$= \frac{\partial L}{\partial y_{ij}} \frac{w_h}{r_i} - \sum_{b=1}^{B} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{bh}} \frac{w_h}{r_i^2} x_{bh} \frac{1}{2H} 2 x_{bj}$$
