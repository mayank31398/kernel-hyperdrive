# Embedding

## Forward
$$y_{ij} = w_{x_ij} = \sum_{v=1}^V I[x_i = v]w_{vj}$$
where $I[x = v] = 1$ if $x = v$ and $0$ otherwise.

## Backward
<!-- weight -->
$$\frac{\partial L}{\partial w_{ij}} = \sum_{v=1}^{V} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{vh}} \frac{\partial }{\partial w_{ij}} \left[ \sum_{k=1}^V I[x_v = k]w_{kj} \right]$$
$$= \sum_{v=1}^{V} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{vh}} \sum_{k=1}^V I[x_v = k] I[i = k] I[j = h] $$
$$= \sum_{v=1}^{V} \sum_{h=1}^{H} \frac{\partial L}{\partial y_{vh}} I[x_v = i] I[j = h] $$
$$= \sum_{v=1}^{V} \frac{\partial L}{\partial y_{vj}} I[x_v = i] $$

Finally, we have
$$\frac{\partial L}{\partial w_{ij}} = \sum_{v=1}^{V} \frac{\partial L}{\partial y_{vj}} I[x_v = i] $$
