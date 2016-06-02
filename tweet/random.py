import math as m
import numpy as np


def random_sample(para_range):
    res = []
    for item in para_range:
        my_min = item[0]
        my_max = item[1]
        my_type = item[2]
        a = m.log(my_min)
        b = m.log(my_max)
        val = m.exp(np.random.uniform(a, b))
        if my_type == 'int':
            val = int(round(val))
        res.append(val)

    return res


if __name__ == '__main__':
    hyper_para = []

    hyper_para.append((1e-6, 1, 'real'))

    hyper_para.append((2, 100, 'int'))

    hyper_para.append((1, 100, 'int'))
    res = []
    for i in range(100):
        res.append(random_sample(hyper_para))
