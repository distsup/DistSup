# -*- coding: utf8 -*-
#   Copyright 2019 JSALT2019 Distant Supervision Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from heapq import *
import numpy as np
#import numba
#from numba import jit, autojit

#@jit((numba.float64[:,:], numba.int64, numba.int64), nopython=True)
def find_best_assign(S, i, j):
    if i > 0:
        tmp = S[j, :] - S[i - 1, :]
    else:
        tmp = S[j, :]
    ret = tmp.argmin()#.cpu().item()
    return ret, tmp[ret]#.cpu().item()

#@jit(nopython=True)
def calc(D, limit, threshold):
    N, K = D.shape
    #S = D.clone()
    #for i in range(1, N):
    #    S[i] = S[i] + S[i - 1]
    S = D.cumsum(dim=0).cpu().numpy()

    l, r = np.arange(N), np.arange(N)
    v = D.min(dim=1)[0].cpu().numpy()

    h = []
    for i in range(1, N):
        idx, value = find_best_assign(S, i - 1, i)
        heappush(h, (value - v[i - 1] - v[i], i - 1, i - 1, i))

    num_segments = N
    values = [0]
    while num_segments > limit + 1:
        value, boundary, ass_l, ass_r = heappop(h)
        values.append(value + values[-1])
        if threshold is not None and values[-1] > threshold:
            break
        if l[boundary] != ass_l:
            continue
        if r[boundary + 1] != ass_r:
            continue

        #merge
        r[boundary] = l[boundary + 1]
        l[boundary + 1] = l[boundary]
        L, R = l[boundary], r[boundary + 1]
        r[L] = R
        l[R] = L

        #add left merge
        #_, cur_v = find_best_assign(S, L, R)
        cur_v = value + v[boundary] + v[boundary + 1]
        v[L] = cur_v
        v[R] = cur_v
        if L != 0:
            idx, value = find_best_assign(S, l[L - 1], R)
            heappush(h, (value - cur_v - v[L - 1], L - 1, l[L - 1], R))
        #add right merge
        if R != N - 1:
            idx, value = find_best_assign(S, L, r[R + 1])
            heappush(h, (value - cur_v - v[R + 1], R, L, r[R + 1]))

        num_segments -= 1

    c = []
    for i in range(N):
        if l[i] == i:
            R = r[i]
            idx, cost = find_best_assign(S, i, R)
            #print("{0} {1} {2}".format(l[i], r[i], idx))
        c.append(idx)
    return np.array(c), values

if __name__ == '__main__':
    #D = abs(np.random.randn(4, 3))
    import torch
    D = np.array([[1, 0], [1, 2], [2, 1.5]])
    D = torch.from_numpy(D)
    print(D)
    calc(D, 1)
    print()
    calc(D, 2)
    print()
    calc(D, 3)
   
