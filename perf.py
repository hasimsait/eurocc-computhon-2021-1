#!/usr/bin/env python
# -*- coding: utf-8 -*-
from statistics import mean

#com-dblp
seq = [
    0.0833482, 0.0870874, 0.0825669, 0.0835959, 0.0838623, 0.0841608,
    0.0843186, 0.0844158, 0.0844804, 0.0844393
]
parr2 = [
    0.0561561, 0.0562774, 0.0562605, 0.0587298, 0.0561784, 0.0569796,
    0.0565435, 0.057943, 0.0564277, 0.05657229
]
parr4 = [
    0.0332295, 0.0330306, 0.0332539, 0.0332748, 0.0335284, 0.0331232,
    0.0334931, 0.0330179, 0.0332188, 0.0331409
]
parr8 = [
    0.0243772, 0.0242942, 0.0242995, 0.0241309, 0.0238716, 0.0243075,
    0.0252531, 0.0243065, 0.0241752, 0.0284354
]

avgseq = mean(seq)
avgp2 = mean(parr2)
avgp4 = mean(parr4)
avgp8 = mean(parr8)

print("seq:", avgseq, "\n2 threads:", avgp2, "\n4 threads:", avgp4,
      "\n8 threads:", avgp8, "\n2 threads offer", avgseq / avgp2,
      "speedup.\n4 threads offer", avgseq / avgp4, "speedup.\n8 threads offer",
      avgseq / avgp8, "speedup.")

#yt
seq = []
parr2 = []
parr4 = []
parr8 = []

avgseq = mean(seq)
avgp2 = mean(parr2)
avgp4 = mean(parr4)
avgp8 = mean(parr8)

print("seq:", avgseq, "\n2 threads:", avgp2, "\n4 threads:", avgp4,
      "\n8 threads:", avgp8, "\n2 threads offer", avgseq / avgp2,
      "speedup.\n4 threads offer", avgseq / avgp4, "speedup.\n8 threads offer",
      avgseq / avgp8, "speedup.")
