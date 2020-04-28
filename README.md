# Fast-BFGSv2

## Introduction
Codes for paper "A Dynamic Subspace Based BFGS Method for Large Scale Optimization Problem".

The Fast-BFGS performance better than BFGS and L-BFGS in the number of function and gradient evaluations (nfg) with the termination criterion $\| \nabla f_k \|_2 < 10^{-5}$ and $m=8$.

| name     | n    | GD    | BFGS  | L-BFGS | Fast-BFGS(sec=F) | Fast-BFGS(sec=T) |
| ---      | ---  | ---   | ---   | ---    | ---   | ---   |
| ARWHEAD  | 1000 | >1000 | 39    | 24     | 12    | 12    |
| BDQRTIC  | 1000 | >1000 | --    | --     | 295   | 589   |
| BDEXP    | 1000 | >1000 | 19    | 19     | 19    | 19    |
| COSINE   | 1000 | >1000 | --    | --     | 15    | 15    |
| CUBE     | 1000 | >1000 | --    | --     | --    | >1000 |
| DIXMAANE | 1500 | >1000 | 195   | 244    | 332   | 327   |
| DIXMAANF | 1500 | >1000 | 336   | 216    | 237   | 281   |
| DIXMAANG | 1500 | >1000 | 954   | 384    | 239   | 236   |
| DQRTIC   | 1000 | --    | --    | --     | 286   | 317   |
| EDENSCH  | 1000 | 59    | 86    | 52     | 27    | 28    |
| ENGVAL1  | 1000 | 66    | 154   | 119    | 28    | 28    |
| EG2      | 1000 | 7     | 6     | 6      | 6     | 6     |
| EXTROSNB | 1000 | 63    | 309   | 333    | 66    | 66    |
| FLETCHER | 100  | >1000 | --    | --     | --    | --    |
| FREUROTH | 1000 | --    | --    | --     | 41    | 41    |
| GENROSE  | 1000 | >1000 | >1000 | 39     | 38    | 38    |
| HIMMELBG | 1000 | >1000 | 3     | 3      | 8     | 8     |
| HIMMELH  | 1000 | 20    | 9     | 9      | 13    | 13    |
| LIARWHD  | 1000 | >1000 | --    | 28     | 70    | 56    |
| NONDIA   | 1000 | >1000 | --    | 55     | 100   | 77    |
| NONDQUAR | 1000 | >1000 | 270   | 320    | 199   | 253   |
| NONSCOMP | 1000 | 86    | 286   | 238    | 59    | 60    |
| POWELLSG | 1000 | >1000 | 459   | 49     | 54    | 55    |
| SCHMVETT | 1000 | 181   | 26    | 24     | 28    | 28    |
| SINQUAD  | 1000 | >1000 | 140   | 143    | 18    | 17    |
| SROSENBR | 1000 | >1000 | --    | 39     | 38    | 38    |
| TOINTGSS | 1000 | 6     | 9     | 9      | 4     | 4     |
| TQUARTIC | 1000 | >1000 | 16    | 17     | 17    | 17    |
| WOODS    | 1000 | >1000 | --    | 92     | 74    | 57    |