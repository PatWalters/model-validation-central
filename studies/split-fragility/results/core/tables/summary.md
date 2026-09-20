# Mean over each scheme's 50 folds

## MAE

| scheme | k-NN (Tanimoto) | LightGBM + Morgan | ChemProp + CheMeleon | Monroe + TabPFN 3.5 | Training median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.500 ± 0.064 worse | 0.488 ± 0.046 worse | 0.463 ± 0.044 worse | 0.440 ± 0.042 **best** | 0.879 ± 0.089 worse |
| Bemis-Murcko scaffold | 0.620 ± 0.081 worse | 0.592 ± 0.072 worse | 0.567 ± 0.078 worse | 0.543 ± 0.068 **best** | 0.882 ± 0.111 worse |
| Butina cluster | 0.786 ± 0.139 worse | 0.739 ± 0.114 worse | 0.711 ± 0.130 worse | 0.691 ± 0.128 **best** | 0.896 ± 0.166 worse |
| Diverse (MaxMin 25%) | 0.793 ± 0.109 worse | 0.753 ± 0.079 worse | 0.754 ± 0.097 worse | 0.718 ± 0.093 **best** | 0.907 ± 0.118 worse |
| UMAP cluster | 0.893 ± 0.130 worse | 0.834 ± 0.141 worse | 0.808 ± 0.173 tied | 0.777 ± 0.122 **best** | 0.898 ± 0.171 worse |
| Time | 0.930 ± 0.141 worse | 0.851 ± 0.129 worse | 0.831 ± 0.124 worse | 0.798 ± 0.125 **best** | 0.940 ± 0.146 worse |

## R2

| scheme | k-NN (Tanimoto) | LightGBM + Morgan | ChemProp + CheMeleon | Monroe + TabPFN 3.5 | Training median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.576 ± 0.104 worse | 0.618 ± 0.087 worse | 0.641 ± 0.085 worse | 0.672 ± 0.074 **best** | -0.013 ± 0.020 worse |
| Bemis-Murcko scaffold | 0.357 ± 0.211 worse | 0.441 ± 0.189 worse | 0.470 ± 0.185 worse | 0.514 ± 0.159 **best** | -0.029 ± 0.038 worse |
| Butina cluster | -0.017 ± 0.360 worse | 0.153 ± 0.217 worse | 0.175 ± 0.275 worse | 0.246 ± 0.245 **best** | -0.098 ± 0.145 worse |
| Diverse (MaxMin 25%) | 0.092 ± 0.247 worse | 0.195 ± 0.194 worse | 0.193 ± 0.217 worse | 0.283 ± 0.184 **best** | -0.042 ± 0.049 worse |
| UMAP cluster | -0.292 ± 0.278 worse | -0.120 ± 0.322 worse | -0.088 ± 0.420 tied | 0.029 ± 0.271 **best** | -0.210 ± 0.422 worse |
| Time | -0.239 ± 0.402 worse | -0.041 ± 0.326 worse | -0.007 ± 0.339 worse | 0.104 ± 0.255 **best** | -0.090 ± 0.102 worse |

## SPEARMAN

| scheme | k-NN (Tanimoto) | LightGBM + Morgan | ChemProp + CheMeleon | Monroe + TabPFN 3.5 |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.753 ± 0.059 worse | 0.773 ± 0.059 worse | 0.794 ± 0.050 worse | 0.809 ± 0.043 **best** |
| Bemis-Murcko scaffold | 0.623 ± 0.125 worse | 0.657 ± 0.146 worse | 0.688 ± 0.132 worse | 0.708 ± 0.116 **best** |
| Butina cluster | 0.391 ± 0.148 worse | 0.460 ± 0.156 worse | 0.496 ± 0.166 tied | 0.516 ± 0.183 **best** |
| Diverse (MaxMin 25%) | 0.424 ± 0.137 worse | 0.464 ± 0.149 worse | 0.477 ± 0.148 worse | 0.524 ± 0.144 **best** |
| UMAP cluster | 0.187 ± 0.156 worse | 0.256 ± 0.238 worse | 0.313 ± 0.242 tied | 0.325 ± 0.258 **best** |
| Time | 0.264 ± 0.192 worse | 0.319 ± 0.196 worse | 0.363 ± 0.233 tied | 0.378 ± 0.230 **best** |
