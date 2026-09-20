# Mean over each scheme's 50 folds

## MAE

| scheme | k-NN (Tanimoto) | LightGBM + Morgan | ChemProp + CheMeleon | Monroe + TabPFN 3.5 | Training median | k-NN (published) | SVM (published) | RF (published) | XGBoost (published) | MLP (published) | GNN (published) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.500 ± 0.064 worse | 0.488 ± 0.046 worse | 0.463 ± 0.044 worse | 0.440 ± 0.042 **best** | 0.879 ± 0.089 worse | 0.500 ± 0.064 worse | 0.471 ± 0.045 worse | 0.487 ± 0.054 worse | 0.490 ± 0.047 worse | 0.492 ± 0.049 worse | 0.595 ± 0.058 worse |
| Bemis-Murcko scaffold | 0.620 ± 0.081 worse | 0.592 ± 0.072 worse | 0.567 ± 0.078 worse | 0.543 ± 0.068 **best** | 0.882 ± 0.111 worse | 0.620 ± 0.081 worse | 0.578 ± 0.075 worse | 0.587 ± 0.078 worse | 0.598 ± 0.079 worse | 0.612 ± 0.082 worse | 0.692 ± 0.082 worse |
| Butina cluster | 0.786 ± 0.139 worse | 0.739 ± 0.114 worse | 0.711 ± 0.130 tied | 0.691 ± 0.128 **best** | 0.896 ± 0.166 worse | 0.786 ± 0.139 worse | 0.732 ± 0.133 worse | 0.735 ± 0.138 worse | 0.734 ± 0.135 worse | 0.773 ± 0.147 worse | 0.857 ± 0.178 worse |
| Diverse (MaxMin 25%) | 0.793 ± 0.109 worse | 0.753 ± 0.079 worse | 0.754 ± 0.097 tied | 0.718 ± 0.093 **best** | 0.907 ± 0.118 worse | 0.793 ± 0.096 worse | 0.759 ± 0.088 worse | 0.753 ± 0.084 worse | 0.775 ± 0.094 worse | 0.781 ± 0.099 worse | 0.893 ± 0.112 worse |
| UMAP cluster | 0.893 ± 0.130 worse | 0.834 ± 0.141 worse | 0.808 ± 0.173 tied | 0.777 ± 0.122 **best** | 0.898 ± 0.171 worse | 0.893 ± 0.130 worse | 0.815 ± 0.140 tied | 0.833 ± 0.143 tied | 0.844 ± 0.140 worse | 0.859 ± 0.171 worse | 0.923 ± 0.195 worse |
| Time | 0.930 ± 0.141 worse | 0.851 ± 0.129 worse | 0.831 ± 0.124 worse | 0.798 ± 0.125 **best** | 0.940 ± 0.146 worse | 0.930 ± 0.141 worse | 0.833 ± 0.125 worse | 0.829 ± 0.119 worse | 0.855 ± 0.132 worse | 0.872 ± 0.141 worse | 0.996 ± 0.185 worse |

## R2

| scheme | k-NN (Tanimoto) | LightGBM + Morgan | ChemProp + CheMeleon | Monroe + TabPFN 3.5 | Training median | k-NN (published) | SVM (published) | RF (published) | XGBoost (published) | MLP (published) | GNN (published) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.576 ± 0.104 worse | 0.618 ± 0.087 worse | 0.641 ± 0.085 worse | 0.672 ± 0.074 **best** | -0.013 ± 0.020 worse | 0.576 ± 0.104 worse | 0.641 ± 0.081 worse | 0.628 ± 0.080 worse | 0.612 ± 0.094 worse | 0.600 ± 0.091 worse | 0.448 ± 0.113 worse |
| Bemis-Murcko scaffold | 0.357 ± 0.211 worse | 0.441 ± 0.189 worse | 0.470 ± 0.185 worse | 0.514 ± 0.159 **best** | -0.029 ± 0.038 worse | 0.357 ± 0.211 worse | 0.479 ± 0.154 worse | 0.471 ± 0.145 worse | 0.438 ± 0.177 worse | 0.388 ± 0.193 worse | 0.248 ± 0.238 worse |
| Butina cluster | -0.017 ± 0.360 worse | 0.153 ± 0.217 worse | 0.175 ± 0.275 worse | 0.246 ± 0.245 **best** | -0.098 ± 0.145 worse | -0.017 ± 0.360 worse | 0.186 ± 0.189 tied | 0.204 ± 0.167 tied | 0.154 ± 0.244 worse | 0.063 ± 0.291 worse | -0.134 ± 0.376 worse |
| Diverse (MaxMin 25%) | 0.092 ± 0.247 worse | 0.195 ± 0.194 worse | 0.193 ± 0.217 worse | 0.283 ± 0.184 **best** | -0.042 ± 0.049 worse | 0.093 ± 0.305 worse | 0.221 ± 0.171 worse | 0.237 ± 0.151 worse | 0.171 ± 0.195 worse | 0.156 ± 0.195 worse | -0.041 ± 0.170 worse |
| UMAP cluster | -0.292 ± 0.278 worse | -0.120 ± 0.322 worse | -0.088 ± 0.420 tied | 0.029 ± 0.271 **best** | -0.210 ± 0.422 worse | -0.292 ± 0.278 worse | -0.048 ± 0.306 tied | -0.024 ± 0.258 tied | -0.159 ± 0.364 worse | -0.217 ± 0.422 worse | -0.426 ± 0.640 worse |
| Time | -0.239 ± 0.402 worse | -0.041 ± 0.326 worse | -0.007 ± 0.339 worse | 0.104 ± 0.255 **best** | -0.090 ± 0.102 worse | -0.239 ± 0.402 worse | 0.045 ± 0.239 tied | 0.066 ± 0.221 tied | -0.051 ± 0.322 worse | -0.089 ± 0.313 worse | -0.406 ± 0.514 worse |

## SPEARMAN

| scheme | k-NN (Tanimoto) | LightGBM + Morgan | ChemProp + CheMeleon | Monroe + TabPFN 3.5 | k-NN (published) | SVM (published) | RF (published) | XGBoost (published) | MLP (published) | GNN (published) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.753 ± 0.059 worse | 0.773 ± 0.059 worse | 0.794 ± 0.050 worse | 0.809 ± 0.043 **best** | 0.753 ± 0.059 worse | 0.792 ± 0.053 worse | 0.785 ± 0.050 worse | 0.769 ± 0.064 worse | 0.770 ± 0.055 worse | 0.681 ± 0.077 worse |
| Bemis-Murcko scaffold | 0.623 ± 0.125 worse | 0.657 ± 0.146 worse | 0.688 ± 0.132 tied | 0.708 ± 0.116 **best** | 0.623 ± 0.125 worse | 0.684 ± 0.119 tied | 0.681 ± 0.114 worse | 0.652 ± 0.134 worse | 0.644 ± 0.133 worse | 0.573 ± 0.144 worse |
| Butina cluster | 0.391 ± 0.148 worse | 0.460 ± 0.156 worse | 0.496 ± 0.166 tied | 0.516 ± 0.183 **best** | 0.391 ± 0.148 worse | 0.491 ± 0.176 tied | 0.471 ± 0.163 tied | 0.459 ± 0.169 worse | 0.440 ± 0.169 worse | 0.377 ± 0.172 worse |
| Diverse (MaxMin 25%) | 0.424 ± 0.137 worse | 0.464 ± 0.149 worse | 0.477 ± 0.148 worse | 0.524 ± 0.144 **best** | 0.428 ± 0.135 worse | 0.476 ± 0.154 worse | 0.487 ± 0.140 tied | 0.430 ± 0.152 worse | 0.452 ± 0.163 worse | 0.200 ± 0.230 worse |
| UMAP cluster | 0.187 ± 0.156 worse | 0.256 ± 0.238 worse | 0.313 ± 0.242 tied | 0.325 ± 0.258 **best** | 0.187 ± 0.156 worse | 0.296 ± 0.221 tied | 0.269 ± 0.232 tied | 0.237 ± 0.237 worse | 0.273 ± 0.215 tied | 0.231 ± 0.273 tied |
| Time | 0.264 ± 0.192 worse | 0.319 ± 0.196 worse | 0.363 ± 0.233 tied | 0.378 ± 0.230 **best** | 0.264 ± 0.192 worse | 0.350 ± 0.218 tied | 0.327 ± 0.216 worse | 0.304 ± 0.221 worse | 0.312 ± 0.217 worse | 0.275 ± 0.227 worse |
