## Hypergradient Descent with Polyak Momentum

This is the official implementation of [Provable and Practical Online Learning Rate Adaptation with
Hypergradient Descent](https://arxiv.org/pdf/2502.11229).

**Reproducing the results**

To reproduce the experiments in the paper,

1. Ensure that `scikit_learn`,  `numpy`,  `scipy`,  `seaborn,` and `matplotlib` are installed

   See `requirements.txt` for more details

2. Download datasets from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). Check the list at the end for benchmark files.

3. Place the datasets in `problems` directory

4. Execute the following commands

   ```
   chmod +x ./*.sh
   ./get_svm_data.sh
   ./get_logistic_data.sh
   ```

   to reproduce the experiments. The following logs will be printed to the screen

   ```
   Running a1a
   ================================================
                 Solver [S]   nFvalCall   nGradCall
                     GD [0]           0        1000
                  GD-HB [0]           0        1000
                AGD-CVX [0]           0        1000
               AGD-SCVX [0]           0        1000
                   Adam [1]           0         566
                AdaGrad [0]           0        1000
                   BFGS [1]          91          91
              L-BFGS-M1 [0]        1124        1124
              L-BFGS-M3 [1]         522         522
              L-BFGS-M5 [1]         531         531
             L-BFGS-M10 [1]         328         328
               HDM-Best [1]         275         276
   ================================================
   ```

and figures will be saved in `figures` directory.

**Tested LIBSVM instances**

```
a1a, a2a, a3a, a4a, a5a, a6a, a7a, a8a, a9a, australian_scale, fourclass_scale, german, gisette_scale, gisette_scale, heart_scale, ijcnn1, ionosphere_scale, leu, liver-disorders_scale, mushrooms, phishing, skin_nonskin, sonar_scale, splice_scale, svmguide1, svmguide3, w1a, w2a, w3a, w4a, w5a, w6a, w7a, w8a
```

**Code maintainance and contact**

gwz@stanford.edu

