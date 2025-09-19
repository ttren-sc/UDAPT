# UDAPT
UDAPT is a depp learning-based deconvolution model for gene expression deconvolution.

## Pre-conditions
- Python == 3.7
- pip or conda
- pytorch (gpu)

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/ttren-sc/UDAPT.git

# 
cd UDAPT

# Install python packages and their dependencies
pip install -r requirements.txt

# 1. Data process: scRNA-seq dataset sc_data,and bulk RNA-seq data testX
## 1.1. simulate pseudo-bulk RNA-seq data using sc_data
trainX, trainy = generate_simulated_data(sc_data, outname=None,
                            d_prior=None,
                            n=500, samplenum=5000,
                            random_state=None, sparse=True, sparse_prob=0.5,
                            rare=False, rare_percentage=0.4)
## 1.2. data preprocess
trainX, testX, inter_genes = ProcessInputData(trainX, testX, sep=None, datatype='counts', variance_threshold=0.98,scaler="mms", genelenfile=None)

# 2. Predict cell type proportion
props, signature = predict(trainX, trainy, testX, inter_genes, n, k, shape = 128, batch_size=128, seed=0, run=1, iters_pre =1000, iters_fine=200, lr_pre = 1e-4, lr_train = 1e-4)
```

## Licence and Attribution
This project is based on [TAPE-main](https://github.com/poseidonchan/TAPE), specifically using the following components:
- Data simulation module (`simulation.py`)
- Data preprocessing functions (`utils.py`)
- Core neural network architecture (`model.py`)

### Original License
The original project is licensed under the [GNU General Public License v3.0](https://github.com/poseidonchan/TAPE/blob/main/LICENSE).

### Derivative Works License
This derivative work is also licensed under the GNU General Public License v3.0. For details, see the [LICENSE](https://github.com/ttren-sc/UDAPT/blob/master/LICENSE) file.

### Main change Notes
We have made the following major improvements to the original code:
1. Expanded the architecture to support further domain adaptation 
2. Optimized the model training process
3. Add two performance evaluation metrics: Root Mean Square Error (RMSE) and Pearson Correlation Coefficient (Pearson)