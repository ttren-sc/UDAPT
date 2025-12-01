## 1. Overview of UDAPT
UDAPT is a domain-adaptive deep learning framework for bulk RNA-seq deconvolution.
The method combines pseudo-bulk pretraining with adversarial domain adaptation to reduce distributional discrepancies between simulated and real bulk RNA-seq, enabling more accurate estimation of cell-type proportions.

## 2. Installation & Environment Setup
### 2.1 UDAPT can be installed on Linux, macOS, or Windows.
```
# Clone the repository
git clone https://github.com/ttren-sc/UDAPT.git
cd UDAPT

# Create a conda environment
conda create -n udapt python=3.7
conda activate udapt

# Install PyTorch 1.9.1 with CUDA 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install additional packages and dependencies
pip install -r requirements.txt
```
### 2.2 GPU support
UDAPT uses PyTorch, and CUDA acceleration is automatically used if available.

- Check CUDA:
```
python -c "import torch; print(torch.cuda.is_available())
```

## 3. Example Datasets
We provide two fully processed pseudo-bulk RNA-seq datasets simulated with known cell type fractions, allowing users to run UDAPT out-of-the-box:

- A training dataset with the sample size of 5000
  
  - Source of the dataset: Marrow scRNA-seq data by Droplet-seq

  - Format: h5

  - Location: 
  ```
  data/training_data/Marrow_droplet.h5
  ```
  - Read dataset:
  ```
  import pandas as pd
  
  train = pd.HDFStore('./data/training_data/Marrow_droplet.h5')
  train_X = train['X_src']  # rows: pseudo-bulk samples, columns: genes
  train_y = train['y_src']  # rows: pseudo-bulk samples, columns: cell types
  ``` 

- A test dataset with the sample size of 500

  - Source of the dataset: Marrow scRNA-seq data by Smart-seq

  - Format: h5

  - Location: 
  ```
  data/test_data/Marrow_smart.h5
  ```
  - Read dataset:
  ```
  import pandas as pd
  
  test = pd.HDFStore('./data/test_data/Marrow_smart.h5')
  test_X = test['X_tgt']  # rows: pseudo-bulk samples, columns: genes
  test_y = test['y_tgt']  # rows: pseudo-bulk samples, columns: cell types
  ``` 

## 4. Input/Output Format
### 4.1 Model Inputs
#### 4.1.1 scRNA-seq Format (csv / tsv)
Required fields:

- Rows: Cells

- Columns: Genes

- Elements: Gene expression values

Example:

C\G | Gene1 | Gene2 | Gene3 | Gene4
---|---|---|---|---
Cell1 | 2 | 10 | 1 | 14
Cell2 | 5 | 9 | 0 | 16
Cell3 | 4 | 12 | 0 | 7
Cell4 | 0 | 8 | 2 | 10

#### 4.1.2  Cell type Annotation Format (csv / tsv)
Required fields:

- Rows: Cells

- Columns: Genes

- Elements: Gene expression values

Example:

C\CT | CellType
---|---
Cell1 | CT1
Cell2 | CT3 
Cell3 | CT2
Cell4 | CT4

#### 4.1.3 Bulk RNA-seq Format (csv / tsv)
Description:

- Rows: Bulk RNA-seq samples

- Columns: Genes

- Elements: Gene expression values

Example:

S\G | Gene1 | Gene2 | Gene3 | Gene4
---|---|---|---|---
Sample1 | 120 | 560 | 30 | 240
Sample2 | 98 | 610 | 25 | 220
Sample3 | 135 | 590 | 41 | 260

### 4.3 Model Outputs

UDAPT outputs:

- props: Cell-type proportion matrix

- signature: Signature matrix

Output file example of the main result props:

sample | CT1 | CT2 | CT3 | CT4
---|---|---|---|---
Sample1 | 0.21 | 0.54 | 0.12 | 0.13
Sample2 | 0.19 | 0.56 | 0.10 | 0.15

## 5. Quick Start: An end-to-end demo
Using example pseudo-bulk RNA-seq data in the 3th section.

This demo reproduces an entire UDAPT workflow in three commands.

```
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

## 6. Detailed Usage Guidance
### 6.1 Simulate pseudo-bulk RNA-seq data using scRNA-seq data
```
python udapt/train_pretrain.py \
    --data data/pseudobulk/ \
    --config udapt/config_default.yaml
```

### 6.2 Domain Adaptation
```
python udapt/train_adapt.py \
    --real data/example_bulk/ \
    --pseudo data/pseudobulk/ \
    --config udapt/config_default.yaml
```

### 6.3 Predicting new bulk samples
```
python udapt/predict.py \
    --model checkpoints/udapt_adapt.pth \
    --input new_bulk.csv \
    --output predictions.csv
```

## 7. Runtime Requirements
UDAPT runs efficiently on the GPU.

The GPU runtime of end-to-end Marrow demo:

Task | GPU Runtime
---|---
Pretraining	| 112.33 s
Domain adaptation	| 27.11 s
Predict 500 bulk samples	| 0.002 s
Total time | 139.442 s

## 8. Licence and Attribution
This project is based on [TAPE-main](https://github.com/poseidonchan/TAPE), specifically using the following components:
- Data simulation module (`simulation.py`)
- Data preprocessing functions (`utils.py`)
- Core neural network architecture (`model.py`)

### 8.1 Original License
The original project is licensed under the [GNU General Public License v3.0](https://github.com/poseidonchan/TAPE/blob/main/LICENSE).

### 8.2 Derivative Works License
This derivative work is also licensed under the GNU General Public License v3.0. For details, see the [LICENSE](https://github.com/ttren-sc/UDAPT/blob/master/LICENSE) file.

### 8.3 Main change Notes
We have made the following major improvements to the original code:
1. Expanded the architecture to support further domain adaptation 
2. Optimized the model training process
3. Add two performance evaluation metrics: Root Mean Square Error (RMSE) and Pearson Correlation Coefficient (Pearson)

## 9. Contact

For inquiries or issues:

📧 ttren@stu.hit.edu.cn