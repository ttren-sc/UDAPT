# UDAPT
UDAPT is a depp learning-based deconvolution model for gene expression deconvolution.

## Pre-conditions
- Python 3.7
- pip or conda
- torch(gpu)

## 🛠️ Installation

```bash
# 克隆仓库
git clone https://github.com/ttren-sc/UDAPT.git

# 进入项目目录
cd UDAPT

# 安装依赖
pip install -r requirements.txt
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

### Change Notes
We have made the following major improvements to the original code:
1. Expanded the architecture to support further domain adaptation 
2. Optimized the model training process


