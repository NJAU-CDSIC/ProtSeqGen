# **ProtSeqGen** Code Repository

This is a directory for storing the **ProtSeqGen** model code and data. **ProtSeqGen** is a deep learning framework for protein sequence design from backbone structures. 

---

The folders in the ProtSeqGen repository:

- **Datasets**:

   a. **CATH4.2**: A non-redundant protein structure dataset from the CATH 4.2 database.

   b. **TS50**: A standard benchmark dataset consisting of 50 protein structures.
  
   c. **TS500**: A large-scale benchmark dataset containing 500 protein structures.

   d. **TS45**: A benchmark dataset consisting of 45 protein structures.

   e. **IDRome**: An additional benchmark dataset for evaluating performance on the IDRome subset.
  
- **ProtSeqGen_code**: Main code file for the ProtSeqGen model.

- **SOTA**:Comparative methods used in the contrast experiments:
  
   SPROF: https://github.com/biomed-AI/SPROF
  
   ProDCoNN: https://github.com/wells-wood-research/timed-design
  
   GraphTrans:https://github.com/jingraham/neurips19-graph-protein-design
  
   GVP: https://github.com/drorlab/gvp
  
   GCA: https://github.com/chengtan9907/gca-generative-protein-design
  
   AlphaDesign: https://github.com/Westlake-drug-discovery/AlphaDesign
  
   ProteinMPNN: https://github.com/dauparas/ProteinMPNN
  
   Frame2seq: https://github.com/dakpinaroglu/Frame2seq
  
   PiFold: https://github.com/A4Bio/PiFold
  
   GeoSeqBuilder: https://github.com/PKUliujl/GeoSeqBuilder

- **Scripts**: Contains auxiliary scripts for data processing and visualization.

   a. **plot**: Compares generated protein structures with original 3D structures and produces visualization plots.
  
   b. **split**: Divides the CATH dataset into subsets based on sequence similarity.

---



###  **Step-by-step Running:**

## 1.Environment Installation
It is recommended to use a conda environment (Python 3.10), mainly installing the following dependencies:

- **pytorch(2.0.0)、pytorch-cuda(11.8)、scipy(1.10.1)、scikit-learn(1.2.2)、pandas(2.0.0)、numpy(1.23.5)**

See environment.yml for details. Use the following command to install the runtime environment:

```
conda env create -f environment.yml
conda activate protseqgen
```

## 2. Datasets

Place the datasets into the Datasets folder. For CATH4.2, you can download it by running the provided script:

```
cd Datasets/CATH4.2
bash getCATH.sh
```

## 3. Training and Testing

After setting up the environment and datasets, you can train and test the model by running the provided scripts.

- Train the model using the CATH4.2 dataset:

```
bash ProtSeqGen_code/Model_training/training.sh
```

- Test on TS50, Make sure the Model folder contains the trained model parameters before running these tests.:

```
bash ProtSeqGen_code/Model_testing/test_50.sh
```

- Test on TS500:

```
bash ProtSeqGen_code/Model_testing/test_500.sh
```

## 4.Scripts Usage

- **plot.py**: Compare generated structures with original 3D structures and generate visualization plots:

```
python Scripts/plot.py --pred pred.pdb --ref ref.pdb
```

- **split.py**: Divide the CATH dataset based on sequence similarity:

```
python Scripts/split.py --input Datasets/CATH4.2 --output Datasets/CATH_split
```

## 5. Installation

Download the code:

```
git clone https://github.com/NJAU-CDSIC/ProtSeqGen.git
```

## 6.Citation

If you use ProtSeqGen in your research, please cite:

```
@misc{ProtSeqGen2025,
  title={ProtSeqGen: Protein Sequence Generation from Backbone Structures},
  author={NJAU-CDSIC},
  year={2025},
  howpublished={\url{https://github.com/NJAU-CDSIC/ProtSeqGen}}
}
```

  

