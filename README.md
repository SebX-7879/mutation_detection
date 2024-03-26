# PIK3CA Mutation Detection in Breast Cancer
**Project** : Weakly supervised learning for the detection of PIK3CA mutations in breast cancer.

**Description** : This project focuses on detecting the PIK3CA mutation in breast cancer using histopathology slide images. These high-resolution images offer a detailed view of tissue samples, essential for identifying cellular abnormalities. Despite limited examples and slide-level labels, we explore the potential of deep neural architectures to overcome these challenges and provide robust and reasonable predictions.
We employ the [CHOWDER](https://arxiv.org/pdf/1802.02212.pdf) model proposed by [Owkin](https://www.owkin.com/), a deep learning approach tailored for weakly supervised learning in medical image analysis. Our goal is to enhance the model's performance in classifying whole-slide images as PIK3CA mutant or wild-type.


#### Author 
- **[Sébastien Mandela]** - *Initial work* - [SebX-7879](https://github.com/SebX-7879)
- Challenge by [**Owkin**](https://www.owkin.com/)

### Project structure

```
mutation_detection/
├── data/
│   ├── supplementary_data/
│   │   ├── test_metadata.csv
│   │   └── train_metadata.csv
│   ├── test_input/
│   │   ├── images/
│   │   └── moco_features/
│   ├── train_input/
│   │   ├── images/
│   │   └── moco_features/
│   └── train_output.csv
├── datasets/
│   ├── __init__.py
│   ├── core.py
├── figures/
├── logs/
├── models/
│   ├── utils/
│   ├── __init__.py
│   └── chowder.py
├── test_output/
├── trainer/
├── utils/
├── .gitignore
├── baseline.ipynb
├── download_data.py
├── LICENSE
├── main.py
├── working_notebook.ipynb
├── README.md
└── requirements.txt
```

### Libraries
Install the required libraries using :
```bash
pip install -r requirements.txt
```

### Data
To download the data, run :
```bash
python download_data.py
```
The `data` is expected to have the same structure as above. Due to storage restrictions (max number of authorized files), we did not include the images in the loading process, but only the `moco_features` and `metadata` files. 


### Training the model 

The model is trained using the `main.py` script. The script uses the `Trainer` class from the `trainer` module to train the model. The `Trainer` class is responsible for loading the data, training the model, and evaluating the model on the test set. The `Trainer` class uses the `CHOWDER` model from the `models` module to train the model. Hyperparameters such as the model's parameters, learning rate, batch size, and number of epochs can be directly set in the `main.py` script.

To train the model, run :
```bash 
python main.py
```

Additionally, a working notebook is available in the `working_notebook.ipynb` file. It contains the entire pipeline, from data preprocessing, model training, to evaluation. It makes use of Kfold cross-validation, and thus benefits from multiple instance learners, providing a more robust training and evaluation process than the `main.py` script.

### License
This project is derived from another under-license project.
See the [LICENSE](.\LICENSE.txt) file for details.
