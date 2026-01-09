````markdown
# eurosat_classifier

## Project Description

The overall goal of this project is to design, train, and deploy an end-to-end land use and land cover (LULC) classification system based on satellite imagery, while applying modern MLOps practices. The final deliverable will be a reproducible and deployable machine learning pipeline that includes data versioning, experiment tracking, model training, evaluation, and deployment. Beyond model performance, the project emphasizes maintainability, monitoring, and scalability, which are key objectives of MLOps.

Land use classification from satellite imagery has many real-world applications. Such a classifier can be used to track urban development and land expansion over time, supporting urban planning and infrastructure decision-making. It is also relevant for environmental monitoring, including deforestation tracking, detection of land degradation, and analysis of agricultural land use. These applications directly contribute to monitoring progress toward the United Nations Sustainable Development Goals (SDGs), such as sustainable cities, climate action, and responsible land use. Additionally, land use classifiers can support disaster management scenarios, for example by identifying regions at risk of forest fires or assessing post-disaster land changes.

The project will initially use the [EuroSAT](https://github.com/phelber/EuroSAT) dataset, a publicly available benchmark dataset for land use and land cover classification. EuroSAT consists of approximately 27,000 labeled satellite images derived from Sentinel-2 data, distributed across 10 classes such as residential areas, industrial zones, highways, crops, forests, and rivers. This dataset is well-suited for the project because it is widely used in the literature, relatively lightweight, and allows for meaningful comparison with published benchmarks. The dataset choice may evolve later if deployment constraints or experimentation goals change.

For the modeling approach, the project will use a convolutional neural network, specifically a pretrained ResNet18 architecture. ResNet18 offers a good balance between performance and computational efficiency, making it suitable for fine-tuning on a laptop without requiring dedicated GPU resources. Transfer learning will be used to adapt the pretrained model to the EuroSAT classification task. The timm (PyTorch Image Models) library will be used to conveniently import and manage the ResNet18 model, while also allowing flexibility for future experimentation with alternative architectures.

Model training and experiments will be tracked using Weights & Biases, enabling logging of metrics, hyperparameters, and model versions. Model performance will be evaluated using standard classification metrics such as accuracy and F1-score, and results will be compared to the benchmarks reported in the original  [EuroSAT](https://ieeexplore.ieee.org/document/8736785) paper. Through this project, the focus is not only on achieving good predictive performance, but also on demonstrating robust MLOps workflows from data to deployment.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````
