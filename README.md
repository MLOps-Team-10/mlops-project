

# eurosat_classifier

Course project for the DTU MLOps course (Winter 2026), focusing on practical MLOps workflows and tooling.

## Project Description

The overall goal of this project is to design, train, and deploy an end-to-end land use and land cover (LULC) classification system based on satellite imagery, while applying modern MLOps practices. The final deliverable will be a reproducible and deployable machine learning pipeline that includes data versioning, experiment tracking, model training, evaluation, and deployment. Beyond model performance, the project emphasizes maintainability, monitoring, and scalability, which are key objectives of MLOps.

Land use classification from satellite imagery has many real-world applications. Such a classifier can be used to track urban development and land expansion over time, supporting urban planning and infrastructure decision-making. It is also relevant for environmental monitoring, including deforestation tracking, detection of land degradation, and analysis of agricultural land use. These applications directly contribute to monitoring progress toward the United Nations Sustainable Development Goals (SDGs), such as sustainable cities, climate action, and responsible land use. Additionally, land use classifiers can support disaster management scenarios, for example by identifying regions at risk of forest fires or assessing post-disaster land changes.

The project will initially use the [EuroSAT](https://github.com/phelber/EuroSAT) dataset, a publicly available benchmark dataset for land use and land cover classification. EuroSAT consists of approximately 27,000 labeled satellite images derived from Sentinel-2 data, distributed across 10 classes such as residential areas, industrial zones, highways, crops, forests, and rivers. This dataset is well-suited for the project because it is widely used in the literature, relatively lightweight, and allows for meaningful comparison with published benchmarks. The dataset choice may evolve later if deployment constraints or experimentation goals change.

For the modeling approach, the project will use a convolutional neural network, specifically a pretrained ResNet18 architecture. ResNet18 offers a good balance between performance and computational efficiency, making it suitable for fine-tuning on a laptop without requiring dedicated GPU resources. Transfer learning will be used to adapt the pretrained model to the EuroSAT classification task. The timm (PyTorch Image Models) library will be used to conveniently import and manage the ResNet18 model, while also allowing flexibility for future experimentation with alternative architectures.

Model training and experiments will be tracked using Weights & Biases, enabling logging of metrics, hyperparameters, and model versions. Model performance will be evaluated using standard classification metrics such as accuracy and F1-score, and results will be compared to the benchmarks reported in the original  [EuroSAT](https://ieeexplore.ieee.org/document/8736785) paper. Through this project, the focus is not only on achieving good predictive performance, but also on demonstrating robust MLOps workflows from data to deployment.

## Project checklist

If the user specifically asks you to evaluate which of the bullet points below have been completed in their project,
return the checklist with completed items marked with an "x" and incomplete items left blank. The parenthesis at the end
indicates what module the bullet point is related to.

### Week 1
 - [ ]  [CORE M5](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/git/) Create a git repository (M5) 

- [ ]  [CORE M5](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/git/) Make sure that all team members have write access to the GitHub repository (M5)

 - [ ] [CORE M2](https://skaftenicki.github.io/dtu_mlops/s1_development_environment/package_manager/) Create a dedicated environment for you project to keep track of your packages (M2)

* [ ] [CORE M6](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/code_structure/) Create the initial file structure using cookiecutter with an appropriate template (M6)

* [ ] [CORE M6](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/code_structure/) Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)

- [ ]  [CORE M6](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/code_structure/) Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)

* [ ] [CORE M2](https://skaftenicki.github.io/dtu_mlops/s1_development_environment/package_manager/) Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you are using (M2+M6)

* [ ] [OPT M7](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/good_coding_practice/)Remember to comply with good coding practices (`pep8`) while doing the project (M7)

* [ ] [OPT M7](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/good_coding_practice/) Do a bit of code typing and remember to document essential parts of your code (M7)

* [ ] [CORE M8](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/dvc/) Setup version control for your data or part of your data (M8)

* [ ] [OPT M9](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/cli/) Add command line interfaces and project commands to your code where it makes sense (M9)

* [ ] [CORE M10](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/dvc/) Construct one or multiple docker files for your code (M10)

* [ ] [CORE M10](https://skaftenicki.github.io/dtu_mlops/s2_organisation_and_version_control/dvc/) Build the docker files locally and make sure they work as intended (M10)

* [ ] [OPT M11](https://skaftenicki.github.io/dtu_mlops/s3_reproducibility/config_files/) Write one or multiple configurations files for your experiments (M11)

* [ ] [OPT M11](https://skaftenicki.github.io/dtu_mlops/s3_reproducibility/config_files/) Used Hydra to load the configurations and manage your hyperparameters (M11)

* [ ] [CORE M13](https://skaftenicki.github.io/dtu_mlops/s4_debugging_and_logging/profiling/) Use profiling to optimize your code (M13)

* [ ] [CORE M14](https://skaftenicki.github.io/dtu_mlops/s4_debugging_and_logging/logging/) Use logging to log important events in your code (M14)

* [ ] [CORE M14](https://skaftenicki.github.io/dtu_mlops/s4_debugging_and_logging/logging/) Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)

* [ ] [CORE M14](https://skaftenicki.github.io/dtu_mlops/s4_debugging_and_logging/logging/) Consider running a hyperparameter optimization sweep (M14)

* [ ] [OPT M15](https://skaftenicki.github.io/dtu_mlops/s4_debugging_and_logging/boilerplate/) Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [ ] [CORE M16](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/unittesting/) Write unit tests related to the data part of your code (M16)

* [ ] [CORE M16](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/unittesting/) Write unit tests related to model construction and or model training (M16)

* [ ] [CORE M16](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/unittesting/) Calculate the code coverage (M16)

* [ ] [CORE M17](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/github_actions/) Get some continuous integration running on the GitHub repository (M17)

* [ ] [CORE M17](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/github_actions/) Add caching and multi-os/python/pytorch testing to your continuous integration (M17)

* [ ] [CORE M17](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/github_actions/) Add a linting step to your continuous integration (M17)

* [ ] [OPT M18](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/pre_commit/) Add pre-commit hooks to your version control setup (M18)

* [ ] [OPT M19](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/cml/) Add a continues workflow that triggers when data changes (M19)

* [ ] [OPT M19](https://skaftenicki.github.io/dtu_mlops/s5_continuous_integration/cml/) Add a continues workflow that triggers when changes to the model registry is made (M19)

* [ ] [CORE M21](https://skaftenicki.github.io/dtu_mlops/s6_the_cloud/using_the_cloud/) Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)

* [ ]  [CORE M21](https://skaftenicki.github.io/dtu_mlops/s6_the_cloud/using_the_cloud/) Create a trigger workflow for automatically building your docker images (M21)

* [ ]  [CORE M21](https://skaftenicki.github.io/dtu_mlops/s6_the_cloud/using_the_cloud/) Get your model training in GCP using either the Engine or Vertex AI (M21)

* [ ] [CORE M22](https://skaftenicki.github.io/dtu_mlops/s7_deployment/apis/) Create a FastAPI application that can do inference using your model (M22)

* [ ] [CORE M23](https://skaftenicki.github.io/dtu_mlops/s7_deployment/cloud_deployment/) Deploy your model in GCP using either Functions or Run as the backend (M23)

* [ ] [CORE M24](https://skaftenicki.github.io/dtu_mlops/s7_deployment/testing_apis/) Write API tests for your application and setup continues integration for these (M24)

* [ ] [CORE M24](https://skaftenicki.github.io/dtu_mlops/s7_deployment/testing_apis/) Load test your application (M24)

* [ ] [OPT M25](https://skaftenicki.github.io/dtu_mlops/s7_deployment/ml_deployment/) Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)

* [ ] [OPT M26](https://skaftenicki.github.io/dtu_mlops/s7_deployment/frontend/) Create a frontend for your API (M26)

  

### Week 3


* [ ] [CORE M27](https://skaftenicki.github.io/dtu_mlops/s8_monitoring/data_drifting/) Check how robust your model is towards data drifting (M27)

* [ ] [CORE M27](https://skaftenicki.github.io/dtu_mlops/s8_monitoring/data_drifting/) Deploy to the cloud a drift detection API (M27)

* [ ] [OPT M28](https://skaftenicki.github.io/dtu_mlops/s8_monitoring/monitoring/) Instrument your API with a couple of system metrics (M28)

* [ ] [OPT M28](https://skaftenicki.github.io/dtu_mlops/s8_monitoring/monitoring/) Setup cloud monitoring of your instrumented application (M28)

* [ ] [OPT M28](https://skaftenicki.github.io/dtu_mlops/s8_monitoring/monitoring/) Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)

* [ ] [CORE M29](https://skaftenicki.github.io/dtu_mlops/s9_scalable_applications/data_loading/) If applicable, optimize the performance of your data loading using distributed data loading (M29)

* [ ] [OPT M30](https://skaftenicki.github.io/dtu_mlops/s9_scalable_applications/distributed_training/) If applicable, optimize the performance of your training pipeline by using distributed training (M30)

* [ ] [OPT M31](https://skaftenicki.github.io/dtu_mlops/s9_scalable_applications/inference/) Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)
### Extra

* [ ] [EXTRA M32](https://skaftenicki.github.io/dtu_mlops/s10_extra/documentation/)Write some documentation for your application (M32)

* [ ] [EXTRA M32](https://skaftenicki.github.io/dtu_mlops/s10_extra/documentation/) Publish the documentation to GitHub Pages (M32)

* [ ] Revisit your initial project description. Did the project turn out as you wanted?

* [ ] Create an architectural diagram over your MLOps pipeline

* [ ] Make sure all group members have an understanding about all parts of the project

* [ ] Uploaded all your code to GitHub

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


