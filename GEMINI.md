# Gemini Project Overview: EuroSAT Classifier

This document provides a comprehensive overview of the `eurosat_classifier` project, designed to serve as a guide for Gemini AI interactions.

## 1. Project Purpose & Technology

This project is a machine learning operations (MLOps) pipeline for land use and land cover (LULC) classification using the EuroSAT satellite imagery dataset. It is structured as a course project for the DTU MLOps course.

The core technologies used are:

*   **Programming Language:** Python 3.12
*   **ML Framework:** PyTorch, with PyTorch Lightning for streamlined training.
*   **Model:** A pretrained ResNet18 model from the `timm` library.
*   **Configuration:** Hydra for managing configurations (e.g., for data, model, training).
*   **Data Versioning:** DVC is used, although the full setup is not completely detailed in the provided files.
*   **API:** FastAPI for serving the model (though the API implementation is currently empty).
*   **Dependency Management:** `uv` is used for managing Python packages.
*   **Task Runner:** `invoke` is used to define and run common project tasks.
*   **Containerization:** Docker is used to create environments for training and API deployment.
*   **CI/CD:** GitHub Actions are set up for continuous integration, running tests on multiple OS and Python versions.
*   **Linting & Formatting:** `ruff` is used for code quality.
*   **Testing:** `pytest` is used for unit testing, with `coverage` for test coverage measurement.

## 2. Key Files and Directories

*   `src/eurosat_classifier/`: The main source code for the project.
    *   `train.py`: The main script for training the model. It uses Hydra for configuration and logs the training process.
    *   `model.py`: Defines the `EuroSATModel` which is a PyTorch `nn.Module` wrapping a `timm` model.
    *   `data.py`: Contains functions for creating PyTorch `DataLoader`s for the EuroSAT dataset, including transformations.
    *   `api.py`: Intended for the FastAPI application to serve the model (currently empty).
    *   `conf/`: Hydra configuration files for the model, data, and training.
*   `requirements.txt` & `pyproject.toml`: Defines the Python dependencies for the project. `pyproject.toml` is the source of truth.
*   `tasks.py`: An `invoke` script that defines command-line tasks for common operations like training, testing, and building Docker images.
*   `dockerfiles/`: Contains Dockerfiles for creating training and API environments.
*   `.github/workflows/`: GitHub Actions workflows for CI.
    *   `tests.yaml`: Runs tests on pull requests.
*   `tests/`: Unit tests for the project.

## 3. How to Run the Project

### Setup

1.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

### Training

To train the model, you can use the `invoke` task:

```bash
invoke train
```

This will run the `src/eurosat_classifier/train.py` script with the default configuration defined in `src/eurosat_classifier/conf/`.

### Testing

To run the test suite:

```bash
invoke test
```

This will execute the `pytest` tests and print a coverage report.

### Docker

To build the Docker images for training and the API:

```bash
invoke docker-build
```

## 4. Development Conventions

*   **Configuration:** All configurations are managed through Hydra. To change hyperparameters, modify the YAML files in `src/eurosat_classifier/conf/`.
*   **Linting:** The project uses `ruff` for linting. It's recommended to run `ruff check .` before committing changes.
*   **Pre-commit:** The project uses `pre-commit` hooks, configured in `.pre-commit-config.yaml`, to automatically run checks before each commit.
*   **Continuous Integration:** All pull requests are automatically tested using the GitHub Actions workflow defined in `.github/workflows/tests.yaml`.
*   **Task Runner:** Use `invoke -l` to see a list of all available tasks defined in `tasks.py`.
