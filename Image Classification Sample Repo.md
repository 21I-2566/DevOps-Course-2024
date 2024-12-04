We will use the [TensorFlow Image Classification Example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/keras/structured_data_classification) from TensorFlow's official GitHub repository. This example is a simple image classification using Keras.

### Containerizing a Machine Learning Algorithm with CI/CD

In this project, we will containerize a **TensorFlow-based image classification** model, which uses Keras for model creation and training. We will also set up a **CI/CD pipeline** using GitHub Actions.

## 1. Containerizing the TensorFlow Image Classification Model

### Steps Taken:
We are going to take the **TensorFlow Image Classification Example** repository and containerize it.

### Project Structure:
```
tensorflow-image-classification/
│
├── Dockerfile
├── requirements.txt
├── model.py
├── train.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── data/
│   └── (store training datasets here)
└── README.md
```

### Dockerfile:
The `Dockerfile` will define the environment in which our model runs.

```dockerfile
# Use an official TensorFlow runtime as a base image
FROM tensorflow/tensorflow:2.11.0-py3

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (if needed for APIs)
EXPOSE 5000

# Define the command to run the training script
CMD ["python", "train.py"]
```

### requirements.txt:
List the required libraries in the `requirements.txt` file. The key dependencies for this model are **TensorFlow** and others.

```text
tensorflow==2.11.0
numpy
matplotlib
pandas
scikit-learn
```

### model.py:
Define the machine learning model in this Python file. For this example, we’ll use a basic **CNN** for image classification.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(32, 32, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model
```

### train.py:
This script trains the model using a dataset, such as **CIFAR-10**.

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from model import create_model

# Load and preprocess the data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Create the model
model = create_model()

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
```

### 2. Check Docker Setup:

To verify that the Docker setup works, follow these steps:

1. **Build the Docker image**:
   ```bash
   docker build -t tf-image-classification .
   ```

2. **Run the container**:
   ```bash
   docker run -v $(pwd)/data:/usr/src/app/data tf-image-classification
   ```

This will mount the `data` directory on your local machine to the container and allow the model to access the training data.

## 3. Setting Up a CI/CD Pipeline

For continuous integration and deployment, we will set up GitHub Actions to automate testing, building, and deployment of the container.

### GitHub Actions Workflow File:
Save this file as `.github/workflows/ci-cd.yml` in the root of the repository.

```yaml
name: CI/CD Pipeline for TensorFlow Image Classification

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  lint-test-and-build:
    runs-on: ubuntu-latest

    steps:
      # 1. Checkout code
      - name: Checkout code
        uses: actions/checkout@v3

      # 2. Set up Python
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.9

      # 3. Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      # 4. Lint the code (Optional but recommended)
      - name: Lint code
        run: |
          pip install flake8
          flake8 .

      # 5. Run tests
      - name: Run tests
        run: |
          pytest

      # 6. Build Docker image
      - name: Build Docker image
        run: |
          docker build -t tf-image-classification .

  deploy:
    needs: lint-test-and-build
    runs-on: ubuntu-latest

    steps:
      # 1. Checkout code
      - name: Checkout code
        uses: actions/checkout@v3

      # 2. Set up Docker
      - name: Set up Docker
        uses: docker/setup-buildx-action@v2

      # 3. Push Docker image to Docker Hub (or any other registry)
      - name: Push Docker image
        run: |
          docker login -u ${{ secrets.DOCKER_USERNAME }} -p ${{ secrets.DOCKER_PASSWORD }}
          docker tag tf-image-classification yourdockerusername/tf-image-classification:latest
          docker push yourdockerusername/tf-image-classification:latest
```

### Explanation of the Workflow:

- **Linting and Testing:**
  - Installs dependencies and lints the code using **flake8**.
  - Runs tests using **pytest**.

- **Docker Build:**
  - Builds the Docker image containing the TensorFlow-based image classification model.

- **Deployment:**
  - Pushes the Docker image to Docker Hub (or any other container registry) for deployment after the image passes tests.

## 4. Outcome:

- **Automated CI/CD Pipeline**:  
  The GitHub Actions pipeline will automatically build, test, and deploy the containerized image classification model whenever code is pushed to the `main` branch or when a pull request is made.

- **Dockerized Environment**:  
  The TensorFlow image classification model is encapsulated in a Docker container, ensuring a consistent environment for model training and inference.

## 5. Conclusion

This project demonstrates how to containerize a **TensorFlow-based image classification model** using Docker, automate testing, and deploy the model through a CI/CD pipeline with **GitHub Actions**. By containerizing the model, we ensure that the environment is consistent across various stages of development, while the CI/CD pipeline automates testing, building, and deployment, making the process efficient and reliable.
