# PyTorch for Deep Learning Professional Certificate

> Deep learning pipelines with PyTorch. Covers CNNs, Transformers, Transfer Learning, and deployment with ONNX & MLflow.

![Status](https://img.shields.io/badge/Status-Complete-success)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![DeepLearning.AI](https://img.shields.io/badge/DeepLearning.AI-Certificate-yellow)

## 📖 Overview
This repository contains my coursework, projects, and notes for the **PyTorch for Deep Learning Professional Certificate** offered by DeepLearning.AI.

The curriculum covers the entire deep learning workflow using PyTorch—from building neural networks from scratch and optimizing data pipelines to deploying advanced architectures like Transformers and diffusion models using tools like MLflow and ONNX.

## 📂 Repository Structure

The content is organized by the three main courses in the specialization:

### 1️⃣ Course 1: PyTorch Fundamentals
* **Focus:** Core building blocks of PyTorch, tensors, auto-differentiation, and training loops.
* **Key Concepts:**
    * Tensor manipulation and shapes.
    * Building fully connected Neural Networks from scratch.
    * Implementing custom `Dataset` and `DataLoader` classes.
    * Binary and Multi-class classification.

### 2️⃣ Course 2: PyTorch Techniques and Ecosystem Tools
* **Focus:** Improving model performance, efficiency, and leveraging the wider ecosystem.
* **Key Concepts:**
    * **TorchVision & Hugging Face:** Handling image and text data.
    * **Optimization:** Hyperparameter tuning with Optuna.
    * **Efficiency:** Profiling models and speeding up training with PyTorch Lightning.
    * **Transfer Learning:** Fine-tuning pre-trained models.

### 3️⃣ Course 3: Advanced Architectures and Deployment
* **Focus:** Modern architectures and getting models ready for production.
* **Key Concepts:**
    * **Advanced Architectures:** ResNets, DenseNets, Siamese Networks, and Transformers.
    * **Deployment:** Model export with ONNX and tracking with MLflow.
    * **Optimization:** Pruning, Quantization, and model compression.

---

## 🛠️ Key Projects

| Project | Description | Tech Stack |
| :--- | :--- | :--- |
| **Pneumonia Diagnostic Assistant** | Built a medical imaging classifier to detect pneumonia from X-ray scans using transfer learning. | PyTorch, TorchVision, ResNet |
| **FakeFinder** | Developed a binary classifier to distinguish between real and AI-generated images. | PyTorch, CNNs |
| **Visual Search Engine** | Implemented a **Siamese Network** to find similar images based on feature embeddings (metric learning). | PyTorch, Contrastive Loss |
| **Smart Fleet Optimization** | Optimized a model for edge deployment using **Quantization** and **Pruning** techniques to reduce size without losing accuracy. | ONNX, MLflow, Quantization |

---

## 🧰 Tech Stack & Tools used
* **Core Framework:** PyTorch
* **Computer Vision:** TorchVision, PIL, OpenCV
* **NLP:** Hugging Face Transformers
* **Ops & Tuning:** MLflow, Optuna, PyTorch Lightning
* **Deployment:** ONNX Runtime

## 🚀 Getting Started

To run the notebooks in this repository, I recommend setting up a virtual environment.

```bash
# Clone the repository
git clone https://github.com/meylismaliq/pytorch-for-deeplearning.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install matplotlib pandas numpy scikit-learn
pip install mlflow optuna onnx
