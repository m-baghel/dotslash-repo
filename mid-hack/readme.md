# Retail Shop Product Identification System

## Overview

This project presents an innovative solution for identifying and categorizing products in retail shop images. By leveraging advanced image segmentation and embedding techniques, the system can detect various products—such as oils, rice, wheat, and soap—and group similar items together. This capability is particularly beneficial for inventory management, automated checkout systems, and retail analytics.

## Features

- **Accurate Product Segmentation**: Utilizes Fast Segment Anything Model (Fast SAM) to precisely detect and segment products within images.
- **Semantic Embedding Generation**: Transforms segmented product images into embeddings that capture semantic similarities, ensuring that related products (e.g., different types of oil) are closely aligned in the embedding space.
- **Unsupervised Product Grouping**: Clusters products based on embedding similarities, facilitating the organization of items without the need for extensive labeled datasets.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/karuto12/retail-product-identification.git
   cd retail-product-identification
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up the Environment**:
   ```bash
   python setup.py
   ```

## Usage

1. **Segment Products in an Image**:
   ```bash
   python segment.py --input images/retail_shop.jpg --output masks/
   ```
   This command processes the input image to generate segmentation masks for each detected product.

2. **Generate Embeddings for Segmented Products**:
   ```bash
   python embeddings.py --masks masks/ --output embeddings/
   ```
   This step converts each segmented product into a semantic embedding, capturing its features and similarities to other products.

3. **Cluster Products Based on Embeddings**:
   ```bash
   python cluster.py --embeddings embeddings/
   ```
   This command groups similar products together, aiding in categorization and analysis.

## Project Structure

- `segment.py`: Script for performing product segmentation on input images.
- `embeddings.py`: Generates embeddings from segmented product images.
- `cluster.py`: Clusters products based on their embeddings.
- `requirements.txt`: Lists all necessary Python packages and dependencies.
- `setup.py`: Initializes the environment and prepares the system for use.
- `images/`: Directory containing sample input images.
- `masks/`: Directory where segmentation masks are saved.
- `embeddings/`: Directory for storing generated embeddings.

## Contribution

We welcome contributions to enhance this project. Please fork the repository and submit a pull request with your proposed changes. Ensure that your code adheres to the project's coding standards and is well-documented.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

---

This README provides a concise yet comprehensive overview of the project, guiding users from understanding its purpose to effectively utilizing its features. 
