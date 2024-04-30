<h1>PhoBERT Text Classification Project</h1>

# I. Overview
- This project aims to classify whether a given student's article is relevant to the school or not using PhoBERT, a pre-trained language model for Vietnamese. The project consists of the following components:

**PhoBERT_Classification.ipynb**: This Jupyter Notebook contains Python code for training and evaluating the PhoBERT-based text classification model. You can use it to experiment, fine-tune, and evaluate the model's performance before deployment.
**server.py**: provide server API to receive data from users (e.g., student articles) and send it to the model for prediction. This server was built using any Flask
**images**: includes images about model evaluation and prediction
**Documentation and Usage Guide**: Provide documentation and a usage guide for end users, including how to use the API and endpoints to make new predictions.

# II. Getting Started
To get started with this project, follow these steps:

- Clone this repository to your local machine:
```cmd
git clone https://github.com/HhiepShadow/NCKH_2023.git
```
- Open and run the **PhoBERT_Classification.ipynb** notebook to train and evaluate the model.
- Prepare new data for next prediction
- Set up server API 
- Test the API endpoints by using Postman or Insomnia
- Provide documentation and usage instructions for end users.

# III. Requirements
- Python >= 3.10
- PyTorch=2.2.0
- Transformers
- Flask (or any other web framework for server API)

# IV. Usage
Train and evaluate the model using the provided notebook.
Set up and run the server API to receive and process new data.
Make predictions by sending data to the API endpoints.
Retrieve prediction results from the API.
Contributing
Contributions to this project are welcome. Feel free to open issues or pull requests for any improvements or bug fixes.

# V. License
This project is licensed under the MIT License.

