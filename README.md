# deepfake-detector
University project - Creating a deepfake sorting model using ResNet18 architecture

The dataset contains 65483 images of which 31483 are real images and 34000 are fakes. The fakes contain 29000 faceswaps using facefusion and 5000 downloads from thispersondoesnotexist.com

The model is a retrained ResNet18 with dropouts. Test accuracy 99.68%.

The .ipynb file contains the model architecture. The .py file is a Web API UI using streamlit.

Requirements - Torch, Torchvision, Pillow, Streamlit.

Link to storage: [Data and Model](https://drive.google.com/drive/folders/1lodTcVemGSLfRHavzTNpzgQmLKuYpyk9?usp=sharing)
