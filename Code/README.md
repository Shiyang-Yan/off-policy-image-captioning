# O-self-critial
Requirements

Python 3.7 (need COCO evaluation package for Python 3)

PyTorch 1.2.1 (with torchvision)

cider

coco-caption

spacy (to tokenize words)

h5py (to store features)

scikit-image (to process images)

Procedure:

1. Download the data from the following link, put it in the directory.
https://drive.google.com/drive/folders/1oFImRV4pyZGjbZC76fT7pmbhLLiNvK22?usp=sharing

2. Download the pre-trained Meshed transformer model on Visual Paragraph Generation from the following link, and put it in the "log_meshed" directory:
https://drive.google.com/file/d/1e8LdR4l-8mehxeL1B6sEFrMQUOtIFtMU/view?usp=sharing

3. Download the pre-trained image-guided language auto-encoder from the following link, and put it in the "language_model" directory.
https://drive.google.com/file/d/1RunHfKv3SAWqhFEM3uep5YnAF7EWmuLz/view?usp=sharing

4. To run the off-policy self-critical sequence learning on Standford Visual Paragraph Dataset, run:

   "python train_off_is_div.py"

5. To evaluate, run:
   
   "python eval_meshed.py"
   


