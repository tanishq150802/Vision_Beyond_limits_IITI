# Vision_Beyond_limits_IITI
Top 3 submission at IITB Techfest'21 by Mihir Karandikar, Tanishq Selot, Atharva Mohite and Yeeshukant singh (Students at IIT Indore)

Dataset Preparation:

1) The script Generate Masks(1024x1024).py will generate the masks from the given annotations. The
   masks generated have dimensions of 1024x1024

2) Due to memory issues in Collab, the mask size has been reduced to 512x512 by appropriately 
   reducing the image size and at the same time not affecting the proportions adversly.


Training :

1) The python notebook VBL_UNET_Training contains all the code for training of the model and also
   evaluation on the dataseet

Generation of Segmented Images

1) The script 'VBL_UNET_Predictions.py' uses the saved model to generate the segmented images 
   which are then saved in the folder VBL_Segmented_Images
