# alignedCelebFaces
Better version of my face editing tool. Explanation video here: https://www.youtube.com/watch?v=NTlXEJjfsQU

Are certain files missing? That's what I expected. There are two required files that are over 25 MB, which is over GitHub's limit.

To find them, please go here
https://drive.google.com/drive/folders/1wuup-fKhksYur9lOCQp2Iqqx8dMDcsc3?usp=sharing

Download the file "model27674.ckpt.data-00000-of-00001" and paste it into the "models" folder. (This describes the weights of the convolutional autoencoder)

Download the file "denseArray27K.npy" and paste it into the main folder (the folder with this readme). (This describes all 13,000 celebs' configurations of the 300 sliders.)

Also relevant are the 13,014 images I trained the models on, but I don't want to upload them all.
