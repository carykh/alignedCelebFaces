# alignedCelebFaces
Better version of my face editing tool. Explanation video here: https://www.youtube.com/watch?v=NTlXEJjfsQU

## How to install:
```bash
pip3 install -r requirements.txt
```

_optional, but recommended: use venv_

## How to run:
```bash
python3 code/final_face_slider_tool.py

```

# Controls:
Click on buttons and drag on sliders.
Use the scroll wheel with your mouse over a PCA box for more fine-tuned control of the component.
Use the scroll wheel with your mouse away from the the PCA boxes to navigate around the full 300-PCA list.


## How to add images
Add your `.jpg` or `.png` images to the `/extra_images` folder and then run
```bash
./add_extra_images.sh
```

_Note: all images in `/extra_images` will be run, even if they already exist in the database. If you wish to avoid duplication, delete all images in this folder before running this script again._


# Notes

There are the 13,014 images I trained the models on, but I don't want to upload them all. You can find that at https://www.famousbirthdays.com/.



