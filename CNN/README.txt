CNN CAT/DOG ML LAB
==================
This version trains a real binary CNN on CIFAR-10 cat (class 3) and dog (class 5)
images. It does not use the old ImageNet label-word heuristic.

Setup from ML:
  .venv\Scripts\python.exe -m pip install -r CNN\requirements.txt
Then double-click "Start CNN.vbs".

Workflow: Prepare Dataset -> choose epochs -> Train CNN -> inspect history and
untouched-test metrics -> choose examiner image -> save report.
The first dataset preparation downloads CIFAR-10. Training saves cat_dog_cnn.keras.
Read LAB_GUIDE_BN.txt or the Guide tab for the full Bengali experiment and viva.

Inputs are validated: 100..5000 training images per class, 1..20 epochs, supported
image formats, fully readable RGB image, and minimum 8x8 size.
CNN_Cat_Dog_GUI_original.py is the previous pretrained inference demo backup.
