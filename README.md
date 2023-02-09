# DeepLabCut_Jupyter_Colab
This tutorial will show you how to create a deeplabcut pose estimation of animals using jupyter notebook and google colab using an example project of my cat.

https://user-images.githubusercontent.com/72940641/217903980-958686d4-397e-401d-9622-9638aa9893e3.mp4

Deeplabcut (DLC) requires a strong GPU with at least 8GB of memory for the actual training portion. If you do not have this, DO NOT ATTEMPT TO RUN TRAINING CODE ON YOUR LOCAL COMPUTER --> it will run very slow and your computer will be unhappy. 

Instead, you can create the project and extract the data frames using Jupyter, then train and analyze videos using google collab 

## Using the GUI 

If at any point you want to perfom actions using the DLC GUI you can open it in the command terminal, using the following commands: 

```
conda activate DEEPLABCUT
ipython
Import deeplabcut
deeplabcut.launch_dlc()
```

## Create Project on Jupyter

After downloading and importing DLC into Jupyter, you need to create the project files with the config file:
```
deeplabcut.create_new_project('name', 'creator', [path\to\video.mp4])
```

## Extract frames from video on Jupyter

```
deeplabcut.extract_frames(path_config_file, 'automatic', 'kmeans', crop = True, userfeedback = False)
```

## Labeling the video using the GUI

After creating the project folders, open the DLC GUI to edit your config file to display which body parts you want to track. Here we will be tracking the left ear, right ear, nose, and left paw. Below is the glossary of parameters for the config file. also see example config file for this project.


![image](https://user-images.githubusercontent.com/72940641/217890158-df4e4f82-cfcc-4b7b-8c9c-c57b3e5a743f.png)

- Open the DLC GUI and click load project
![Screenshot (10)](https://user-images.githubusercontent.com/72940641/217906511-f85f55af-348f-4127-9d1c-c7a3ba168da7.png)

- Navigate to your project config file and open
- Click label frames 
![Screenshot (8)](https://user-images.githubusercontent.com/72940641/217906545-448569cd-be39-494f-8027-6d31d919ad96.png)

- Open folder with extracted frames 
- Use the select points tool (arrow figure) and the keypoint selection (bottom right) to click and label body parts.
![Screenshot (9)](https://user-images.githubusercontent.com/72940641/217906567-f3dc539e-9dda-47da-959f-6bdd3502bd8d.png)

- Do this for each frame and save screenshots in new, labeled frames folder


## Training DLC using Google Colab

Now that you have made the project and labeled the video frames you can train your dataset to reconginze and predict where these points will be on your video in Google colab. (If you have a nice workstation with a strong GPU continue this step in Jupyter Notebook)

Google colab relies on google's virtual machine, which means you are utilizing google's cloud system and borrowing their resources to perform tasks. However, this means that each time you start up your colab notebook, you must reinstall the deeplabcut software. 

First, transfer your files from your local machine to your google drive

Then, mount your google drive to google colab:
```
from google.colab import drive
drive.mount('/content/gdrive')
```

Instal dlc using pip:
```
!pip install deeplabcut
```

import:
```
import deeplabcut
```

Set the path to the config file in your drive:
```
path_config = '/content/drive/MyDrive/Cat_video-Ian-2023-02-02/config.yaml'
```

Create the training dataset:
```
deeplabcut.create_training_dataset(path_config, augmenter_type='imgaug')
```

Begin training:
```
deeplabcut.train_network(path_config)
```

The dlc network training should take between 1-3 hours. You want it to run for about 50,000 iterations so that the loss is very low (around 0.0015-ish)

Evaluate the training network:
```
deeplabcut.evaluate_network(path_config,Shuffles=[1], plotting=True)
```

Analyze and plot:
```
deeplabcut.analyze_videos(path_config, ['/content/drive/MyDrive/Cat_video-Ian-2023-02-02/videos/Cat_Video.mp4'], save_as_csv=True)

deeplabcut.filterpredictions(path_config,['/content/drive/MyDrive/Cat_video-Ian-2023-02-02/videos'], videotype='.mp4',filtertype= 'arima',ARdegree=5,MAdegree=2)

deeplabcut.plot_trajectories(path_config, ['/content/drive/MyDrive/Cat_video-Ian-2023-02-02/videos/Cat_Video.mp4'])
```

If you want to add an extra skeleton layer navigate to your config file and under, #plotting configuration, add something that looks like the following 
that matches each point you want to connect:

```
skeleton: [[nose, leftear], [nose, rightear], [leftear, rightear]]
skeleton_color: white
pcutoff: 0.4
dotsize: 4
alphavalue: 0.5
colormap: jet
```

Create labeled video:
```
deeplabcut.create_labeled_video(path_config, ['/content/drive/MyDrive/Cat_video-Ian-2023-02-02/videos'], save_frames = False, draw_skeleton = True)
```

Create csv files that analyze the angles and pixel lengths between each skeleton point:
```
deeplabcut.analyzeskeleton(path_config, ['/content/drive/MyDrive/Cat_video-Ian-2023-02-02/videos'], videotype='.mp4', shuffle=1, trainingsetindex=0, save_as_csv=True, destfolder=None)
```
CSV file shown in Cat_VideoDLC_resnet50_Cat_videoFeb2shuffle1_50000_skeleton.csv

## Now you are done!
Be sure to view the main DLC github page for any unanswered questions!
https://github.com/DeepLabCut/DeepLabCut/blob/main/docs/standardDeepLabCut_UserGuide.md

Remember, because this was done in google colab, you may need to transfer files from your google drive to your local desktop if you want to work with them further in other software

![image](https://user-images.githubusercontent.com/72940641/217901051-7969a7ec-e492-4b4b-ae7a-4ecf7549e63b.png)











