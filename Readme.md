# Dependencies
This extension has dependencies on
- Chest Imaging Platform
- Total Segmentator

On Nasal Model Launch it will verify that all dependencies are available.
If not it will attempt to install each extension one followed by a restart
so your slicer program might restart twice

# Data processing 
The NasalModelLogic class handles all of the actual data processing  
The main logic loop happens in the Function called Process.

Function Process :

1. Create save folder

2. Initalize the work space / scene

3. Save the Spacing scale and IsotropicSpacing from user input

4. for each image node of each patient in the Dicom dir

    1. crop the image node such that all of the white space surrounding the patient is removed

        - during the cropping processes, it will also

            - apply IsotropicResampling (user input value)

            - apply SpacingScaling (user input value)

    2. create a lungmask from the image node

    3. generate airway segments using total segmenetator

        - specifically "head_glands_cavities" which includes : 

        ```
        eye_left,
        eye_right,
        eye_lens_left,
        eye_lens_right,
        optic_nerve_left,
        optic_nerve_right,
        parotid_gland_left,
        parotid_gland_right,
        submandibular_gland_right,
        submandibular_gland_left,
        nasopharynx,
        oropharynx,
        hypopharynx,
        nasal_cavity_right,
        nasal_cavity_left,
        auditory_canal_right,
        auditory_canal_left,
        soft_palate,
        hard_palate
        ```
    4. save the resulting segmentation inside a file

    5. clear the scene

5. done

Cropping image and saving the scene to file are done using slicer 3d's libraries  
Lungmask is created using the extension chest imaging platform  
Generating airway segmenets is created using the extension total segmentator  

## Lung mask rotation
Due to issues with the lung mask library, applying Lung mask on a volume node (image node) after applying space scaling with isometric sampling causes the image to comeout backwards. Therefore, we need to rotate the model by 180 on the Y axis.

this is the rotation matrix :
```
   [[-1,  0, 0, 0],
    [ 0, -1, 0, 0],
    [ 0,  0, 1, 0],
    [ 0,  0, 0, 1]]
```

this is same as scaling by `(-1, -1, 1, 1)`

then we need to shift the segment to the actual model, using the world coordinate (RAS) instead of the local model coordiate :
```
   [[-1,  0, 0, (segmentx - modelx)], 
    [ 0, -1, 0, (segmenty - modely)],   
    [ 0,  0, 1, (segmentz - modelz)],    
    [ 0,  0, 0,           1        ]]          
```            

which is the same as `shift (delta x, delta y, delta z, 1)`

in total we have :
```
Transformation =   [[-1,  0, 0, (segmentx - modelx)],
                    [ 0, -1, 0, (segmenty - modely)],
                    [ 0,  0, 1, (segmentz - modelz)],
                    [ 0,  0, 0,           1        ]]

Fixed_point = Transformation * segment_point
segmentModelNode.ApplyTransform(Transformation)
```
Using this transformation, the image and model will now match up as expected

# User interface 
The NasalModelWidget class handles all of the user facing logic

On extension load, slicer makes a call to `setup`. The setup function loads all of the predefined NasalModel design from a file called "NasalModel.ui" in the folder "UI". The "NasalModel.ui" file declears what and where the buttons go and how they should behave in a computer understandable way. It then sets, loads and connects all of the necessary buttons. When the user hits the apply button, we save all of the user input and hands it off to `NasalModelLogic` for processing.

All of the NasalModel is done using Qt which comes with slicer 3d.

# Global variables
All of the global variables that needs to be shared between the NasalModel and the actual script is stored inside `NasalModelParameterNode`. There must also be matching corresponding logic inside the `NasalModel.ui` file to indicate to the computer that a variable is shared between the NasalModel and the script.
