import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from pathlib import Path
#
# gui
#

class gui(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("gui")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Nasal")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Tianze Kuang"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#gui">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Tianze Kuang
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # gui1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="gui",
        sampleName="gui1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "gui1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="gui1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="gui1",
    )

    # gui2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="gui",
        sampleName="gui2",
        thumbnailFileName=os.path.join(iconsPath, "gui2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="gui2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="gui2",
    )


#
# guiParameterNode
#
@parameterNodeWrapper
class guiParameterNode:
    """
    The parameters needed by module.
    DicomDir : directory of dicom files
    OutputDir : directory to store results to
    """
    DicomDir : Path
    OutputDir : Path
    
#
# guiWidget
#
class guiWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/gui.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = guiLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            
    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[guiParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.DicomDir.exists() and self._parameterNode.OutputDir.exists():
            self.ui.applyButton.toolTip = _("Compute")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output directory")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        print(f"Dicom Dir {self._parameterNode.DicomDir}")
        print(f"Output Dir {self._parameterNode.OutputDir}")
        print(f"Spacing scale {self.ui.SpacingScalingSpinBox.value}")
        print(f"Isotropic spacing {self.ui.IsotropicCheckbox.isChecked()}")
        
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            
            self.logic.Process(self._parameterNode.DicomDir, self._parameterNode.OutputDir, self.ui.SpacingScalingSpinBox.value, self.ui.IsotropicCheckbox.isChecked())


#
# guiLogic
#

class guiLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return guiParameterNode(super().getParameterNode())
    
    def OpenDicom(self, DicomDir):
        from DICOMLib import DICOMUtils
        
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(DicomDir, db)

            return db.patients()
        
    def CropVolume(self, PatientNode):
        # new crop volume config
        CropVolumeNode = slicer.vtkMRMLCropVolumeParametersNode() 
        OutputNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        OutputNode.SetName(f"{PatientNode.GetName()}_Cropped")
        
        # add to scene
        slicer.mrmlScene.AddNode(CropVolumeNode)
        CropVolumeNode.SetSpacingScalingConst(self.SpacingScale)
        CropVolumeNode.SetIsotropicResampling(self.IsotropicSpacing)

        # Set the volume as the input volume in the crop volume module
        CropVolumeNode.SetInputVolumeNodeID(PatientNode.GetID())
        # Set output volume as the same volume to overwrite original volume (only needed if you actually want to crop the volume)
        CropVolumeNode.SetOutputVolumeNodeID(OutputNode.GetID())

        # add this config to scene
        # slicer.mrmlScene.AddNode(CropVolumeNode)

        CropVolumeLogic = slicer.modules.cropvolume.logic()
        # Set the input ROI
        ROI = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        slicer.mrmlScene.AddNode(ROI)
        CropVolumeNode.SetROINodeID(ROI.GetID())
        # slicer.mrmlScene.AddNode(ROI)  # Ensure ROI is added to the scene
        # Use the Fit ROI to Volume function of the crop volume module
        CropVolumeLogic.FitROIToInputVolume(CropVolumeNode)
        # this is optional
        CropVolumeLogic.SnapROIToVoxelGrid(CropVolumeNode)

        # run
        CropVolumeLogic.Apply(CropVolumeNode)

        return CropVolumeNode.GetOutputVolumeNode(), ROI

    def LungMask(self, InputNode):
        # chest imaging platform -> toolkit -> setgmentation -> generate simple lung mask using tissue + airway
        LungMask = slicer.modules.generatesimplelungmask

        OutputNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        OutputNode.SetName(f"{InputNode.GetName()}_Lungmask")

        Config = {
            # do i have to put them into a file first???
            'inputVolume': InputNode.GetID(),
            'outputVolume': OutputNode.GetID(),
            'lowDose' : False
        }

        CliNode = slicer.cli.runSync(module = LungMask, parameters = Config)
        if CliNode.GetStatus() & CliNode.ErrorsMask:
            raise ValueError("Failed to generate lung mask")

        if not self.IsotropicSpacing:
            return OutputNode

        # apply transformation fix
        '''
        from current observation,
        applying space scaling with isometric sampling causes the image to comeout backwards,
        we need to rotate the model by 180 on the Y axis.
        
        this is the rotation matrix :
        [[-1,  0, 0, 0],
         [ 0, -1, 0, 0],
         [ 0,  0, 1, 0],
         [ 0,  0, 0, 1]]

        this is same as scaling by (-1, -1, 1, 1)

        then we need to shift the segment to the actual model, remember to use the world coordinate (RAS) :
        [[-1,  0, 0, (segmentx - modelx)], 
         [ 0, -1, 0, (segmenty - modely)],   
         [ 0,  0, 1, (segmentz - modelz)],    
         [ 0,  0, 0,           1        ]]                      

        shift (delta x, delta y, delta z, 1)

        in total we have :
        
        Transformation = [[-1,  0, 0, (segmentx - modelx)],
                          [ 0, -1, 0, (segmenty - modely)],
                          [ 0,  0, 1, (segmentz - modelz)],
                          [ 0,  0, 0,           1        ]]

        Fixed_point = Transformation * segment_point
        
        segmentModelNode.ApplyTransform(Transformation)

        issue is the rotation changes the coordinate positions in world, so must do it seperately
        '''
        
        RotateTransform = vtk.vtkTransform()
        RotateTransform.Scale(-1, -1, 1)
        AdjustmentTransform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
        AdjustmentTransform.SetMatrixTransformToParent(RotateTransform.GetMatrix())
        OutputNode.SetAndObserveTransformNodeID(AdjustmentTransform.GetID())
        
        # xmin, xmax, ymin, ymax, zmin, zmaxs
        OutputBounds = [0] * 6
        OutputNode.GetRASBounds(OutputBounds)
        
        # find the max of the original model
        InputBounds = [0] * 6
        InputNode.GetRASBounds(InputBounds)
        
        # add to the shift
        RotateTransform.Translate((OutputBounds[1] - InputBounds[1]), (OutputBounds[3] - InputBounds[3]), (OutputBounds[5] - InputBounds[5]))
        AdjustmentTransform.SetMatrixTransformToParent(RotateTransform.GetMatrix())

        return OutputNode

    def GenerateSegment(self, InputNode):
        TotalSegmentator = slicer.util.getModuleLogic('TotalSegmentator')
        OutputNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        OutputNode.SetName(f"{InputNode.GetName()}_TotalSegmentatorNode")
        """
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param fast: faster and less accurate output
        :param task: one of self.tasks, default is "total"
        :param subset: a list of structures (TotalSegmentator classe names https://github.com/wasserth/TotalSegmentator#class-detailsTotalSegmentator) to segment.
            Default is None, which means that all available structures will be segmented."
            `--roi_subset`: Takes a space-separated list of class names (e.g. `spleen colon brain`) and only predicts those classes. Saves a lot of runtime and memory. Might be less accurate especially for small classes (e.g. prostate).
            wasserth/TotalSegmentator/totalsegmentator/resources/totalsegmentator_snomed_mapping.csv
        :param interactive: set to True to enable warning popups to be shown to users
        :param sequenceBrowserNode: if specified then all frames of the inputVolume sequence will be segmented
        """
        TotalSegmentator.process(InputNode, OutputNode, fast = False, cpu = False, task = "head_glands_cavities", subset = None, interactive = False)
        return OutputNode

    def Process(self, DicomDir, SavePath, SpacingScale = 0.5, IsotropicSpacing = True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param DicomDir: directory of dicom files
        :param OutputDir: directory to store results to
        :param SpacingScale: spacing scale used by crop volume
        :param IsotropicSpacing: Isotropic spacing enabled for crop volume
        """
        if not DicomDir.exists() or not SavePath.exists():
            raise ValueError("DicomDir and or OutputDir is invalid")
            
        # create a subfolder for results
        from datetime import datetime
        SaveDir = SavePath / datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250615_153045"
        SaveDir.mkdir()

        self.SpacingScale = SpacingScale
        self.IsotropicSpacing = IsotropicSpacing
        
        slicer.app.processEvents()
        slicer.mrmlScene.Clear(0)
        slicer.app.processEvents()
        
        import gc

        if True:
        #for i, P in enumerate(OpenDicom(DicomDataDir)):
            # clear the scene

            # CurrentNode = DICOMUtils.loadPatientByUID(P)
            import SampleData
            CurrentNode = SampleData.SampleDataLogic().downloadSample('CTChest')
            
            # Crop volume
            Cropped, ROI = self.CropVolume(CurrentNode)
            # lungmask
            Lung = self.LungMask(Cropped)
            # Total segmentator -> head cavities and glands
            Segment = self.GenerateSegment(Cropped)
            # consider saveNode instead
            '''
            https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#save-the-scene-into-a-single-mrb-file
            '''
            save_path = str(SaveDir) + f"/{CurrentNode.GetName()}_{i}_processed.mrb"
            if not slicer.util.saveScene(save_path):
                raise IOError("Failed to save")

            slicer.mrmlScene.RemoveNode(ROI)
            
            slicer.app.processEvents()
            slicer.mrmlScene.Clear(0)
            slicer.app.processEvents()
            gc.collect()

#
# guiTest
#

class guiTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_gui1()

    def test_gui1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("gui1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = guiLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
