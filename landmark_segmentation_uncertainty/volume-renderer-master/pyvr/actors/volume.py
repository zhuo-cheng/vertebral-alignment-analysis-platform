from __future__ import absolute_import

import vtk
import numpy as np

from ..data.volume import load_volume
from ..data.volume import centering
from ..data.volume import numpy_to_volume
from ..actor import Actor
from ..utils.vtk.transforms import centered_transform
from ..preset import VolumePreset


class VolumeActor(Actor):
    """ Actor for volume raycasting """
    def __init__(self, volume, spacing, preset='bone', centered=True, gpu=True):

        super().__init__()

        if isinstance(volume, str):
            volume = load_volume(volume)

        if isinstance(volume, np.ndarray):
            origin = [0,0,0]
            volume = numpy_to_volume(volume, spacing, origin, order='zyx')

        if centered:
            volume = centering(volume)

        preset = VolumePreset(preset)

        self._volume = volume
        self._preset = preset
        self._gpu = gpu

        self.update_mapper()
        self.update_property()


    def update_mapper(self):

        if self._gpu:
            mapper = vtk.vtkGPUVolumeRayCastMapper()
        else:
            mapper = vtk.vtkFixedPointVolumeRayCastMapper()

        mapper.SetInputData(self._volume)
        mapper.SetBlendModeToComposite()

        self._mapper = mapper


    def update_property(self):

        preset = self._preset

        color = vtk.vtkColorTransferFunction()
        for (v, r, g, b, mid, sharp) in preset.color_transfer:
            color.AddRGBPoint(v, r, g, b, mid, sharp)

        scalar_opacity = vtk.vtkPiecewiseFunction()
        for (v, a, mid, sharp) in preset.scalar_opacity:
            scalar_opacity.AddPoint(v, a, mid, sharp)

        gradient_opacity = None
        if preset.gradient_opacity is not None:
            gradient_opacity = vtk.vtkPiecewiseFunction()
            for (v, a, mid, sharp) in preset.gradient_opacity:
                gradient_opacity.AddPoint(v, a, mid, sharp)

        prop = vtk.vtkVolumeProperty()
        prop.SetIndependentComponents(True)
        prop.SetColor(color)
        prop.SetScalarOpacity(scalar_opacity)
        if gradient_opacity is not None:
            prop.SetGradientOpacity(gradient_opacity)

        if preset.interpolation:
            prop.SetInterpolationTypeToLinear()
        else:
            prop.SetInterpolationTypeToNearest()

        if preset.shade:
            prop.ShadeOn()
        else:
            prop.ShadeOff()

        prop.SetAmbient(preset.ambient)
        prop.SetDiffuse(preset.diffuse)
        prop.SetSpecular(preset.specular)
        prop.SetSpecularPower(preset.specular_power)

        unit_distance = min(self._volume.GetSpacing())
        prop.SetScalarOpacityUnitDistance(unit_distance)

        self._property = prop


    def build(self):

        actor = vtk.vtkVolume()

        actor.SetMapper(self._mapper)
        actor.SetProperty(self._property)

        if self._transform is not None:
            actor.SetUserTransform(self._transform)

        actor.Update()

        return actor




class VertebraVolumeActor(Actor):

    """ Actor for vertebrae volume raycasting """
    def __init__(self, volume, spacing, preset='bone', centered=True, gpu=True):

        super().__init__()

        if isinstance(volume, str):
            volume = load_volume(volume)

        if isinstance(volume, np.ndarray):
            origin = [0,0,0]
            volume = numpy_to_volume(volume, spacing, origin, order='zyx')

        if centered:
            volume = centering(volume)

        preset = VolumePreset(preset)

        self._volume = volume
        self._preset = preset
        self._gpu = gpu

        self.update_mapper()
        self.update_property()


    def update_mapper(self):

        if self._gpu:
            mapper = vtk.vtkGPUVolumeRayCastMapper()
        else:
            mapper = vtk.vtkFixedPointVolumeRayCastMapper()

        mapper.SetInputData(self._volume)
        mapper.SetBlendModeToComposite()

        self._mapper = mapper


    def update_property(self):

        # The gradient opacity function is used to decrease the opacity
        # in the "flat" regions of the volume while maintaining the opacity
        # at the boundaries between tissue types.  The gradient is measured
        # as the amount by which the intensity changes over unit distance.
        # For most medical data, the unit distance is 1mm.
        gradient_opacity = vtk.vtkPiecewiseFunction()

        # The color transfer function maps voxel intensities to colors.
        color = vtk.vtkColorTransferFunction()
        scalar_opacity = vtk.vtkPiecewiseFunction()
        prop = vtk.vtkVolumeProperty()

        # The opacity transfer function is used to control the opacity
        # of different tissue types.
        opacityValue = 0.05

        # Surface
        color.AddRGBPoint(240, 200.0 / 255.0, 200.0 / 255.0, 200.0 / 255.0, 0, 0.0)
        # scalar_opacity.AddPoint(1024, 1, 0.5, 0.0)
        scalar_opacity.AddPoint(4024, 1, 0.5, 0.0)
        # # scalar_opacity.AddPoint(1024, 1)

        # Bone
        scalar_opacity.AddPoint(80, 0, 0.5, 0.0)
        gradient_opacity.AddPoint(480, 0.2)
        prop.SetInterpolationTypeToLinear()
        prop.SetAmbient(0.12)
        prop.SetSpecular(0.0)


        # The VolumeProperty attaches the color and opacity functions to the
        # volume, and sets other volume properties.  The interpolation should
        # be set to linear to do a high-quality rendering.  The ShadeOn option
        # turns on directional lighting, which will usually enhance the
        # appearance of the volume and make it look more "3D".  However,
        # the quality of the shading depends on how accurately the gradient
        # of the volume can be calculated, and for noisy data the gradient
        # estimation will be very poor.  The impact of the shading can be
        # decreased by increasing the Ambient coefficient while decreasing
        # the Diffuse and Specular coefficient.  To increase the impact
        # of shading, decrease the Ambient and increase the Diffuse and Specular.

        prop.SetColor(color)
        prop.SetScalarOpacity(scalar_opacity)
        prop.SetGradientOpacity(gradient_opacity)
        prop.ShadeOn()
        # prop.SetAmbient(0.12)
        prop.SetDiffuse(1.0)
        prop.SetSpecularPower(10.0)
        prop.SetScalarOpacityUnitDistance(1)
        self._property = prop


    def build(self):

        actor = vtk.vtkVolume()

        actor.SetMapper(self._mapper)
        actor.SetProperty(self._property)

        if self._transform is not None:
            actor.SetUserTransform(self._transform)

        actor.Update()

        return actor
