import tigre
from tigre.utilities import CTnoise
import numpy as np
import matplotlib.pyplot as plt



def visualize_projections(path, projections, angles, figs_per_row=10):
    n_row = np.ceil(len(projections) / figs_per_row).astype(int)
    projections = projections.copy()
    projections = (projections - projections.min()) / (projections.max() - projections.min())

    for i in range(len(projections)):
        angle = int((angles[i] / np.pi) * 180)
        plt.subplot(n_row, figs_per_row, i + 1)
        plt.imshow(projections[i] * 255, cmap='gray', vmin=0, vmax=255)
        plt.title(f'{angle}')
        plt.axis('off')

    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=500)
    plt.close()


class ConeGeometry_special(tigre.utilities.geometry.Geometry):
    def __init__(self, config):
        super().__init__()

        self.DSD = config['DSD'] / 1000
        self.DSO = config['DSO'] / 1000
        
        # detector parameters
        self.nDetector = np.array(config['nDetector'])        # number of pixels           (px)
        self.dDetector = np.array(config['dDetector']) / 1000 # size of each pixel         (m)
        self.sDetector = self.nDetector * self.dDetector      # total size of the detector (m)
        
        # image parameters: in the form of [z, y, x]
        self.nVoxel = np.array(config['nVoxel'][::-1])        # number of voxels        (vx)
        self.dVoxel = np.array(config['dVoxel'][::-1]) / 1000 # size of each voxel      (m)
        self.sVoxel = self.nVoxel * self.dVoxel               # total size of the image (m)

        # offsets
        self.offOrigin = np.array(config['offOrigin'][::-1]) / 1000 # offset of image from origin (m)
        self.offDetector = np.array(
            [config['offDetector'][1], config['offDetector'][0], 0]) / 1000 # offset of detector (m)

        # auxiliary
        self.accuracy = config['accuracy'] # accuracy of FWD proj (vx/sample)  # noqa: E501
        
        # mode
        self.mode = config['mode'] # options: parallel/cone 
        self.filter = config['filter']


class Projector:
    def __init__(self, config, angles):
        self._geo = ConeGeometry_special(config)
        self._angles = angles / 180 * np.pi

    def __call__(self, image):
        projections = tigre.Ax(
            image.transpose(2, 1, 0).copy(), # [z, y, x]
            self._geo, 
            self._angles
        )[:, ::-1, :]
        projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))
        projections[projections < 0.0] = 0.0
        return {
            'projs': projections, 
            'angles': self._angles
        }
