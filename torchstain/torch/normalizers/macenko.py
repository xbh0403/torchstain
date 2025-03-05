import torch
from torchstain.base.normalizers.he_normalizer import HENormalizer
from torchstain.torch.utils import cov, percentile

"""
Source code ported from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class TorchMacenkoNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = torch.tensor([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
        self.maxCRef = torch.tensor([1.9705, 1.0308])

        # Avoid using deprecated torch.lstsq (since 1.9.0)
        self.updated_lstsq = hasattr(torch.linalg, 'lstsq')
        
    def __convert_rgb2od(self, I, Io, beta):
        I = I.permute(1, 2, 0)

        # Handle zeros to match TileNormalization.py
        I_reshaped = I.reshape((-1, I.shape[-1])).float()
        I_reshaped = torch.where(I_reshaped == 0, torch.tensor(1.0, device=I.device), I_reshaped)
        
        # calculate optical density
        OD = -torch.log(I_reshaped / Io)

        # remove transparent pixels (same as TileNormalization)
        ODhat = OD[(OD >= beta).all(dim=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues (eigenvectors 1 and 2, matching TileNormalization.py)
        That = torch.matmul(ODhat, eigvecs[:, 1:3])
        phi = torch.atan2(That[:, 1], That[:, 0])

        # Use torch.quantile with alpha/100 to match TileNormalization.py
        minPhi = torch.quantile(phi, alpha / 100.0)
        maxPhi = torch.quantile(phi, 1 - alpha / 100.0)

        vMin = torch.matmul(eigvecs[:, 1:3], torch.tensor([[torch.cos(minPhi)], [torch.sin(minPhi)]], device=eigvecs.device))
        vMax = torch.matmul(eigvecs[:, 1:3], torch.tensor([[torch.cos(maxPhi)], [torch.sin(maxPhi)]], device=eigvecs.device))

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second (same as TileNormalization)
        if vMin[0] > vMax[0]:
            HE = torch.stack([vMin[:, 0], vMax[:, 0]], dim=1)
        else:
            HE = torch.stack([vMax[:, 0], vMin[:, 0]], dim=1)

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = OD.T

        # determine concentrations of the individual stains
        if not self.updated_lstsq:
            return torch.lstsq(Y, HE)[0][:2]
    
        return torch.linalg.lstsq(HE, Y)[0]

    def __compute_matrices(self, I, Io, alpha, beta):
        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors (match TileNormalization.py)
        cov_matrix = torch.cov(ODhat.T)
        eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)
        # Match TileNormalization.py using quantile
        maxC = torch.tensor([
            torch.quantile(C[0, :], 0.99), 
            torch.quantile(C[1, :], 0.99)
        ], device=C.device)

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        ''' Normalize staining appearence of H&E stained images

        Example use:
            see example.py

        Input:
            I: RGB input image: tensor of shape [C, H, W] and type uint8
            Io: (optional) transmitted light intensity
            alpha: percentile
            beta: transparency threshold
            stains: if true, return also H & E components

        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        # normalize stain concentrations
        C *= (self.maxCRef / maxC).unsqueeze(-1)

        # recreate the image using reference mixing matrix
        Inorm = Io * torch.exp(-torch.matmul(self.HERef, C))
        Inorm[Inorm > 255] = 255
        Inorm = Inorm.T.reshape(h, w, c).int()

        H, E = None, None

        if stains:
            H = torch.mul(Io, torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0))))
            H[H > 255] = 255
            H = H.T.reshape(h, w, c).int()

            E = torch.mul(Io, torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0))))
            E[E > 255] = 255
            E = E.T.reshape(h, w, c).int()

        return Inorm, H, E
