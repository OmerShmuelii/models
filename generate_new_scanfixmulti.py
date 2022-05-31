
import torch
import numpy as np
import cv2
import albumentations
import os
import sys
import scipy
import scipy.io
import matplotlib.pyplot as plt
import math
import random
import nibabel as nib  # might need to be downloaded
import time  # imports from here on are used for tests
import warnings
import pickle
warnings.filterwarnings("ignore", category=UserWarning)


class NormDict:
    """
    Dictionary of normalization parameters, mean and STD. per brain
    """
    def __init__(
            self,
            dat_flr,
            samples_list,
            sbj_nme='',
            fin_sze=(),
            pad_flg=False,
            crp_flg=True,
            grd_flg=True,
            ms__flg=False,
            tst_flg=False,
            ref_flg=False,
            erd_thc=0,
            qt2_flg=False,
            qt1_flg=False,
            t2w_flg=False,
            t2flair_flg=False,
            pd_flg=False,
            cuda_flg=False,
            t2w_t_echo=90,
            flair_t_echo=80,
            flair_t_inv=2372,
            flair_fa=150,
            flair_etl=16,
            pe_ax=1,
            sever_per_arr=(0, 6, 9, 12, 15, 18, 21, 25, 30),
            les_num=1,
            les_prob=0.5,
            les_size=10,
            lod_flg=True,
            sve_flg=False,
            dic_nme='normalization_dictionary'
            ):
        """

        :param dat_flr:
        :param samples_list:
        :param sbj_nme:
        :param fin_sze:
        :param pad_flg:
        :param crp_flg:
        :param grd_flg:
        :param ms__flg:
        :param tst_flg:
        :param ref_flg:
        :param erd_thc:
        :param qt2_flg:
        :param qt1_flg:
        :param t2w_flg:
        :param t2flair_flg:
        :param pd_flg:
        :param cuda_flg:
        :param t2w_t_echo:
        :param flair_t_echo:
        :param flair_t_inv:
        :param flair_fa:
        :param flair_etl:
        :param pe_ax:
        :param sever_per_arr:
        :param les_num:
        :param les_prob:
        :param les_size:
        :param lod_flg:         [bool]  whether to load dictionary
        :param sve_flg:         [bool]  whether to save dictionary
        :param dic_nme:         [str]   name for dictionary file to be saved
        """
        cmp_flg = not lod_flg  # flag whether to compute new dictionary (turns true if there is no file to load)
        # Check if dictionary is already saved
        if lod_flg:
            dic_fle_lst = [fle_nme for fle_nme in os.listdir(dat_flr) if dic_nme in fle_nme]
            if len(dic_fle_lst) == 0:
                cmp_flg = True
            else:
                if len(dic_fle_lst) > 1:
                    warnings.warn('More then 1 dictionary detected, using last one')
                dic_fle = dic_fle_lst[-1]
                with open(os.path.join(dat_flr, dic_fle), 'rb') as f:
                    self.dict = pickle.load(f)
        # Compute dictionary
        if cmp_flg:
            if tst_flg and not ref_flg:
                samples_list_1 = get_qmri_data_list(dat_flr, ms__flg, tst_flg, ref_flg=True, erd_thc=erd_thc)
            else:
                samples_list_1 = samples_list
            self.dict = get_qmri_norm_dict(
                dat_flr,
                samples_list_1,
                sbj_nme=sbj_nme,
                fin_sze=fin_sze,
                pad_flg=pad_flg,
                crp_flg=crp_flg,
                grd_flg=grd_flg,
                ms__flg=ms__flg,
                tst_flg=tst_flg,
                ref_flg=True,
                qt2_flg=qt2_flg,
                qt1_flg=qt1_flg,
                t2w_flg=t2w_flg,
                t2flair_flg=t2flair_flg,
                pd_flg=pd_flg,
                cuda_flg=cuda_flg,
                t2w_t_echo=t2w_t_echo,
                flair_t_echo=flair_t_echo,
                flair_t_inv=flair_t_inv,
                flair_fa=flair_fa,
                flair_etl=flair_etl,
                pe_ax=pe_ax,
                sever_per_arr=sever_per_arr,
                les_num=les_num,
                les_prob=les_prob,
                les_size=les_size,
                )
        if sve_flg and cmp_flg:
            if 'dic_fle_lst' not in locals():
                dic_fle_lst = [fle_nme for fle_nme in os.listdir(dat_flr) if dic_nme in fle_nme]
            dic_nme_new = dic_nme + '_' + str(len(dic_fle_lst))
            with open(os.path.join(dat_flr, dic_nme_new) + '.pkl', 'wb') as f:
                pickle.dump(self.dict, f, pickle.HIGHEST_PROTOCOL)

    def update_dict(self, new_dict):
        self.dict = new_dict

    def print_dict(self):
        print(self.dict)


def mri_t2w(pd_m, t2_map, t_echo):
    """
    Returns T2-weighted MR image assuming mono-exponential model

    :param pd_m: proton-density map
    :param t2_map: T2 map
    :param t_echo: echo time
    :return: T2 weighted MR images

    Created by Chen Solomon, 28/08/2020
    """
    if t_echo == 0:
        return pd_m
    try:
        t2weighted = pd_m * torch.exp(-torch.div(t_echo, t2_map))
    except TypeError:
        t2weighted = pd_m * np.exp(-np.divide(t_echo, t2_map, out=np.inf * np.ones(t2_map.shape), where=t2_map != 0))
    return t2weighted


# noinspection SpellCheckingInspection
class MRIEstimateT1:
    def __init__(self):
        kernel_size = 15
        # sigma = 3
        rsz_fct = 8
        flt_fct = 0.25
        flt_ker = flt_fct * rsz_fct
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = flt_ker ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = \
            torch.exp(
                -torch.sum(
                    (xy_grid.float() - mean) ** 2.,
                    dim=-1)
                / (2 * variance)
            )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)
        padd = (kernel_size - 1) // 2
        self.gaussian_filter = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            groups=1,
            padding=padd,
            bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel.cuda()
        self.gaussian_filter.weight.requires_grad = False

    def imp(self, t2_map, cuda_flg=False):
        """
        Linearly estimates T1 values from T2values in a brain scan

        :param t2_map: T2 map (according to which T1-map is estimated)
        :param cuda_flg: flag about whether to use GPU
        :return: T1 map

        Created by Chen Solomon 31/08/2020
        Edited by Omer Shmueli 03/10/2020
        """
        if torch.cuda.is_available() and cuda_flg:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
        p1 = 5.951  # Predetermined estimation parameters, computed given data from https://doi.org/10.1002/mrm.27927
        p2 = 0.6058

        # edited parameters

        # csf_th_t2 = 90  # [ms] # Predetermined estimation parameters, manually optimized  # original
        csf_th_t2 = 100  # edited

        csf_t2_val = 470  # [ms]
        rsz_fct = 8

        # flt_fct = 0.25  # original
        flt_fct = 1  # edited

        flt_ker = flt_fct * rsz_fct

        # erd_npx = 2  # original
        erd_npx = 1  # edited

        # csf_wgt = 0.625  # original
        csf_wgt = 0.5  # edited

        try:  # segment CSF
            t2_map_rsz = torch.nn.functional.interpolate(  # torch.nn.functional.interpolate(
                t2_map.unsqueeze(dim=0).unsqueeze(dim=0),
                scale_factor=rsz_fct,
                mode='bilinear',
                align_corners=False
                ).squeeze()
            csf_msk_t2 = (t2_map_rsz > csf_th_t2)
            csf_t2 = torch.clone(t2_map_rsz)
        except AttributeError:
            t2_map_rsz = cv2.resize(t2_map, None, fx=rsz_fct, fy=rsz_fct, interpolation=cv2.INTER_CUBIC)
            csf_msk_t2 = (t2_map_rsz > csf_th_t2)
            csf_t2 = np.copy(t2_map_rsz)
        # Caculate without vectorization (which is suboptimal in GPGPU)
        # csf_t2[csf_msk_t2] = csf_t2_val * csf_wgt + csf_t2[csf_msk_t2] * (1 - csf_wgt)
        # csf_t2 = csf_t2 + (csf_t2_val * csf_wgt + csf_t2 * (1 - csf_wgt) - csf_t2) * csf_msk_t2.double()
        # csf_t2 = csf_t2 + (csf_t2_val * csf_wgt + csf_t2 * (- csf_wgt)) * csf_msk_t2.double()
        csf_t2 = csf_t2 + (csf_t2_val - csf_t2) * csf_wgt * csf_msk_t2.double()
        if cuda_flg:
            csf_t2 = self.gaussian_filter(csf_t2.unsqueeze(0).unsqueeze(0).float()).squeeze()
        else:
            albumentations.augmentations.functional.gaussian_filter(csf_t2, sigma=flt_ker)
        csf_msk_t2 = morph_dilate_erode_2d(csf_msk_t2, erd_npx, True, cuda_flg)
        try:
            if cuda_flg:
                if not torch.is_tensor(csf_t2):
                    csf_t2 = torch.from_numpy(csf_t2).to(device=dev)
                # t2_map_rsz[csf_msk_t2] = csf_t2[csf_msk_t2]  # old version with vectorization
                t2_map_rsz = csf_t2 * csf_msk_t2.float() + (~csf_msk_t2).float() * t2_map_rsz.float()
        except TypeError:
            if not torch.is_tensor(csf_t2):
                csf_t2 = torch.from_numpy(csf_t2)
            t2_map_rsz[csf_msk_t2] = csf_t2[csf_msk_t2]
        rsz_fct_inv = rsz_fct ** -1
        try:
            t2_map_thr = torch.nn.functional.interpolate(  # torch.nn.functional.interpolate(
                t2_map_rsz.unsqueeze(dim=0).unsqueeze(dim=0),
                size=None,
                scale_factor=rsz_fct_inv,
                mode='bilinear',
                align_corners=False,
                # recompute_scale_factor=False
                ).squeeze()
        except AttributeError:
            t2_map_thr = cv2.resize(t2_map, None, fx=rsz_fct, fy=rsz_fct, interpolation=cv2.INTER_CUBIC)
        t1_map_est = p1 * t2_map_thr + p2
        t1_map_est = torch.max(t1_map_est, torch.zeros_like(t1_map_est))
        return t1_map_est


mri_estimate_t1_ob = MRIEstimateT1()


def mri_ir(pd_m,
           t1_map=torch.tensor([]),
           t_inv=-1,
           t2_map=torch.tensor([]),
           fa_deg=150,
           t_rep=scipy.inf,
           cuda_flg=False):
    """
    Simulates T1-Weighted MR image assuming mono-exponential model

    :param pd_m: proton-density map
    :param t1_map: T1 map
    :param t_inv: inversion time
    :param t2_map: T2_map (according to which T1-map is estimated)
    :param fa_deg: flip angle [degrees]
    :param t_rep: repetition time (currently not used)
    :param cuda_flg: whether to use GPU
    :return: T1-Weighted MR image

    Created by Chen Solomon,  31/08/2020
    """
    if t_rep is not scipy.inf:
        warnings.warn('TR is not used')
    if t_inv == -1:  # handle input
        return pd_m
    if len(t1_map) == 0:  # T1-map is not given -> estimate it
        t1_map = mri_estimate_t1_ob.imp(t2_map, cuda_flg=cuda_flg)
    fa_rad = np.pi * fa_deg / 180  # convert degrees to radians
    factor = 1 - np.cos(fa_rad)
    try:  # parametric maps can be given either as numpy.array or torch.tensor types
        if t1_map.size() != pd_m.size():
            warnings.warn('Proton-density map and T1 map are of different sizes, interpolating T1 map')
            t1_map = torch.nn.functional.interpolate(
                t1_map.unsqueeze(dim=0).unsqueeze(dim=0),
                size=pd_m.size(),
                mode='bilinear',
                align_corners=False
                ).squeeze()
        t1weighted = pd_m * abs(1 - factor * torch.exp(-torch.div(t_inv, t1_map.double())))
    except TypeError:
        t1weighted = pd_m * abs(1 - factor * np.exp(-t_inv / t1_map))
    return t1weighted


def k_space_acq_tse(etl, size_im, pe_ax, numpy_flg=True, cuda_flg=False):
    """
    Returns an LUT determining the order of K space acquisition assuming a turbo spin echo protocol

    :param etl: echo train length
    :param size_im: image size
    :param pe_ax: phase encoding axis
    :param numpy_flg: whether to return numpy.ndarray or torch.Tensor
    :param cuda_flg:whether to use GPU
    :return: k space acquisition indexes
    """
    pe_num = size_im[pe_ax]
    if numpy_flg:
        k_space_idc_vec = np.arange(pe_num)
        k_space_idc_vec = 1 + (k_space_idc_vec * etl) // pe_num
        if pe_ax == 0:
            k_space_idc_vec = np.reshape(k_space_idc_vec, (pe_num, 1))
            tile_mat = (1, size_im[1])
        else:
            tile_mat = (size_im[0], 1)
        k_space_idc_mat = np.tile(k_space_idc_vec, tile_mat)
        k_space_idc_mat = np.fft.fftshift(k_space_idc_mat)
    else:
        if torch.cuda.is_available() and cuda_flg:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
        k_space_idc_vec = torch.arange(pe_num, device=dev)
        # this step is mathematically equivalent to fftshift on the resulting matrix
        k_space_idc_vec = (k_space_idc_vec + pe_num // 2).fmod_(pe_num)
        k_space_idc_vec = 1 + (k_space_idc_vec * etl) // pe_num
        if pe_ax == 0:
            k_space_idc_vec = torch.reshape(k_space_idc_vec, (pe_num, 1))
            tile_mat = (1, size_im[1])
        else:
            tile_mat = (size_im[0], 1)
        k_space_idc_mat = k_space_idc_vec.repeat(tile_mat)
    return k_space_idc_mat


def mri_t2w_tse(pd_m, t2_map, t_echo, etl, pe_ax, numpy_flg=False, cuda_flg=False):
    """
    Simulates T2-weighted MR image using mono-exponential signal model + turbo spin echo acquisition

    :param pd_m: proton density map
    :param t2_map: T2 map
    :param t_echo: echo time
    :param etl: echo train length
    :param pe_ax: phase encoding axis
    :param numpy_flg: whether to return numpy.ndarray or torch.Tensor
    :param cuda_flg:whether to use GPU
    :return: T2 weighted image (TSE)

    Created by chen Solomon
    """
    if etl == 1:  # Simple case: no turbo-spin-echo
        t2weighted_tse = mri_t2w(pd_m, t2_map, t_echo)
        return t2weighted_tse
    if numpy_flg:
        echoes_idc = np.arange(etl)
        echoes = echoes_idc * t_echo * 2 / etl
        size_im = t2_map.shape
        k_space = np.zeros(size_im, dtype='complex128')
        k_space_idc = k_space_acq_tse(etl, size_im, pe_ax)
        for te_idx in echoes_idc:
            cur_te = echoes[te_idx]
            cur_t2w_img = mri_t2w(pd_m, t2_map, cur_te)
            cur_ksp = np.fft.fft2(cur_t2w_img)
            k_space_msk = (k_space_idc == te_idx)
            k_space[k_space_msk] = cur_ksp[k_space_msk]
        t2weighted_tse = np.fft.ifft2(k_space)
        t2weighted_tse = np.abs(t2weighted_tse)
    else:
        if torch.cuda.is_available() and cuda_flg:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
        echoes_idc = torch.arange(etl, device=dev)
        echoes = echoes_idc * t_echo * 2 / etl
        size_im = t2_map.shape
        # imy_t2w = torch.zeros_like(t2_map).reshape(size_im + (1,))
        imy_t2w = torch.zeros(size_im + (1,), dtype=torch.float64, device=dev)
        k_space = torch.zeros(size_im + (2,), dtype=torch.float64, device=dev)
        k_space_idc = k_space_acq_tse(etl, size_im, pe_ax, numpy_flg, cuda_flg).reshape(size_im + (1,))
        # k_space_idc = torch.cat((k_space_idc, k_space_idc), dim=2)
        for te_idx in echoes_idc:
            cur_te = echoes[te_idx]
            cur_t2w_img = mri_t2w(pd_m, t2_map, cur_te).reshape(size_im + (1,))
            cur_t2w_img = torch.cat((cur_t2w_img, imy_t2w), dim=2)
            cur_ksp = cur_t2w_img.fft(2)
            k_space_msk = (k_space_idc == te_idx).double()
            # k_space[k_space_msk] = cur_ksp[k_space_msk]  # mask operations are not efficient on GPU
            k_space += cur_ksp * k_space_msk
        t2weighted_tse = k_space.ifft(2)
        t2weighted_tse = torch.sqrt(t2weighted_tse[:, :, 0] ** 2 + t2weighted_tse[:, :, 1] ** 2).squeeze()
    return t2weighted_tse


def mri_t2flair(
        t_inv,
        t_echo,
        etl,
        pe_ax,
        proton_density,
        t2_map,
        t1_map=(),
        fa_deg=150,
        t_rep=scipy.inf,
        numpy_flg=False,
        cuda_flg=False):
    """
    Simulate inversion recovery

    :param t_inv: inversion time
    :param t_echo: echo time
    :param etl: echo train length
    :param pe_ax: phase encoding axis
    :param proton_density: proton density map
    :param t2_map: t2 map (used to estimate t1 map)
    :param t1_map: t1 map (optional)
    :param fa_deg: flip angle
    :param t_rep: repetition time(currently not used)
    :param numpy_flg: whether to return numpy.ndarray or torch.Tensor
    :param cuda_flg: whether to use GPU
    :return: T2 FLAIR image

    Created by Chen Solomon
    """
    proton_density_ir = mri_ir(proton_density, t1_map, t_inv, t2_map, fa_deg=fa_deg, t_rep=t_rep, cuda_flg=cuda_flg)
    t2flair = mri_t2w_tse(proton_density_ir, t2_map, t_echo, etl, pe_ax, numpy_flg, cuda_flg)
    return t2flair


def load_rnd_qmri_data(
        dat_flr,
        sbj_nme='',
        scn_idx=float('nan'),
        slc_idx=float('nan'),
        cuda_flg=False,
        numpy_flg=False):
    """
    Randomly loads qMRI data

    :param dat_flr: folder from which qMRI data is loaded
    :param sbj_nme: subject to load (if this is '' then subject is drawn randomly)
    :param scn_idx: scan index to choose (if this is nan then subject is drawn randomly)
    :param slc_idx: slice index to choose (if this is nan then subject is drawn randomly)
    :param cuda_flg: flag about whether to use GPU (currently not optimized)
    :param numpy_flg: whether to return data as numpy.array (mutually exclusive with cuda_flag)
    :return: T2 map, Proton-Density map, Freesurfer look-up table, and volumetric resolution [mm/voxel]

    Created by chen Solomon
    """
    # flags determining output modalities: output cannot be numpy.array and use GPU
    assert (not (cuda_flg and numpy_flg)), 'cuda_flg and numpy_flg are mutually exclusive'
    # Get directories
    if sbj_nme == '':
        sbj_lst = os.listdir(dat_flr)
        sbj_num = len(sbj_lst)
        # Remove irrelevant files
        for sbj_idx in range(0, sbj_num):
            sbj_nme = sbj_lst[sbj_idx]
            if not sbj_nme[0].isdigit():
                del sbj_lst[sbj_idx]
        # Choose subject at random
        sbj_num = len(sbj_lst)
        sbj_idx = np.random.randint(sbj_num)
        sbj_nme = sbj_lst[sbj_idx]
    # Load .mat file
    sbj_dir = os.path.join(dat_flr, sbj_nme)
    map_dic = scipy.io.loadmat(sbj_dir)
    map_dic = map_dic['Sbj_dict']
    # Load qT2 Data from .mat file
    qt2_arr = map_dic['qT2'].flatten()[0].flatten()
    pdm_arr = map_dic['PD'].flatten()[0].flatten()
    lut_arr = map_dic['LUT'].flatten()[0].flatten()
    res_arr = map_dic['volres'][0][0][0]
    scn_num = len(qt2_arr)
    if math.isnan(scn_idx):
        scn_idx = torch.randint(scn_num, (1,))
    qt2_vol = qt2_arr[scn_idx]
    pdm_vol = pdm_arr[scn_idx]
    lut_vol = lut_arr[scn_idx]
    res_vec = res_arr[scn_idx].flatten()
    # Load Slice Parameters
    if qt2_vol.ndim == 2:
        qt2_slc = qt2_vol[:, :].astype('f8')
        pdm_slc = pdm_vol[:, :].astype('f8')
        lut_slc = lut_vol[:, :].astype('f8')
    elif qt2_vol.ndim == 3:
        slc_num = qt2_vol.shape[2]
        if math.isnan(slc_idx):
            slc_idx = torch.randint(slc_num, (1,))
        qt2_slc = qt2_vol[:, :, slc_idx].astype('f8')
        pdm_slc = pdm_vol[:, :, slc_idx].astype('f8')
        lut_slc = lut_vol[:, :, slc_idx].astype('f8')
    else:
        raise IndexError('maps are of unsupported dimensions')
    # brn_msk: torch.tensor = lut_slc != 0  # use freesurfer for segmentation
    brn_msk: np.ndarray = get_brn_msk(qt2_slc, pdm_slc, res_vec[0])  # use qT2 & PDm for segmentation
    if not numpy_flg:
        qt2_slc = torch.from_numpy(qt2_slc)
        pdm_slc = torch.from_numpy(pdm_slc)
        lut_slc = torch.from_numpy(lut_slc)
        brn_msk = torch.from_numpy(brn_msk)
        res_vec = torch.from_numpy(res_vec)
        if cuda_flg:
            assert torch.cuda.is_available(), 'cuda is not available'
            cuda = torch.device('cuda')
            qt2_slc = qt2_slc.pin_memory()
            pdm_slc = pdm_slc.pin_memory()
            lut_slc = lut_slc.pin_memory()
            brn_msk = brn_msk.pin_memory()
            qt2_slc = qt2_slc.contiguous().to(device=cuda, non_blocking=True)
            pdm_slc = pdm_slc.contiguous().to(device=cuda, non_blocking=True)
            lut_slc = lut_slc.contiguous().to(device=cuda, non_blocking=True)
            brn_msk = brn_msk.contiguous().to(device=cuda, non_blocking=True)
    return qt2_slc, pdm_slc, lut_slc, brn_msk, res_vec


def morph_dilate_erode_2d(image, npx, erd_flg=False, cuda_flg=False):
    """
    Erode/dilate binary image

    :param image: input image to be processed
    :param npx: number of pixels for erosion/dilation
    :param erd_flg: true gives erosion, false gives dilation
    :param cuda_flg: whether to use GPU (not optimized yet)
    :return: eroded/dilated binary image

    Created by Chen Solomon
    """
    ker_sze = torch.Size([1 + 2 * npx, 1 + 2 * npx])
    if erd_flg:
        image = ~image
    img_inp = image.unsqueeze(0).unsqueeze(0).float()
    if torch.cuda.is_available() and cuda_flg:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    cor_1dx = torch.linspace(0, ker_sze[0] - 1, ker_sze[0], device=dev)
    cor_1dy = torch.linspace(0, ker_sze[1] - 1, ker_sze[1], device=dev)
    cor_2dx, cor_2dy = torch.meshgrid(cor_1dx, cor_1dy)
    ker = torch.zeros(ker_sze, device=dev)
    ker[((cor_2dx - (ker_sze[0] - 1) / 2) ** 2 + (cor_2dy - (ker_sze[1] - 1) / 2) ** 2) <= (npx ** 2)] = 1
    ker = ker.unsqueeze(0).unsqueeze(0).to(device=dev)
    pad_x = (ker_sze[0] - 1) // 2
    pad_y = (ker_sze[1] - 1) // 2
    out_img = torch.nn.functional.conv2d(img_inp, ker, padding=(pad_x, pad_y))
    out_img = (out_img >= 1).squeeze()  # torch.clamp(out_img, 0, 1).squeeze().bool()
    if erd_flg:
        out_img = ~out_img
    return out_img


def get_brn_msk(qt2_slc=np.array(1), pdm_slc=np.array(1), vol_res=1):
    """

    :param qt2_slc: [np.ndarray] T2 (transverse relaxation time) map
    :param pdm_slc: [np.ndarray] PD (proton density)  map
    :param vol_res: [float] volumetric resolution
    :return: [np.ndarray] Brain mask (after skull stripping)
    """
    brn_msk = (qt2_slc > 40) * (pdm_slc > 0)

    erd_msk = int((15 // vol_res))  # Pixels for mask closure.
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erd_msk, erd_msk))
    brn_msk = cv2.erode(brn_msk.astype('uint8'), ker)
    brn_cc = cv2.connectedComponentsWithStats(brn_msk)
    brn_cc_stat = brn_cc[2]
    cc_idc = np.argsort(brn_cc_stat[:,-1])[::-1]
    if cc_idc[0] != 0 or brn_cc[0] == 1:
        cc_idx = cc_idc[0]
    else:
        cc_idx = cc_idc[1]
    brn_msk = (brn_cc[1] == cc_idx).astype('uint8')
    brn_msk = cv2.dilate(brn_msk, ker)
    brn_msk[0, :] = 0
    brn_msk[-1, :] = 0
    brn_msk = cv2.floodFill(~brn_msk, None, (0, 0), 0)[1]
    brn_msk[brn_msk != 0] = 1
    return brn_msk


def gen_rand_roi(pre_mask, roi_size_min, roi_size_max=[], seed=-1, les_num=1, cuda_flg=False):
    """
    Generate random ROI of a certain size

    :param pre_mask: mask that determines allowed pixels to be part of the ROI
    :param roi_size_min: minimal size of ROI (array/tensor of length 2)
    :param roi_size_max: maximal size of ROI, defaulted to equal roi_size_min (array/tensor of length 2)
    :param seed: used for randomization, -1 indicates not using seed
    :param les_num: Number of ROIs (default 1)
    :param cuda_flg: whether to use GPU (not optimized yet)
    :return: binary mask for ROI(s)

    Created by Chen Solomon
    """
    # Handle input
    max_int32 = sys.maxsize / 2**32
    if seed < 0:
        seed = np.random.randint(max_int32)
    rng = np.random.seed(seed=seed)
    if len(roi_size_max) == 0:
        roi_size_max = roi_size_min + 1
    elif (roi_size_max == roi_size_min).any():
        roi_size_max = roi_size_max + 1
    if type(roi_size_min) is not type(roi_size_max):
        raise AssertionError('variables "roi_size_min" and "roi_size_max" should be of the same type')
    if type(roi_size_min) is np.ndarray:
        pass
    elif type(roi_size_min) is torch.Tensor:
        if roi_size_min.is_cuda:
            roi_size_min = roi_size_min.cpu()
        if roi_size_max.is_cuda:
            roi_size_max = roi_size_max.cpu()
        roi_size_min = roi_size_min.numpy()
        roi_size_max = roi_size_max.numpy()
    elif type(roi_size_min) is int:
        roi_size_min = np.array((roi_size_min, roi_size_min))
        roi_size_max = np.array((roi_size_max, roi_size_max))
    elif (type(roi_size_min) != tuple and type(roi_size_min) != torch.Size) or len(roi_size_min) != 2:
        raise(TypeError('roi_size is invalid'))
    if torch.cuda.is_available() and cuda_flg:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    # Define parameters and variables
    roi_size = np.random.randint(roi_size_min, roi_size_max)
    pre_mask[[0, -1], :] = 0  # do not include edges of image
    pre_mask[:, [0, -1]] = 0
    roi_size_arr = np.array(roi_size)
    roi_radius = roi_size_arr // 2
    pre_mask_erd = morph_dilate_erode_2d(pre_mask, np.max(roi_radius), True, cuda_flg)
    # try smaller radii if erosion nullify the mask
    while pre_mask_erd.sum() == 0 and (roi_radius > roi_size_min // 2).all():
        roi_radius = (roi_radius + roi_size_min // 2) // 2
        pre_mask_erd = morph_dilate_erode_2d(pre_mask, np.max(roi_radius), True, cuda_flg)
    im_size = pre_mask.shape
    min_ar: float = 0.1
    roi_a = roi_size[0] * roi_size[1]
    cnt = (roi_size_arr - 1) / 2
    mask = torch.zeros(pre_mask.shape, dtype=torch.uint8, device=dev)
    # Loop over lesions
    for les_idx in range(les_num):
        # Choose random position allowed by pre_mask
        if les_idx > 1:
            mask_dil = morph_dilate_erode_2d(mask, np.max(roi_radius), erd_flg=False, cuda_flg=cuda_flg)
            pre_mask_erd = pre_mask_erd * ~mask_dil
        cor_1dy = torch.arange(0, im_size[0], 1, device=dev)
        cor_1dx = torch.arange(0, im_size[1], 1, device=dev)
        cor_2dy, cor_2dx = torch.meshgrid(cor_1dy, cor_1dx)
        idc_y = cor_2dy[pre_mask_erd]
        idc_x = cor_2dx[pre_mask_erd]
        if len(idc_x) == 0:
            return mask, (), seed
        idx = np.random.choice(len(idc_x))
        pos_y = idc_y[idx]
        pos_x = idc_x[idx]
        # Generate random convex shape (numpy & float *only*, cv2.convexHull does not work o.w.)
        roi = np.zeros(roi_size)
        cor_1dy = np.arange(0, roi_size[0], 1)
        cor_1dx = np.arange(0, roi_size[1], 1)
        cor_2dx, cor_2dy = np.meshgrid(cor_1dy, cor_1dx)
        roi_sup = ((cor_2dy - cnt[0]) / roi_radius[0]) ** 2 + ((cor_2dx - cnt[1]) / roi_radius[1]) ** 2 <= 1
        idc_y = cor_2dy[roi_sup]
        idc_x = cor_2dx[roi_sup]
        # idx = np.random.choice(len(idc_x))
        num_vrt = min(int(np.round(2 * np.sqrt(roi_a))), len(idc_x))  # formula used for Matlab simulations
        # num_vrt = int(np.floor(4 * np.sqrt(roi_a * min_ar)))  # a formula I came up with through trial & error
        vrt_idx = np.random.permutation(len(idc_x))
        vrt_idx = vrt_idx[0:num_vrt]
        vrt_y = idc_y[vrt_idx]
        vrt_x = idc_x[vrt_idx]
        vrt_xy = np.vstack((vrt_x, vrt_y)).T
        vrt_xy = cv2.convexHull(vrt_xy)
        cv2.drawContours(roi, [vrt_xy], 0, 1, thickness=cv2.FILLED)
        # Insert roi to location in image
        roi = torch.from_numpy((roi == 1).astype(np.uint8))
        if cuda_flg:
            roi = roi.pin_memory().contiguous().to(device=dev, non_blocking=True)
        y_hlf_1 = np.floor(roi_size[0] / 2)
        y_hlf_2 = np.ceil(roi_size[0] / 2)
        x_hlf_1 = np.floor(roi_size[1] / 2)
        x_hlf_2 = np.ceil(roi_size[1] / 2)
        pos_y_1 = int(pos_y - y_hlf_1)
        pos_y_2 = int(pos_y + y_hlf_2)
        pos_x_1 = int(pos_x - x_hlf_1)
        pos_x_2 = int(pos_x + x_hlf_2)
        mask[pos_y_1:pos_y_2, pos_x_1:pos_x_2] = roi
        # Compute boundbox for lesion
        box_y = pos_y_1 + min(vrt_y)
        box_h = max(vrt_y) - min(vrt_y) + 1
        box_x = pos_x_1 + min(vrt_x)
        box_w = max(vrt_x) - min(vrt_x) + 1
        # (pos_y_1, pos_x_1, pos_y_2 - pos_y_1, pos_x_2 - pos_x_1)  # old code
        # Inflate image if necessary
        npx = 1  # factor for dilation of mask
        while torch.sum(mask) < (roi_a * min_ar):
            # print('enlarge lesion')
            mask = morph_dilate_erode_2d(mask, npx, cuda_flg)
            box_y = max(0, box_y - 1)
            box_h = min(min(box_y + box_h + 1, im_size[0]) + 1, im_size[0]) - box_y
            box_x = max(0, box_x - 1)
            box_w = min(min(box_x + box_w + 1, im_size[1]) + 1, im_size[1]) - box_x
    seed_1 = random.randrange(max_int32)
    rng = np.random.seed(seed=seed_1)
    return mask, (box_x, box_y, box_w, box_h), seed


def create_roi_layers(roi_mask, num_layers, thc_layers, erd_flg=True):
    """
    Create layered ROI (3D array - each slice is a layer)

    :param roi_mask: initial, un-layered ROI
    :param num_layers: number of layers
    :param thc_layers: thickness of layers
    :param erd_flg: whether the layers are outside (dilation) or inside (erosion) the initial ROI
    :return: layered ROI

    Created by Chen Solomon
    """
    im_size = roi_mask.shape
    roi_layers = torch.zeros(im_size + (num_layers,), dtype=torch.uint8)
    roi_layers[:, :, 0] = torch.clone(roi_mask)
    for lyr_idx in range(1, num_layers):
        lyr_prv_dif = 1 + np.random.choice(min(lyr_idx, 3))
        lyr_prv_idx = lyr_idx - lyr_prv_dif
        lyr_prv_msk = roi_layers[:, :, lyr_prv_idx]
        layer = thc_layers * lyr_prv_dif
        roi_layers[:, :, lyr_idx] = morph_dilate_erode_2d(lyr_prv_msk, layer, erd_flg)
    return roi_layers


def create_lesion(par_map, roi_layers, num_layers, change_ratio=None, change_percent=None):
    """
    Create layered lesion in a parametric map

    :param par_map: parametric map to be manipulated
    :param roi_layers: layered ROI array
    :param num_layers: number of layers in lesion
    :param change_ratio: ratio of change in lesion
    :param change_percent: percentage of change in lesion
    :return: lesioned parametric map

    Created by Chen Solomon
    """
    if change_ratio is None:
        if change_percent is not None:
            change_ratio = 1 + change_percent / 100
        else:
            raise ValueError('input required: either change_ratio or change_percent')
    change_fctr = change_ratio ** (num_layers ** - 1)
    for lyr_idx in range(num_layers):
        cur_layer = roi_layers[:, :, lyr_idx]
        par_map[cur_layer] = par_map[cur_layer] * change_fctr
    return par_map


def create_multiple_lesions(
        seg_mask,
        par_map,
        sever_per_arr,
        lyr_num,
        lyr_thc,
        roi_size_min,
        roi_size_max=[],
        seed=-1,
        change_idc=np.array([]),
        les_num=10,
        les_prob=1.,
        cuda_flg=False):
    """
    Creates random lesions in a map according to parameters given

    :param seg_mask: segmentation of where lesions can appear
    :param par_map: parametric map that gets lesioned
    :param sever_per_arr: array of percentages for pathological changes (not including zero % change)
    :param lyr_num: number of layers per lesion
    :param lyr_thc: thickness of layers
    :param roi_size_min: minimal size of ROI
    :param roi_size_max: maximal size of ROI
    :param seed: used for randomization of lesions, -1 indicates not using seed
    :param change_idc: used to determine severity level
    :param les_num: # of lesions, size of lesions
    :param les_prob: probability of being lesioned
    :param cuda_flg: whether to use GPU (not optimized yet)
    :return: segmentation map, lesioned parametric map

    Created by Chen Solomon
    """
    num_lvl = len(sever_per_arr)
    seg_mask_edt = torch.clone(seg_mask)
    les_roi = torch.zeros_like(par_map)
    les_seg = torch.clone(les_roi)
    par_map_les = torch.clone(par_map)
    unlesioned_flg = np.random.binomial(1, 1 - les_prob)
    bnd_box_lst = []
    if unlesioned_flg or les_num == 0:
        pass
    else:
        if type(change_idc) is np.ndarray:
            if len(change_idc) == 0:
                change_idc = 1 + np.random.choice(num_lvl - 1, les_num)  # if zero level is included
                # change_idc = np.random.choice(num_lvl, les_num)  # zero level is NOT included
            else:
                assert len(change_idc) == les_num, 'len(change_idc) should be equal to les_num'
        elif type(change_idc) is int:
            change_idc = np.array(change_idc)
        else:
            raise TypeError('change_idc is of invalid type')
        for les_idx in range(les_num):
            # roi_size = np.random.randint(roi_size_min, roi_size_max)  # old code: randomize size of lesion
            if les_idx > 0:
                if type(roi_size_max) is list:
                    roi_radius = roi_size_max // 2
                else:
                    roi_radius = roi_size_min // 2
                roi_dil = morph_dilate_erode_2d(les_roi, max(roi_radius), erd_flg=False, cuda_flg=cuda_flg)
                seg_mask_edt = seg_mask_edt * ~roi_dil
            les_roi, bnd_box, seed_1 = gen_rand_roi(seg_mask_edt > 0, roi_size_min, roi_size_max, seed=seed, cuda_flg=cuda_flg)
            if not torch.any(les_roi):
                break
            change_idx = change_idc[les_idx]
            change_prcnt = sever_per_arr[change_idx]
            change_ratio = 1 + change_prcnt / 100
            les_seg = (les_seg.float() + les_roi.float() * (change_idx * 255 / num_lvl)).type(torch.uint8)
            les_layers = create_roi_layers(les_roi, lyr_num, lyr_thc)
            par_map_les = create_lesion(par_map_les, les_layers, lyr_num, change_ratio)
            if len(bnd_box) != 0:
                bnd_box_lst += [bnd_box, ]
    return les_seg, par_map_les, bnd_box_lst


def load_tst_ms_data(
        dat_flr='HT_Data',
        sbj_nme='',
        ref_flg=False,
        img_idx=float('nan'),
        ech_idx=float('nan'),
        numpy_flg=False,
        cuda_flg=False
):
    """
    Loads random possibly lesioned slice from test DB (that was used to evaluate human subject performance)

    :param dat_flr: input folder name
    :param sbj_nme: subject to load (if this is '' then subject is drawn randomly)
    :param ref_flg: whether to look at reference figures (used to train subjects)
    :param img_idx: image index to choose (if this is nan then draw randomly)
    :param ech_idx: echo time (TE) to choose (if this is nan then draw randomly)
    :param numpy_flg: whether to return data as numpy.array (mutually exclusive with cuda_flag)
    :param cuda_flg: whether to use GPU
    :return: clean qT2 map, PD map, lesion segmentation mask, lesioned qT2 map, bound box list for lesions

    Created by Chen Solomon, 07/12/2020
    """
    # flags determining output modalities: output cannot be numpy.array and use GPU
    assert (not (cuda_flg and numpy_flg)), 'cuda_flg and numpy_flg are mutually exclusive'
    # Get directories
    if sbj_nme == '':
        sbj_lst = os.listdir(dat_flr)
        sbj_num = len(sbj_lst)
        # Remove irrelevant files
        del_idx = 0
        for sbj_idx in range(0, sbj_num):
            sbj_nme = sbj_lst[sbj_idx - del_idx]
            if not sbj_nme.endswith('.mat'):
                del sbj_lst[sbj_idx - del_idx]
                del_idx += 1
        # Choose subject at random
        sbj_num = len(sbj_lst)
        sbj_idx = np.random.randint(sbj_num)
        sbj_nme = sbj_lst[sbj_idx]
    elif type(sbj_nme) is tuple:
        sbj_nme = sbj_nme[0]
    # Load .mat file
    sbj_dir = os.path.join(dat_flr, sbj_nme)
    map_dic = scipy.io.loadmat(sbj_dir)
    # Load qT2 Data from .mat file
    qt2_arr = abs(map_dic['T2MapsCell'][0])
    pdm_arr = map_dic['PDMapsCell'][0]
    ana_num = len(qt2_arr)
    # Handle image indexing: echo time (TE)
    ech_arr = map_dic['TEs'].flatten()
    num_ech = len(ech_arr)
    if math.isnan(ech_idx):
        ech_idx = torch.randint(num_ech, (1,))
    # Handle image indexing: "reference" or test images
    num_ref = int(map_dic['NumRef'][0][0])
    ana_arr = map_dic['LesionAnatomy_idxs'].flatten()
    num_img = len(ana_arr)
    seg_arr = map_dic['ROIArray']
    try:  # TODO: debug and add documentation
        # reference flag implies different behavior
        if ref_flg and (
                math.isnan(img_idx)  # if ref_flg then 2 options: 1) choose at random 2) image predefined
                or
                img_idx < num_ref  # if this is a reference image then severity array behaves differently
        ):
            if math.isnan(img_idx):  # randomly choose image from reference
                img_idx = torch.randint(num_ref, (1,))
            #  indices behave differently depending on ref_flg anf img_idx
            sev_arr = map_dic['RefT2_Ratios'].flatten()
            sev_idx = img_idx
            sev_rat = sev_arr[sev_idx]
        else:
            if math.isnan(img_idx):
                tst_img_idx = torch.randint(num_img - num_ref, (1,))
            else:
                tst_img_idx = img_idx - ref_flg * num_ref
            #  indices behave differently depending on ref_flg anf img_idx
            sev_arr = map_dic['T2_Ratios'].flatten()
            img_per_sev = map_dic['NumImagesPerT2'].flatten()[0]
            sev_idx = int(tst_img_idx // img_per_sev)
            sev_rat = sev_arr[sev_idx]
    except IndexError:
        raise IndexError
    # les_seg_arr = seg_arr[img_idx, ech_idx].flatten()  # Old bug 20210720
    les_seg_arr = seg_arr[img_idx + num_ref * (1 - ref_flg), ech_idx].flatten()
    les_lyr_shp = les_seg_arr[0].shape + les_seg_arr.shape
    num_lyr = les_seg_arr.shape[0]
    les_lyr = np.zeros(les_lyr_shp, dtype=np.uint8)
    for lyr_idx in range(num_lyr):
        les_lyr[:, :, lyr_idx] = les_seg_arr[lyr_idx]
    # Choose anatomy
    ana_arr = map_dic['LesionAnatomy_idxs'].flatten()
    # ana_idx = ana_arr[img_idx] - 1  # Old bug 20210720
    ana_idx = ana_arr[img_idx + num_ref * (1 - ref_flg)] - 1
    qt2_slc = qt2_arr[ana_idx].astype('f8') * 1000
    pdm_slc = pdm_arr[ana_idx].astype('f8')
    # Load Slice Parameters
    # if qt2_slc.ndim == 2:
    #     qt2_slc = qt2_slc[:, :].astype('f8')
    #     pdm_slc = pdm_slc[:, :].astype('f8')
    # elif qt2_slc.ndim == 3:  # OPTIONAL: if there is possibility for 3D data
    #     slc_num = qt2_slc.shape[2]
    #     slc_idx = torch.randint(slc_num, (1,))
    #     qt2_slc = qt2_slc[:, :, slc_idx].astype('f8')
    #     pdm_slc = pdm_slc[:, :, slc_idx].astype('f8')
    # else:
    #     raise IndexError('maps are of unsupported dimensions')
    # Create estimated brain mask (no LUT to use so it's not accurate)
    brn_msk = get_brn_msk(qt2_slc, pdm_slc)
    # Mask open: excluded 20210404
    # if not numpy_flg:
    #     # # Hard coded: perform "image open" on brain mask (needed vol_res for data set)
    #     brn_msk = morph_dilate_erode_2d(brn_msk, 5, cuda_flg=cuda_flg)
    #     brn_msk = morph_dilate_erode_2d(brn_msk, 10, erd_flg=True, cuda_flg=cuda_flg)
    #     brn_msk = morph_dilate_erode_2d(brn_msk, 5, cuda_flg=cuda_flg)

    if not numpy_flg:
        qt2_slc = torch.from_numpy(qt2_slc)
        pdm_slc = torch.from_numpy(pdm_slc)
        les_lyr = torch.from_numpy(les_lyr)
        brn_msk = torch.from_numpy(brn_msk)
        if cuda_flg:
            assert torch.cuda.is_available(), 'cuda is not available'
            cuda = torch.device('cuda')
            qt2_slc = qt2_slc.pin_memory()
            pdm_slc = pdm_slc.pin_memory()
            les_lyr = les_lyr.pin_memory()
            brn_msk = brn_msk.pin_memory()
            qt2_slc = qt2_slc.contiguous().to(device=cuda, non_blocking=True)
            pdm_slc = pdm_slc.contiguous().to(device=cuda, non_blocking=True)
            les_lyr = les_lyr.contiguous().to(device=cuda, non_blocking=True)
            brn_msk = brn_msk.contiguous().to(device=cuda, non_blocking=True)
        qt2_slc_les = torch.clone(qt2_slc)
    # Take care of normalization
    pass
    # Get bound-boxes for lesions
    bnd_box_lst = []
    les_seg = les_lyr[:, :, 0]
    if cuda_flg:
        seg_slc = les_seg.cpu().numpy()
    elif not numpy_flg:
        seg_slc = les_seg.numpy()
    seg_slc = seg_slc.astype('uint8')
    num_les, lbl_slc = cv2.connectedComponents(seg_slc)
    for les_idx in range(1, num_les):
        msk = lbl_slc == les_idx
        bnd_box = find_boundbox(msk, numpy_flg=True, min_sze=1, mod=0)
        bnd_box_lst += [bnd_box, ]
    # Add label to les_seg
    sev_arr_tst = map_dic['T2_Ratios'].flatten()
    num_lvl = len(sev_arr_tst) + 1
    change_idc = np.where(sev_arr_tst == sev_rat)  # zero level is NOT included
    if len(change_idc) == 1:
        try:  # handle case where ref_flg is True
            change_idx = change_idc[0][0] + 1
        except IndexError:
            if sev_rat == 1:
                change_idx = 0
            else:
                raise (IndexError(''))
    else:
        raise(IndexError('variable sev_arr_tst is invalid'))
    if numpy_flg:
        np.uint8(les_seg.cpu().numpy() * (change_idx * 255 / num_lvl))
    else:
        les_seg = (les_seg.float() * (change_idx * 255 / num_lvl)).type(torch.uint8)
    # Create lesion in parametric map
    qt2_slc_les = create_lesion(qt2_slc_les, les_lyr, num_lyr, sev_rat)
    return qt2_slc, pdm_slc, les_seg, qt2_slc_les, bnd_box_lst, brn_msk


def synt_rnd_ms_data(
        dat_flr='HS_Data',
        sbj_nme='',
        tst_flg=False,
        ref_flg=False,
        scn_idx=float('nan'),
        slc_idx=float('nan'),
        qt2_flg=False,
        qt1_flg=False,
        t2w_flg=False,
        t2flair_flg=False,
        pd_flg=False,
        cuda_flg=False,
        t2w_t_echo=90,
        flair_t_echo=80,
        flair_t_inv=2372,
        flair_fa=150,
        flair_etl=16,
        pe_ax=1,
        sever_per_arr=(0, 6, 9, 12, 15, 18, 21, 25, 30),
        seed=-1,
        change_idc=np.array([]),
        les_num=1,
        les_prob=0.5,
        les_size=5,
        lyr_size=3,
        roi_factor=1
):
    """
    Generate random WM-lesioned brain scan

    :param dat_flr: input folder name
    :param sbj_nme: subject to load (if this is '' then subject is drawn randomly)
    :param tst_flg: whether to use database used for tests (for humans)
    :param ref_flg: whether to use reference images from test
    :param scn_idx: scan index to choose (if this is nan then subject is drawn randomly)
    :param slc_idx: slice index to choose (if this is nan then subject is drawn randomly)
    :param qt2_flg: whether to return qT2 map
    :param qt1_flg: whether to return qT1 map
    :param t2w_flg: whether to return T2w image
    :param t2flair_flg: whether to return T2 FLAIR image
    :param pd_flg: whether to return proton-density map
    :param cuda_flg: whether to use GPU
    :param t2w_t_echo: echo time for T2 weighted image
    :param flair_t_echo: echo time for T2 FLAIR image
    :param flair_t_inv: inversion time for T2 FLAIR image
    :param flair_fa: flip-angle time for T2 FLAIR image
    :param flair_etl: echo train length for T2 FLAIR image
    :param pe_ax: phase encoding axis for T2 FLAIR image
    :param sever_per_arr: array of percentages for pathological changes
    :param seed: used for randomization of lesions, -1 indicates not using seed
    :param change_idc: used to determine severity level
    :param les_num: number of lesions
    :param les_prob: probability of being lesioned
    :param les_size: [mm] approximate diameter of lesion
    :param lyr_size: [mm] approximate radius of layers between lesioned and healthy tissue
    :param roi_factor: factor by which lesion maximum size is larger then its minimum size
    :return: list containing lesioned MR images according to input flags

    Original version created by Chen Solomon, 04/09/2020
    """
    # Load quantitative maps
    if tst_flg:
        qt2_slc, pdm_slc, les_seg, qt2_slc_les, bnd_box_lst, brn_msk = load_tst_ms_data(
            dat_flr=dat_flr,
            sbj_nme=sbj_nme,
            ref_flg=ref_flg,
            ech_idx=scn_idx,
            img_idx=slc_idx,
            cuda_flg=cuda_flg
        )
    else:
        qt2_slc, pdm_slc, lut_slc, brn_msk, vol_res = load_rnd_qmri_data(
            dat_flr,
            sbj_nme=sbj_nme,
            scn_idx=scn_idx,
            slc_idx=slc_idx,
            cuda_flg=cuda_flg)
        # if numpy_flg:
        #     roi_size = np.ceil(les_size / vol_res[:2]).astype(np.int)
        #     lyr_num = int(np.ceil(lyr_size / vol_res[0]))
        # else:
        # Determine lesion dimensions
        roi_size = torch.ceil(les_size / vol_res[:2]).type(torch.int)
        lyr_num = torch.ceil(lyr_size / vol_res[0]).type(torch.int).data.tolist()
        lyr_thc = 1
        # create lesion
        wm_mask: torch.tensor = (lut_slc == 2) | (lut_slc == 7) | (lut_slc == 41) | (lut_slc == 46)
        les_seg, qt2_slc_les, bnd_box_lst = create_multiple_lesions(
            wm_mask,
            qt2_slc,
            sever_per_arr,
            lyr_num,
            lyr_thc,
            roi_size,
            roi_size * roi_factor,
            seed=seed,
            change_idc=change_idc,
            les_num=les_num,
            les_prob=les_prob,
            cuda_flg=cuda_flg)
    # Normalize PD
    # pdm_med = pdm_slc.masked_select(brn_msk).std()
    # pdm_slc = pdm_slc * (pdm_med ** -1)
    # Create output tuple
    # wm_mask = wm_mask.type(torch.uint8)
    # img_idx=0: Lesion segmentation + img_idx=1: Brain mask
    output_images = [les_seg, brn_msk]
    # img_idx=2: T2 map
    if qt2_flg:
        output_images = output_images + [qt2_slc_les, ]
    else:
        output_images = output_images + [torch.tensor([]), ]
    # img_idx=3: T1 map
    if qt1_flg:
        t1m = mri_estimate_t1_ob.imp(qt2_slc_les, cuda_flg=cuda_flg)
        output_images = output_images + [t1m, ]
    else:
        output_images = output_images + [torch.tensor([]), ]
    # img_idx=4: T2 weighted image
    if t2w_flg:
        t2w = mri_t2w(pdm_slc, qt2_slc_les, t2w_t_echo)
        output_images = output_images + [t2w, ]
    else:
        output_images = output_images + [torch.tensor([]), ]
    # img_idx=5: T2 FLAIR
    if t2flair_flg:
        t1m_cln = mri_estimate_t1_ob.imp(qt2_slc, cuda_flg=cuda_flg)
        t2w_flair = mri_t2flair(
            flair_t_inv,
            flair_t_echo,
            flair_etl,
            pe_ax,
            pdm_slc,
            qt2_slc_les,
            t1_map=t1m_cln,
            fa_deg=flair_fa,
            cuda_flg=cuda_flg)
        # normalize FLAIR
        # t2w_med = t2w_flair.masked_select(brn_msk).std()
        # t2w_flair = t2w_flair * (t2w_med ** -1)
        #
        output_images = output_images + [t2w_flair, ]
    else:
        output_images = output_images + [torch.tensor([]), ]
    # img_idx=6: Proton density map
    if pd_flg:
        output_images = output_images + [pdm_slc, ]
    else:
        output_images = output_images + [torch.tensor([]), ]
    return bnd_box_lst, output_images


def get_nii_vol(
        scn_pth,
        slc_idx=float('nan'),
        slc_flg=False,
        perm_flg=False,
        numpy_flg=False,
        cuda_flg=False):
    """
    Loads volume from .nii file

    :param scn_pth:     Path to .nii data
    :param slc_idx:     slice index to choose (if this is nan then subject is drawn randomly)
                        * relevant if slc_flg is True
    :param slc_flg:     whether to return slice instead of volume
    :param perm_flg:    Whether to permute the volume
    :param numpy_flg:   whether to return data as numpy.array (mutually exclusive with cuda_flag)
    :param cuda_flg:    whether to use GPU
    :return: img_vol:   volumetric data

    Created by Chen Solomon, 05/10/2020
    """
    # flags determining output modalities: output cannot be numpy.array and use GPU
    assert (not (cuda_flg and numpy_flg)), 'cuda_flg and numpy_flg are mutually exclusive'
    if cuda_flg:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Load and permute volume
    img_nii = nib.load(scn_pth)
    img_vol = img_nii.get_fdata(dtype=np.float32)
    if slc_flg:
        get_slc = math.isnan(slc_idx)
        if get_slc:
            img_slc_sum = img_vol.sum((0, 1))
            rel_slc_idc = np.nonzero(img_slc_sum)[0]
            # Choose non-zero slice
            slc_num = len(rel_slc_idc)
            slc_idx_idx = np.random.randint(slc_num)
            slc_idx = rel_slc_idc[slc_idx_idx]
        img_vol = img_vol[:, :, slc_idx]
        img_vol = img_vol.reshape(img_vol.shape + (1,))
    if numpy_flg and perm_flg:
        img_vol = np.transpose(img_vol, (1, 0, 2)).astype(np.float64)
    elif numpy_flg:
        img_vol = img_vol.astype(np.float64)
    else:
        if cuda_flg:
            img_vol = torch.from_numpy(img_vol).pin_memory()
            img_vol = img_vol.contiguous().to(device=device, non_blocking=True, dtype=torch.float64)
        else:
            img_vol = torch.from_numpy(img_vol).to(device=device, dtype=torch.float64)
        if perm_flg:
            img_vol = img_vol.permute(1, 0, 2)
    if slc_flg:
        img_vol = img_vol.squeeze()
        if get_slc:
            return img_vol, slc_idx
    return img_vol


def load_rnd_ms_data(
        dat_flr='MS_Data',
        sbj_nme='',
        slc_idx=float('nan'),
        fin_sze=(),
        t2w_flg=False,
        t2flair_flg=False,
        numpy_flg=False,
        cuda_flg=False):
    """
    Loads data from MS patients dataset (source for data: http://lit.fe.uni-lj.si/tools)

    :param dat_flr:  [str]  input folder name
    :param sbj_nme: subject to load (if this is '' then subject is drawn randomly)
    :param slc_idx: slice index to choose (if this is nan then subject is drawn randomly)
    :param fin_sze: [tuple of the form (height, width)] determine output final size
    :param t2w_flg: whether to return T2w image
    :param t2flair_flg: whether to return T2 FLAIR image
    :param numpy_flg: whether to return data as numpy.array (mutually exclusive with cuda_flag)
    :param cuda_flg: whether to use GPU
    :return: list containing lesioned MR images according to input flags

    Created by Chen Solomon, 02/10/2020
    """
    # flags determining output modalities: output cannot be numpy.array and use GPU
    assert (not (cuda_flg and numpy_flg)), 'cuda_flg and numpy_flg are mutually exclusive'
    # if cuda_flg:
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    # Get directories
    if sbj_nme == '':
        sbj_lst = [fle_nme for fle_nme in os.listdir(dat_flr) if 'patient' in fle_nme]
        # Choose subject at random
        sbj_num = len(sbj_lst)
        sbj_idx = np.random.randint(sbj_num)
        sbj_nme = sbj_lst[sbj_idx]
    # Load .nii image
    sbj_dir = os.path.join(dat_flr, sbj_nme)
    scn_lst = [x for x in os.listdir(sbj_dir) if sbj_nme in x]
    # Determine relevant slices: load brainmask and exclude slices with no brain
    brn_key = 'brainmask'
    img_fnm = [x for x in scn_lst if brn_key in x][0]
    brn_pth = os.path.join(sbj_dir, img_fnm)
    # vol_res = nib.load(brn_pth).header.get_zooms()
    # print(vol_res)
    if math.isnan(slc_idx):
        brn_slc, slc_idx = get_nii_vol(
            brn_pth,
            slc_flg=True,
            perm_flg=True,
            numpy_flg=numpy_flg,
            cuda_flg=cuda_flg)
    else:
        brn_slc = get_nii_vol(
            brn_pth,
            slc_idx=slc_idx,
            slc_flg=True,
            perm_flg=True,
            numpy_flg=numpy_flg,
            cuda_flg=cuda_flg)
    # Load images
    output_images = [torch.tensor([])] * 7  # initialize output
    # img_idx=0: Lesion segmentation- non-existent
    msk_key = 'consensus'
    img_fnm = [x for x in scn_lst if msk_key in x][0]
    scn_pth = os.path.join(sbj_dir, img_fnm)
    output_images[0] = get_nii_vol(
        scn_pth,
        slc_idx=slc_idx,
        slc_flg=True,
        perm_flg=True,
        numpy_flg=numpy_flg,
        cuda_flg=cuda_flg)
    # img_idx=1: Brain mask
    output_images[1] = brn_slc
    # img_idx=2: T2 map             - non-existent
    # img_idx=3: T1 map             - non-existent
    # img_idx=4: T2 weighted image
    if t2w_flg:
        t2w_key = 'T2W.nii'
        img_fnm = [x for x in scn_lst if t2w_key in x][0]
        scn_pth = os.path.join(sbj_dir, img_fnm)
        output_images[4] = get_nii_vol(
            scn_pth,
            slc_idx=slc_idx,
            slc_flg=True,
            perm_flg=True,
            numpy_flg=numpy_flg,
            cuda_flg=cuda_flg)
    # img_idx=5: T2 FLAIR
    if t2flair_flg:
        flair_key = 'FLAIR.nii'
        img_fnm = [x for x in scn_lst if flair_key in x][0]
        scn_pth = os.path.join(sbj_dir, img_fnm)
        output_images[5] = get_nii_vol(
            scn_pth,
            slc_idx=slc_idx,
            slc_flg=True,
            perm_flg=True,
            numpy_flg=numpy_flg,
            cuda_flg=cuda_flg)
    # img_idx=6: Proton density map - non-existent
    # Determine output size (Hard coded to match dataset, can be adjusted to be adaptive according to vol_res)
    if len(fin_sze) == 0:
        rsz_dim = (256, 150)
    else:
        rsz_dim = (fin_sze[0],)
        rsz_dim += (math.ceil(fin_sze[0] * 0.46875 / 0.8),)
    # for loop over output list
    for img_idx in range(0, len(output_images)):
        img_slc = output_images[img_idx]
        if len(img_slc):
            # adjust image dimensions
            if numpy_flg:
                cv2.resize(img_slc, dsize=rsz_dim)
            else:
                img_slc = img_slc.unsqueeze(0).unsqueeze(0)
                img_slc = torch.nn.functional.interpolate(img_slc, size=rsz_dim).squeeze()
            output_images[img_idx] = img_slc
    # get lesions bound boxes
    bnd_box_lst = []
    if cuda_flg:
        seg_slc = output_images[0].cpu().numpy()
    elif not numpy_flg:
        seg_slc = output_images[0].numpy()
    seg_slc = seg_slc.astype('uint8')
    num_les, lbl_slc = cv2.connectedComponents(seg_slc)
    for les_idx in range(1, num_les):
        msk = lbl_slc == les_idx
        bnd_box = find_boundbox(msk, numpy_flg=True, min_sze=1, mod=0)
        bnd_box_lst += [bnd_box, ]
    return bnd_box_lst, output_images


def norm_map(q_map, msk=[], grd_flg=False, x_grd=torch.tensor([]), y_grd=torch.tensor([]), nrm_fct=torch.tensor([])):
    """
    Normalize image and return normalization parameters: mean and max

    :param q_map:   an image whose pixel values are to be normalized to the interval [-1, 1]
    :param grd_flg: whether to return grid with the image
    :param x_grd:   grid for x axis (dim=1)
    :param y_grd:   grid for y axis (dim=0)
    :param nrm_fct: normalization factor 1 & 2 (expectation & standard variation)
    :return: normalized map along with its maximum and mean value

    Created by Chen Solomon, 09/2020
    Edited by Omer Shmueli, 10/2020
    """
    q_map_1 = q_map.double()
    dimention = len(q_map_1.shape)
    if len(nrm_fct) != 0:
        nrm_avg = nrm_fct[0]
        nrm_std = nrm_fct[1]
        if nrm_avg == -1 or nrm_std == -1:
            nrm_sum = (q_map_1 * msk).sum()
            nrm_n = max(msk.sum(), 2)
            nrm_avg = nrm_sum / nrm_n
            nrm_ssq = (((q_map_1 - nrm_avg) * msk) ** 2).sum()
            nrm_std = (nrm_ssq / (nrm_n - 1)).sqrt()
        q_map_1 = (q_map_1 - nrm_avg) / nrm_std
    if grd_flg and len(x_grd) != 0 and len(y_grd) != 0:
        chl_1 = y_grd.unsqueeze(dim=dimention).to(dtype=q_map_1.dtype)
        chl_2 = q_map_1.unsqueeze(dim=dimention)
        chl_3 = x_grd.unsqueeze(dim=dimention).to(dtype=q_map_1.dtype)
    else:  # map_max.data.tolist() == 0:
        chl_1 = q_map_1.unsqueeze(dim=dimention)  # torch.zeros_like(q_map).unsqueeze(dim=dimention)
        chl_2 = q_map_1.unsqueeze(dim=dimention)  # torch.zeros_like(q_map).unsqueeze(dim=dimention)
        chl_3 = q_map_1.unsqueeze(dim=dimention)  # torch.zeros_like(q_map).unsqueeze(dim=dimention)
    return torch.cat((chl_1, chl_2, chl_3), -1)


def find_boundbox(msk, numpy_flg=False, sqr_flg=False, min_sze=64, mod=0):
    """
    Find boundbox of non-zero elements in binary mask

    :param msk:         binary mask (or any mask whose background is zeros)
    :param numpy_flg:   [bool] whether mask is numpy.array (o.w. assumed to be torch.Tensor)
    :param sqr_flg:     [bool] whether to return square boundbox (unless it causes box to index elements out of range)
    :param min_sze:     [int]  determines minimum size of boundbox
    :param mod:         [int]  can be 0 or 1, used to change order of returned indices' axes
    :return:            [tuple of ints] boundbox measures: top-left corner and size (id_x1, id_y1, width, height)

    Created by Chen Solomon, 13/10/2020
    """
    msk_sum_x = msk.sum((mod,))
    msk_sum_y = msk.sum((1 - mod,))
    if numpy_flg:
        idc_x = np.nonzero(msk_sum_x)[0]
        idc_y = np.nonzero(msk_sum_y)[0]
    else:
        idc_x = torch.nonzero(msk_sum_x).squeeze()
        idc_y = torch.nonzero(msk_sum_y).squeeze()
    try:
        if len(idc_y) == 0:
            return 0, 0, msk.shape[1 - mod], msk.shape[mod]
        elif numpy_flg:
            id_x1 = idc_x[0]
            id_x2 = idc_x[-1]
            id_y1 = idc_y[0]
            id_y2 = idc_y[-1]
        else:
            id_x1 = idc_x[0].data.tolist()
            id_x2 = idc_x[-1].data.tolist()
            id_y1 = idc_y[0].data.tolist()
            id_y2 = idc_y[-1].data.tolist()
    except TypeError:
        return 0, 0, msk.shape[1 - mod], msk.shape[mod]
    width = id_x2 - id_x1 + 1
    height = id_y2 - id_y1 + 1
    max_dim = max(width, height, min_sze - 1)
    if sqr_flg or max_dim < min_sze:
        id_x1 = max(0, id_x1 - (max_dim - width) // 2)
        id_y1 = max(0, id_y1 - (max_dim - height) // 2)
        width = min(max_dim, msk.shape[1 - mod] - id_x1)
        height = min(max_dim, msk.shape[mod] - id_y1)
    return id_x1, id_y1, width, height


def get_resize(fin_sze, pad_flg, ini_sze):
    """
    get resize parameters

    :param fin_sze: [tuple of the form (height, width)] determine output final size
    :param pad_flg: whether to stretch or pad with zeroes
    :param ini_sze: [tensor.Size] determine initial size
    :return: rsz_dim: dimensions for resize
             pad_dim: dimensions for zero-padding

    Created by Chen Solomon, 01/10/2020
    """
    if pad_flg:
        # determine resize dimensions
        sze_img = torch.tensor(list(ini_sze))
        fin_dim_max = max(fin_sze)
        rsz_dim = ((sze_img * torch.tensor(fin_dim_max, dtype=torch.float)).div(max(sze_img)))
        rsz_dim = rsz_dim.int().data.tolist()
        # determine padding dimension
        pad_y_1 = int(math.ceil((fin_sze[0] - rsz_dim[0]) / 2.))
        pad_y_2 = int(math.floor((fin_sze[0] - rsz_dim[0]) / 2.))
        pad_x_1 = int(math.ceil((fin_sze[1] - rsz_dim[1]) / 2.))
        pad_x_2 = int(math.floor((fin_sze[1] - rsz_dim[1]) / 2.))
        pad_dim = (pad_x_1, pad_x_2, pad_y_1, pad_y_2)
    else:
        rsz_dim = fin_sze
        pad_dim = (0, 0)
    return rsz_dim, pad_dim


def get_padded_grid(
        org_shp,
        rsz_dim,
        fin_sze,
        crp_flg=True,
        trn_flg=False,
        crop_y1=0,
        crop_y2=-1,
        crop_x1=0,
        crop_x2=-1,
        pad_dim=(0, 0, 0, 0),
        cuda_flg=False):
    """
    Generate padded grid for an image according to its resize and padding

    :param org_shp: original image shape
    :param rsz_dim: image resize parameters
    :param fin_sze: final size of image
    :param crp_flg: whether image is cropped
    :param trn_flg: whether to truncate grid at the edges of crop or to continue it linearly
    :param crop_y1: 1st axis 1st crop idx
    :param crop_y2: 1st axis 2nd crop idx
    :param crop_x1: 2nd axis 1st crop idx
    :param crop_x2: 2nd axis 2nd crop idx
    :param pad_dim: num of rows / columns padded
    :param cuda_flg: whether to use GPU
    :return:
    """
    # Get mesh grid for image
    if cuda_flg:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    y_max = org_shp[0]
    x_max = org_shp[1]
    y_axe = torch.arange(y_max, device=dev, dtype=torch.float64)
    x_axe = torch.arange(x_max, device=dev, dtype=torch.float64)
    y_axe = (y_axe - 0.5 * (y_max - 1)) * 2 / y_max
    x_axe = (x_axe - 0.5 * (x_max - 1)) * 2 / x_max
    if crop_y2 == -1 or crop_x2 == -1:
        crop_y2 = y_max
        crop_x2 = x_max
    if trn_flg:
        if crp_flg:
            y_axe = y_axe[crop_y1:crop_y2]
            x_axe = x_axe[crop_x1:crop_x2]
        y_axe = y_axe.unsqueeze(0).unsqueeze(0)
        x_axe = x_axe.unsqueeze(0).unsqueeze(0)
        y_axe = torch.nn.functional.interpolate(y_axe, size=rsz_dim[0], mode='linear').squeeze()
        x_axe = torch.nn.functional.interpolate(x_axe, size=rsz_dim[1], mode='linear').squeeze()
        y_ax1 = torch.ones((fin_sze[0],), dtype=torch.float64, device=dev)
        x_ax1 = torch.ones((fin_sze[1],), dtype=torch.float64, device=dev)
        y_ax1[0:pad_dim[2]] = -1
        y_ax1[pad_dim[2]:fin_sze[0] - pad_dim[3]] = y_axe
        x_ax1[0:pad_dim[0]] = -1
        x_ax1[pad_dim[0]:fin_sze[1] - pad_dim[1]] = x_axe
    else:
        grd_y_min = y_axe[crop_y1]
        grd_y_max = y_axe[crop_y2 - 1]
        grd_x_min = x_axe[crop_x1]
        grd_x_max = x_axe[crop_x2 - 1]
        grd_y_stp = (grd_y_max - grd_y_min) / (rsz_dim[0] - 1)
        grd_x_stp = (grd_x_max - grd_x_min) / (rsz_dim[1] - 1)
        grd_y_min = grd_y_min - (pad_dim[2] + 1) * grd_y_stp
        grd_y_max = grd_y_max + pad_dim[3] * grd_y_stp
        grd_x_min = grd_x_min - (pad_dim[0] + 1) * grd_x_stp
        grd_x_max = grd_x_max + pad_dim[1] * grd_x_stp
        y_ax1 = torch.linspace(grd_y_min, grd_y_max, fin_sze[0], dtype=torch.float64, device=dev)
        x_ax1 = torch.linspace(grd_x_min, grd_x_max, fin_sze[0], dtype=torch.float64, device=dev)
    # return grids: y_grd, x_grd = torch.meshgrid(y_ax1, x_ax1)
    return torch.meshgrid(y_ax1, x_ax1)


def get_qmri_data_list(
        dat_flr,
        ms__flg=False,
        tst_flg=False,
        ref_flg=False,
        erd_thc=0
        ):
    """ randomly loads qMRI data
    :param dat_flr: folder from which qMRI data is loaded
    :param ms__flg: Whether data is synthesized or original
    :param tst_flg: Whether synthetic data is from human test DB
    :param ref_flg: if showing images from test, whether to show images from reference phase of the test
    :param erd_thc: thickness to erode wm by to decide whether to use slice [mm]
    :return: List of slices (T2 map, Proton-Density map, Freesurfer look-up table, and volumetric resolution [mm/voxel])

    Written by Omer Shmueli
    Edited by Chen Solomon
    """
    samples_list = []
    if ms__flg:
        sbj_lst = [fle_nme for fle_nme in os.listdir(dat_flr) if 'patient' in fle_nme]
        # Loop over subjects
        sbj_num = len(sbj_lst)
        for sbj_idx in range(sbj_num):
            sbj_nme = sbj_lst[sbj_idx]
            # Load .nii image
            sbj_dir = os.path.join(dat_flr, sbj_nme)
            # Determine relevant slices: load brainmask and exclude slices with no brain
            brn_key = 'brainmask'
            img_fnm = [x for x in os.listdir(sbj_dir) if brn_key in x][0]
            brn_pth = os.path.join(sbj_dir, img_fnm)
            brn_vol = get_nii_vol(brn_pth, perm_flg=False)
            tresh = 25
            brn_slc_sum = (brn_vol.sum((0, 1)) > tresh).int()
            rel_slc_idc = torch.nonzero(brn_slc_sum).squeeze()
            slc_num = len(rel_slc_idc)
            for slc_idx_idx in range(slc_num):
                slc_idx = rel_slc_idc[slc_idx_idx].data.tolist()
                samples_list.append((sbj_nme, float('nan'), slc_idx))
    elif tst_flg:
        # Get directories
        sbj_lst = os.listdir(dat_flr)
        sbj_num = len(sbj_lst)
        # Remove irrelevant files
        del_idx = 0
        for sbj_idx in range(0, sbj_num):
            sbj_nme = sbj_lst[sbj_idx - del_idx]
            if not sbj_nme.endswith('.mat'):
                del sbj_lst[sbj_idx - del_idx]
                del_idx += 1
        # Loop over subjects
        sbj_num = len(sbj_lst)
        for sbj_idx in range(sbj_num):
            sbj_nme = sbj_lst[sbj_idx]
        # Load .mat file
            sbj_dir = os.path.join(dat_flr, sbj_nme)
            map_dic = scipy.io.loadmat(sbj_dir)
            ech_arr = map_dic['TEs'].flatten()
            num_ech = len(ech_arr)
            # Handle image number: "reference" or test images
            num_ref = int(map_dic['NumRef'][0][0])
            ana_arr = map_dic['LesionAnatomy_idxs'].flatten()
            ana_arr = ana_arr - 1
            sbj_nme_arr = map_dic['Sbj_nme'][0]
            sbj_nme_arr = sbj_nme_arr[ana_arr]
            if not ref_flg:
                ana_arr = ana_arr[num_ref:]
            num_img = len(ana_arr)
            # Default: sort img_idc s.t. anatomies appear in order
            img_idc = ana_arr.argsort()
            for ech_idx in range(num_ech):
                for img_idx_idx in range(num_img):
                    img_idx = img_idc[img_idx_idx]
                    sbj_nme_2 = sbj_nme_arr[img_idx][0]
                    sbj_nme_tup = (sbj_nme, sbj_nme_2)
                    samples_list.append((sbj_nme_tup, ech_idx, img_idx))
    else:
        # Get directories
        sbj_lst = os.listdir(dat_flr)
        sbj_num = len(sbj_lst)
        # Remove irrelevant files
        del_idx = 0
        for sbj_idx in range(0, sbj_num):
            sbj_nme = sbj_lst[sbj_idx - del_idx]
            if not sbj_nme[0].isdigit():
                del sbj_lst[sbj_idx - del_idx]
                del_idx += 1
        # Loop over subjects
        sbj_num = len(sbj_lst)
        for sbj_idx in range(sbj_num):
            sbj_nme = sbj_lst[sbj_idx]
            # Load .mat file
            sbj_dir = os.path.join(dat_flr, sbj_nme)
            map_dic = scipy.io.loadmat(sbj_dir)
            map_dic = map_dic['Sbj_dict']
            # Load qT2 Data from .mat file
            brn_arr = map_dic['LUT'].flatten()[0].flatten()
            scn_num = len(brn_arr)
            for scn_idx in range(scn_num):
                brn_vol = brn_arr[scn_idx]
                # Load Slice Parameters
                if brn_vol.ndim == 2:
                    # brn_slc = brn_vol[:, :].astype('f8')
                    slc_num = 0
                elif brn_vol.ndim == 3:
                    wm_vol = (brn_vol == 2) | (brn_vol == 7) | (brn_vol == 41) | (brn_vol == 46)
                    # if erd_npx != 0:
                    #     slc_num_tot =
                    wm_slc_sum = wm_vol.sum((0, 1))
                    rel_slc_idc = wm_slc_sum.nonzero()[0]
                    slc_num = len(rel_slc_idc)
                else:
                    # warnings.warn('Volume is of unsupported dimensionality')
                    continue
                # calculate number of pixels for erosion
                res = map_dic['volres'][0][0][0][0][0][0:2].min()
                erd_npx = int(math.ceil(erd_thc / res))
                for slc_idx_idx in range(slc_num):
                    slc_idx = rel_slc_idc[slc_idx_idx].data.tolist()
                    if erd_npx !=0:
                        slc_tns = torch.from_numpy(wm_vol[:, :, slc_idx])
                        slc_tns_erd = morph_dilate_erode_2d(slc_tns, erd_npx, erd_flg=True)
                        if slc_tns_erd.sum() != 0:
                            samples_list.append((sbj_nme, scn_idx, slc_idx))
                    else:
                        samples_list.append((sbj_nme, scn_idx, slc_idx))
    return samples_list


def get_qmri_norm_dict(
        dat_flr,
        samples_list,
        sbj_nme='',
        fin_sze=(),
        pad_flg=False,
        crp_flg=True,
        grd_flg=True,
        ms__flg=False,
        tst_flg=False,
        ref_flg=False,
        qt2_flg=False,
        qt1_flg=False,
        t2w_flg=False,
        t2flair_flg=False,
        pd_flg=False,
        cuda_flg=False,
        t2w_t_echo=90,
        flair_t_echo=80,
        flair_t_inv=2372,
        flair_fa=150,
        flair_etl=16,
        pe_ax=1,
        sever_per_arr=(0, 6, 9, 12, 15, 18, 21, 25, 30),
        les_num=1,
        les_prob=0.5,
        les_size=10,
        ):
    """
    load qMRI data and returns normalization parameters
    :param dat_flr:         folder from which qMRI data is loaded
    :param samples_list:    list of samples data
    :param sbj_nme:         (optional) specification of subject file name
    :param fin_sze:         [tuple of the form (height, width)] determine output final size
    :param pad_flg:         whether to stretch or pad with zeroes
    :param crp_flg:         whether to crop image according to brain mask
    :param grd_flg:         whether to return grid with th image
    :param ms__flg:         Whether data is synthesized or original
    :param tst_flg:         Whether synthetic data is from human test DB
    :param ref_flg:         if tst_flg whether to show ref images
    :param qt2_flg:         whether to return qT2 map
    :param qt1_flg:         whether to return qT1 map
    :param t2w_flg:         whether to return T2w image
    :param t2flair_flg:     whether to return T2 FLAIR image
    :param pd_flg:          whether to return proton-density map
    :param cuda_flg:        whether to use GPU
    :param t2w_t_echo:      echo time for T2 weighted image
    :param flair_t_echo:    echo time for T2 FLAIR image
    :param flair_t_inv:     inversion time for T2 FLAIR image
    :param flair_fa:        flip-angle time for T2 FLAIR image
    :param flair_etl:       echo train length for T2 FLAIR image
    :param pe_ax:           phase encoding axis for T2 FLAIR image
    :param sever_per_arr:   array of percentages for pathological changes
    :param les_num:         number of lesions
    :param les_prob:        probability of being lesioned
    :param les_size:        [mm] approximate diameter of lesion
    :param lyr_size:        [mm] approximate radius of layers between lesioned and healthy tissue
    :param roi_factor:      factor by which lesion maximum size is larger then its minimum size
    :return: List of slices normalization parameters

    Written by Chen Solomon,
    Based on functions:
     1) "get_qmri_data_list", written by Omer Shmueli
     2) generate_new_scan", written by Chen Solomon
     and on script:
     1) Generate_new_scan_test.py
     written by Chen Solomon and Omer Shmueli
    """
    # Filter irrelevant subjects
    if not sbj_nme:
        pass
    else:
        samples_list = [sample for sample in samples_list if sample[0] == sbj_nme]
    # initialize variables
    normal_dict_tmp = {}
    normal_dict = {}
    frm_num = 5
    if len(samples_list) > 0:
        for idx in range(len(samples_list)):
            if ms__flg:
                les_roi, brn_msk, qt2, qt1, t2w, t2flair, pdm = generate_new_scan(  # the relevant function
                        dat_flr=dat_flr,
                        sbj_nme=samples_list[idx][0],
                        scn_idx=samples_list[idx][1],
                        slc_idx=samples_list[idx][2],
                        fin_sze=fin_sze,
                        pad_flg=pad_flg,
                        crp_flg=crp_flg,
                        grd_flg=grd_flg,
                        ms__flg=ms__flg,
                        qt2_flg=qt2_flg,
                        qt1_flg=qt1_flg,
                        t2w_flg=t2w_flg,
                        t2flair_flg=t2flair_flg,
                        cuda_flg=cuda_flg,
                        t2w_t_echo=t2w_t_echo,  # Scan parameters (predefined)
                        flair_t_echo=flair_t_echo,
                        flair_t_inv=flair_t_inv,
                        flair_fa=flair_fa,
                        flair_etl=flair_etl,
                        pe_ax=pe_ax)[1]
                pass
            elif tst_flg:
                les_roi, brn_msk, qt2, qt1, t2w, t2flair, pdm = generate_new_scan(  # the relevant function
                        dat_flr=dat_flr,
                        sbj_nme=samples_list[idx][0],
                        scn_idx=samples_list[idx][1],
                        slc_idx=samples_list[idx][2],
                        fin_sze=fin_sze,
                        pad_flg=pad_flg,
                        crp_flg=crp_flg,
                        grd_flg=grd_flg,
                        ms__flg=ms__flg,
                        tst_flg=tst_flg,
                        ref_flg=ref_flg,
                        qt2_flg=qt2_flg,
                        qt1_flg=qt1_flg,
                        t2w_flg=t2w_flg,
                        t2flair_flg=t2flair_flg,
                        pd_flg=pd_flg,
                        cuda_flg=cuda_flg,
                        t2w_t_echo=t2w_t_echo,  # Scan parameters (predefined)
                        flair_t_echo=flair_t_echo,
                        flair_t_inv=flair_t_inv,
                        flair_fa=flair_fa,
                        flair_etl=flair_etl,
                        pe_ax=pe_ax)[1]
            else:
                les_roi, brn_msk, qt2, qt1, t2w, t2flair, pdm = generate_new_scan(  # the relevant function
                        dat_flr=dat_flr,
                        sbj_nme=samples_list[idx][0],
                        scn_idx=samples_list[idx][1],
                        slc_idx=samples_list[idx][2],
                        fin_sze=fin_sze,
                        pad_flg=pad_flg,
                        crp_flg=crp_flg,
                        grd_flg=grd_flg,
                        ms__flg=ms__flg,
                        qt2_flg=qt2_flg,
                        qt1_flg=qt1_flg,
                        t2w_flg=t2w_flg,
                        t2flair_flg=t2flair_flg,
                        pd_flg=pd_flg,
                        sever_per_arr=sever_per_arr,
                        les_num=les_num,
                        les_prob=les_prob,
                        les_size=les_size,
                        cuda_flg=cuda_flg,
                        t2w_t_echo=t2w_t_echo,  # Scan parameters (predefined)
                        flair_t_echo=flair_t_echo,
                        flair_t_inv=flair_t_inv,
                        flair_fa=flair_fa,
                        flair_etl=flair_etl,
                        pe_ax=pe_ax)[1]
            brn_msk = brn_msk.double()
            msk = brn_msk
            # alternative mask computed with qT2
            if not ms__flg:
                qT2_slc = qt2[:, :, 1]
                qt2_mode = scipy.stats.mode(qT2_slc[brn_msk.bool()].cpu()).mode[0]
                wm_msk = (qT2_slc > (qt2_mode - 5)) * (qT2_slc < (qt2_mode + 5))
                wm_msk = wm_msk.double()
                msk = msk * wm_msk
            msk_sum = msk.sum().int()
            if msk_sum == 0:
                pass
            else:
                sbj_cur = (samples_list[idx][0], samples_list[idx][1])
                [
                    avg_vec,  std_vec,  avg_vec2,  std_vec2,
                    avg_2vec, std_2vec, avg_2vec2, std_2vec2,
                    avg_3vec, std_3vec, avg_3vec2, std_3vec2,
                    num_vec
                ] = normal_dict_tmp.get(
                        sbj_cur,
                        [
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.double),
                            torch.zeros(frm_num, dtype=torch.int)
                         ]
                    )
                for frm_idx in range(frm_num):
                    if frm_idx == 0:
                        img = qt2
                    elif frm_idx == 1:
                        img = qt1.double()
                    elif frm_idx == 2:
                        img = t2w
                    elif frm_idx == 3:
                        img = t2flair
                    elif frm_idx == 4:
                        img = pdm
                    # ignore empty images
                    if len(img) == 0:
                        if num_vec[frm_idx] == 0:
                            num_vec[frm_idx] = 1
                        continue
                    elif len(img.shape) == 3:
                        img = img[:, :, 1].squeeze()
                    # Compute average and std
                    avg, std,  upper_std, lower_std, med = get_img_stt(img, msk, msk_sum)
                    # compute avg and std while ignoring values far from the median
                    maskrel = ((img > med - 3 * lower_std) * (img < med + 3 * upper_std)).double() * msk
                    avg2, std2, upper_std, lower_std, med2 = get_img_stt(img, maskrel)
                    maskrel = ((img > med2 - 1 * lower_std) * (img < med2 + 0.5 * upper_std)).double() * msk
                    avg3, std3, upper_std, lower_std, med3 = get_img_stt(img, maskrel)
                    # if idx>500 and False:
                    #     plt.figure(1)
                    #     plt.imshow(maskrel.cpu().numpy())
                    #     plt.figure(2)
                    #     plt.imshow(msk.cpu().numpy())
                    #     plt.figure(3)
                    #     plt.imshow(img.cpu().numpy())
                    #     plt.show()
                    avg_vec[frm_idx] += avg  # avg * 1
                    std_vec[frm_idx] += std * 1
                    avg_vec2[frm_idx] += avg * avg * 1
                    std_vec2[frm_idx] += std * std * 1
                    avg_2vec[frm_idx] += avg2  # avg2 * 1
                    std_2vec[frm_idx] += std2 * 1
                    avg_2vec2[frm_idx] += avg2 * avg2 * 1
                    std_2vec2[frm_idx] += std2 * std2 * 1
                    avg_3vec[frm_idx] += avg3  # avg3 * 1
                    std_3vec[frm_idx] += std3 * 1
                    avg_3vec2[frm_idx] += avg3 * avg3 * 1
                    std_3vec2[frm_idx] += std3 * std3 * 1
                    num_vec[frm_idx] += 1
                    normal_dict_tmp[sbj_cur] = [
                        avg_vec,  std_vec,  avg_vec2,  std_vec2,
                        avg_2vec, std_2vec, avg_2vec2, std_2vec2,
                        avg_3vec, std_3vec, avg_3vec2, std_3vec2,
                        num_vec
                    ]
        for sbj_cur in normal_dict_tmp.keys():
            [
                avg_vec,  std_vec,  avg_vec2,  std_vec2,
                avg_2vec, std_2vec, avg_2vec2, std_2vec2,
                avg_3vec, std_3vec, avg_3vec2, std_3vec2,
                num_vec
            ] = normal_dict_tmp[sbj_cur]
            if num_vec.bool().any() or sbj_cur not in normal_dict.keys():
                avg_vec = avg_vec / num_vec
                std_vec = std_vec / num_vec
                std_avg_vec = ((avg_vec2 / num_vec - avg_vec * avg_vec) ** 0.5) / avg_vec
                std_std_vec = ((std_vec2 / num_vec - std_vec * std_vec) ** 0.5) / std_vec
                avg_2vec = avg_2vec / num_vec
                std_2vec = std_2vec / num_vec
                std_2avg_vec = ((avg_2vec2 / num_vec - avg_2vec * avg_2vec) ** 0.5) / avg_2vec
                std_2std_vec = ((std_2vec2 / num_vec - std_2vec * std_2vec) ** 0.5) / std_2vec
                avg_3vec = avg_3vec / num_vec
                std_3vec = std_3vec / num_vec
                std_3avg_vec = ((avg_3vec2 / num_vec - avg_3vec * avg_3vec) ** 0.5) / avg_3vec
                std_3std_vec = ((std_3vec2 / num_vec - std_3vec * std_3vec) ** 0.5) / std_3vec
                normal_dict[sbj_cur] = (
                    avg_vec,  std_vec,  std_avg_vec,  std_std_vec,
                    avg_2vec, std_2vec, std_2avg_vec, std_2std_vec,
                    avg_3vec, std_3vec, std_3avg_vec, std_3std_vec
                                        )
    return normal_dict


def generate_new_scan(
        dat_flr='HS_Data',
        sbj_nme='',
        scn_idx=float('nan'),
        slc_idx=float('nan'),
        fin_sze=(),
        msk_prb=0,
        pad_flg=False,
        crp_flg=True,
        grd_flg=True,
        ms__flg=False,
        tst_flg=False,
        ref_flg=False,
        nrm_mat=torch.tensor([]),
        qt2_flg=False,
        qt1_flg=False,
        t2w_flg=False,
        t2flair_flg=False,
        pd_flg=False,
        cuda_flg=False,
        t2w_t_echo=90,
        flair_t_echo=80,
        flair_t_inv=2372,
        flair_fa=150,
        flair_etl=16,
        pe_ax=1,
        sever_per_arr=(0, 6, 9, 12, 15, 18, 21, 25, 30),
        seed=-1,
        change_idc=np.array([]),
        les_num=1,
        les_prob=0.5,
        les_size=10,
        lyr_size=3,
        roi_factor=1
):
    """
    Generate random WM-lesioned brain scan

    :param dat_flr: input folder name
    :param sbj_nme: subject to load (if this is '' then subject is drawn randomly)
    :param scn_idx: scan index to choose (if this is nan then subject is drawn randomly)
    :param slc_idx: slice index to choose (if this is nan then subject is drawn randomly)
    :param fin_sze: [tuple of the form (height, width)] determine output final size
    :param msk_prb: probability of masking output (effictively perform skull stripping)
    :param pad_flg: whether to stretch or pad with zeroes
    :param crp_flg: whether to crop image according to brain mask
    :param grd_flg: whether to return grid with th image
    :param ms__flg: whether to use MS patients dataset (source for data: http://lit.fe.uni-lj.si/tools)
    :param tst_flg: whether to use test images from trial
    :param ref_flg: if tst_flg then load also reference images
    :param nrm_mat: normalization parameters
    :param qt2_flg: whether to return qT2 map
    :param qt1_flg: whether to return qT1 map
    :param t2w_flg: whether to return T2w image
    :param t2flair_flg: whether to return T2 FLAIR image
    :param pd_flg: whether to return proton-density map
    :param cuda_flg: whether to use GPU
    :param t2w_t_echo: echo time for T2 weighted image
    :param flair_t_echo: echo time for T2 FLAIR image
    :param flair_t_inv: inversion time for T2 FLAIR image
    :param flair_fa: flip-angle time for T2 FLAIR image
    :param flair_etl: echo train length for T2 FLAIR image
    :param pe_ax: phase encoding axis for T2 FLAIR image
    :param sever_per_arr: array of percentages for pathological changes
    :param seed: used for randomization of lesions, -1 indicates not using seed
    :param change_idc: used to determine severity level
    :param les_num: number of lesions
    :param les_prob: probability of being lesioned
    :param les_size: [mm] approximate diameter of lesion
    :param lyr_size: [mm] approximate radius of layers between lesioned and healthy tissue
    :param roi_factor: factor by which lesion maximum size is larger then its minimum size
    :return: list containing lesioned MR images according to input flags

    Created by Chen Solomon, 04/09/2020
    """
    # optional: load MS data from data set
    if ms__flg:
        bnd_box_lst, output_images = load_rnd_ms_data(
            dat_flr=dat_flr,
            sbj_nme=sbj_nme,
            slc_idx=slc_idx,
            fin_sze=fin_sze,
            t2w_flg=t2w_flg,
            t2flair_flg=t2flair_flg,
            cuda_flg=cuda_flg)
    else:
        bnd_box_lst, output_images = synt_rnd_ms_data(
            dat_flr=dat_flr,
            sbj_nme=sbj_nme,
            tst_flg=tst_flg,
            ref_flg=ref_flg,
            scn_idx=scn_idx,
            slc_idx=slc_idx,
            qt2_flg=qt2_flg,
            qt1_flg=qt1_flg,
            t2w_flg=t2w_flg,
            t2flair_flg=t2flair_flg,
            pd_flg=pd_flg,
            cuda_flg=cuda_flg,
            t2w_t_echo=t2w_t_echo,
            flair_t_echo=flair_t_echo,
            flair_t_inv=flair_t_inv,
            flair_fa=flair_fa,
            flair_etl=flair_etl,
            pe_ax=pe_ax,
            sever_per_arr=sever_per_arr,
            seed=seed,
            change_idc=change_idc,
            les_num=les_num,
            les_prob=les_prob,
            les_size=les_size,
            lyr_size=lyr_size,
            roi_factor=roi_factor)
    # Resize if needed
    if len(fin_sze) != 0:
        if crp_flg:
            # Crop brain area from image - find bound box of brain
            if ms__flg:
                brn_msk = output_images[1]  # brain msk with skull stripping
            else:
                qt2 = output_images[2]
                pd = output_images[6]
                brn_msk = (qt2 > 0) * (pd > 0)
            crop_x, crop_y, crop_w, crop_h = find_boundbox(brn_msk, sqr_flg=True)
            img_shp = torch.Size((crop_w, crop_h))
        else:
            # Do not crop brain
            img_shp = output_images[0].shape
        rsz_dim, pad_dim = get_resize(fin_sze, pad_flg, img_shp)
        zero_pad = torch.nn.ZeroPad2d(pad_dim)
        if grd_flg:
            # Get mesh grid for image
            org_shp = output_images[0].shape
            y_grd, x_grd = get_padded_grid(
                org_shp,
                rsz_dim,
                fin_sze,
                crop_y1=crop_y,
                crop_y2=crop_y + crop_h,
                crop_x1=crop_x,
                crop_x2=crop_x + crop_w,
                pad_dim=pad_dim,
                cuda_flg=cuda_flg)
        else:
            # No grid
            y_grd = torch.tensor([])
            x_grd = torch.tensor([])
        # Whether to mask output
        msk_flg = np.random.binomial(1, msk_prb)
        # Loop over images
        for img_idx in range(len(output_images)):
            # Loop over images: mask, crop, interpolate, and pad with zeros
            image = output_images[img_idx]
            if image.nelement() != 0:
                if crp_flg:
                    image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                if \
                        isinstance(image, torch.ByteTensor) \
                        or isinstance(image, torch.cuda.ByteTensor)\
                        or isinstance(image, torch.BoolTensor) \
                        or isinstance(image, torch.cuda.BoolTensor):
                    image = image.float()
                image = image.unsqueeze(0).unsqueeze(0)
                image = torch.nn.functional.interpolate(image, size=rsz_dim).squeeze()
                image = zero_pad(image)
                # Mask image
                if msk_flg and img_idx > 1:
                    image = image * output_images[1]
                output_images[img_idx] = image
        # Loop over images
        for img_idx in range(len(output_images)):
            # Loop over images: normalize voxel values
            image = output_images[img_idx]
            if img_idx == 0 or img_idx == 1:
                img_ten = image
            elif image.nelement() != 0:
                # Handle normalization factors
                if len(nrm_mat) != 0:
                    nrm_vec = nrm_mat[int(ms__flg), :, img_idx - 2]
                else:
                    nrm_vec = torch.tensor([])
                # Add channels and possibly normalize
                brn_msk_1 = output_images[1]  # use freesurfer segmentation
                # brn_msk_1 = (output_images[2] > 0) * (output_images[-1] > 0)   # use qT2 & PDm for segmentation
                # erd_msk = 15  # Pixels for mask closure. (Hard coded - really does depend on fin_sze)
                # brn_msk_1 = morph_dilate_erode_2d(brn_msk_1, erd_msk, erd_flg=True, cuda_flg=cuda_flg)
                # brn_msk_1 = morph_dilate_erode_2d(brn_msk_1, erd_msk, erd_flg=False, cuda_flg=cuda_flg)
                img_ten = norm_map(image, brn_msk_1, grd_flg=grd_flg, x_grd=x_grd, y_grd=y_grd, nrm_fct=nrm_vec)
            else:
                img_ten = image
            output_images[img_idx] = img_ten
        # Resize boundbox
        box_num = len(bnd_box_lst)
        for box_idx in range(box_num):
            bbox_x, bbox_y, bbox_w, bbox_h = bnd_box_lst[box_idx]
            if crp_flg:
                bbox_x = bbox_x - crop_x
                bbox_y = bbox_y - crop_y
                bbox_w = max(min(bbox_w, crop_w + crop_x - bbox_x), 0)
                bbox_h = max(min(bbox_h, crop_h + crop_y - bbox_y), 0)
            bbox_x = round(bbox_x * rsz_dim[1] / crop_w)
            bbox_y = round(bbox_y * rsz_dim[0] / crop_h)
            bbox_w = round(bbox_w * rsz_dim[1] / crop_w)
            bbox_h = round(bbox_h * rsz_dim[0] / crop_h)
            if len(pad_dim) == 4:
                bbox_x += pad_dim[0]
                bbox_y += pad_dim[2]
            bnd_box_lst[box_idx] = (bbox_x, bbox_y, bbox_w, bbox_h)
    return bnd_box_lst, output_images


def get_img_stt(img, msk=torch.tensor([]), msk_sze=float('nan')):
    """
    estimate images 1st and 2nd order moments (mean & standard-deviation)

    :param img:     input image
    :param msk:     mask of relevant pixels
    :param msk_sze: mask size
    :return:        mean & stdev
    """
    if len(msk) == 0:
        msk = torch.tensor(1)
        msk_sze = img.nelement()
    elif math.isnan(msk_sze):
        msk_sze = msk.sum()
    if msk_sze == 0:
        return 0, 0, 0, 0, 0
    img_lin_sum = (msk * img).sum()
    med = img.masked_select(msk > 0).median()
    img_sqr_sum = ((msk * img) ** 2).sum()
    avg = (img_lin_sum / msk_sze).cpu()
    mask_upper = (img > med).double() * msk
    mask_lower = (img < med).double() * msk
    upper_std = (((mask_upper * (img - med)) ** 2).sum() / mask_upper.sum()) ** 0.5
    lower_std = (((mask_lower * (img - med)) ** 2).sum() / mask_lower.sum()) ** 0.5
    std = \
        (
                (
                        img_sqr_sum / (msk_sze - 1) - img_lin_sum ** 2 / (msk_sze * (msk_sze - 1))
                )
                ** 0.5
        ).cpu()
    return avg, std, upper_std, lower_std, med


def get_lbl_wgt(val, labels):
    """
    return labels weights for value

    :param val:     value in question
    :param labels:  labels - assumed to sorted from small to large
    :return:        vectors of weight per label
    """
    weights = torch.zeros_like(labels)
    idx = (val > labels).sum()
    # calculate weights linearly
    weights[idx - 1] = (val - labels[idx - 1]).float() / (labels[idx] - labels[idx - 1]).float()
    weights[idx] = (labels[idx] - val).float() / (labels[idx] - labels[idx - 1]).float()
    return weights


def get_msk_lbl_wgt(msk, labels):
    """
    return labels weights for value

    :param msk:     valued mask in question
    :param labels:  labels - assumed to sorted from small to large
    :return:        vectors of weight per label for image
    """
    weights = torch.zeros_like(labels)
    return weights


# Script for tests:
if __name__ == '__main__':

    # Script for tests:

    # Ex 1 (Updated version, last edited 28/08/2020 - folder HS_Data)

    # # flags:            - NOT USED (not optimized / cause errors)
    # cuda_flag = False
    # numpy_flag = False

    # Measure Runtime (for tests)
    num_trials = 1
    start_t = time.time()

    # Scan parameters:  - Hyper Parameters
    #                   - some of them can be randomly modified to get varying contrast)
    TE_T2w = 90
    TE_T2FLAIR = 80
    TI_T2FLAIR = 2372
    ETL_T2FLAIR = 16  # NOT to be changed
    FA_T2FLAIR = 150
    TE_T1w = 2.61
    TI_T1w = 100
    ETL_T1w = 1  # NOT to be changed
    FA_T1w = 180
    TR_T1w = 1750
    PE_AX = 1  # NOT to be changed

    # Directories
    train_dir = 'HS_Data'
    # Subject_name = '5_ChSo_2020_09_24'
    Subject_name = ''  # causes randomization of subject

    # Dimensions of net input
    final_size = (256, 256)
    Pad_flag = True

    # Generate scans with lesions
    # for ii in range(max(1, num_trials)):  # Loop used for testing runtime  # for multiple trials
    Bnd_Box_Lst, (les_ROI, Brn_msk, qT2, qT1, T2w, T2FLAIR, PDm) = generate_new_scan(  # the relevant function
        dat_flr=train_dir,
        sbj_nme=Subject_name,
        fin_sze=final_size,
        pad_flg=Pad_flag,
        qt2_flg=True,
        t2flair_flg=True,
        t2w_t_echo=TE_T2w,  # Scan parameters (predefined)
        flair_t_echo=TE_T2FLAIR,
        flair_t_inv=TI_T2FLAIR,
        flair_fa=FA_T2FLAIR,
        flair_etl=ETL_T2FLAIR,
        pe_ax=PE_AX)

    # Measure time
    end_t = time.time()
    run_t = (end_t - start_t) / num_trials
    print("---Runtime: %s seconds---" % "{:.4}".format(run_t))

    # Show Result
    # if num_trials >= 1:  # for multiple trials
    T2FLAIR = T2FLAIR[:, :, 1].squeeze()
    qT2 = qT2[:, :, 1].squeeze()
    # if cuda_flag:  # irrelevant if cuda_flg is False or not defined
    #     T2FLAIR = T2FLAIR.cpu().numpy()
    #     les_ROI = les_ROI.cpu().numpy()
    #     qT2 = qT2.cpu().numpy()
    plt.figure(1)
    plt.imshow(T2FLAIR, cmap=plt.cm.gist_gray)
    plt.figure(2)
    plt.imshow(les_ROI, cmap=plt.cm.gist_gray)
    plt.figure(3)
    plt.imshow(qT2, cmap=plt.cm.gist_gray)
    plt.show()

    # # End of un-commented test script

    # Template: Measuring runtime

    # start_t = time.time()
    # generate_new_scan()  # Or whatever other script / function
    # end_t = time.time()
    # run_t = end_t - start_t
    # print("---Runtime: %s seconds---" % "{:.2}".format(run_t))

    # Ex. 3 (Old version, Edited 25/08/2020 - folder HS_Data_old)

    # a)

    # >>> Map_dic = scipy.io.loadmat('/home/noambe/Public/FreeSurfer/Python/qMRI_Augmentation/HS_Data/Sbj_Map_struc')
    # >>> print(Map_dic['Sbj_Map_struc'][0, 0]['qT2'][0, 0][:, :, 1].shape)
    # [128, 128]

    # b)

    # start_t = time.time()
    # Load .mat file
    # Map_dic = scipy.io.loadmat('/home/noambe/Public/FreeSurfer/Python/qMRI_Augmentation/HS_Data/Sbj_Map_struc')
    # Load matlab structure from .mat file
    # Sbj_Arr = Map_dic['Sbj_Map_struc'].flatten()
    # Choose subject
    # Sbj_Num = len(Sbj_Arr)
    # Sbj_idx = np.random.choice(Sbj_Num)
    # Scn_Arr = Sbj_Arr[Sbj_idx]
    # qT2_Arr = Scn_Arr['qT2'].flatten()
    # Scn_Num = len(qT2_Arr)
    # Scn_idx = np.random.choice(Scn_Num)
    # qT2_Vol = qT2_Arr[Scn_idx]
    # if qT2_Vol.ndim == 2:
    #     qT2_Slc = qT2_Vol[:, :]
    #     PDm_Slc = Scn_Arr['PD'].flatten()[Scn_idx][:, :]
    #     LUT_Slc = Scn_Arr['LUT'].flatten()[Scn_idx][:, :]
    # elif qT2_Vol.ndim == 3:
    #     Slc_Num = qT2_Vol.shape[2]
    #     Slc_idx = np.random.choice(Slc_Num)
    #     qT2_Slc = qT2_Vol[:, :, Slc_idx]
    #     PDm_Slc = Scn_Arr['PD'].flatten()[Scn_idx][:, :, Slc_idx]
    #     LUT_Slc = Scn_Arr['LUT'].flatten()[Scn_idx][:, :, Slc_idx]
    # TE = 80
    # T2w = mri_t2w(PDm_Slc, qT2_Slc, TE)
    # end_t = time.time()
    # run_t = end_t - start_t
    # print("---Runtime: %s seconds---" % "{:.2}".format(run_t))
    # plt.imshow(T2w)
    # plt.show()

# # moving MS patients files (source for data: http://lit.fe.uni-lj.si/tools):
# # 1. use WinRAR to extract the folder (You might need to download it)
# # 2. use WinRAR to extract the folders of the patients by right clicking the files and choosing 'extract here'
# #  run this code for moving patients files
# ipt_pth = 'MS_Data\\MS_Data'
# for idx_ptn in range(1, 31):
#     idx_str = ''
#     if idx_ptn < 10:
#         idx_str += '0'
#     idx_str += str(idx_ptn)
#     pat_nme = 'patient' + idx_str
#     opt_pth = os.path.join(ipt_pth, pat_nme)
#     os.mkdir(opt_pth)
#     # print(opt_pth)
#     ipt_nme = ('_brainmask.nii', )
#     ipt_nme += ('_consensus_gt.nii', )
#     ipt_nme += ('_FLAIR(1).nii', )
#     ipt_nme += ('_T2W.nii', )
#     ipt_nme += ('_FLAIR.nii', )
#     ipt_nme += ('_T1W(1).nii', )
#     ipt_nme += ('_T1W.nii', )
#     ipt_nme += ('_T1WKS(1).nii', )
#     ipt_nme += ('_T1WKS.nii', )
#     ipt_nme += ('_T2W(1).nii', )
#
#     opt_nme = ('_brainmask.nii', )
#     opt_nme += ('_consensus_gt.nii', )
#     opt_nme += ('_FLAIR_raw.nii', )
#     opt_nme += ('_T2W.nii', )
#     opt_nme += ('_FLAIR.nii', )
#     opt_nme += ('_T1W_raw.nii', )
#     opt_nme += ('_T1W.nii', )
#     opt_nme += ('_T1WKS_raw.nii', )
#     opt_nme += ('_T1WKS.nii', )
#     opt_nme += ('_T2W_raw.nii', )
#     for idx_img in range (0, len(ipt_nme)):
#         ipt_fle = ipt_nme[idx_img]
#         opt_fle = opt_nme[idx_img]
#         old_nme = pat_nme + ipt_fle
#         new_nme = pat_nme + opt_fle
#         old_pth = os.path.join(ipt_pth, old_nme)
#         new_pth = os.path.join(opt_pth, new_nme)
#         # print(os.path.isfile(old_pth))
#         os.rename(old_pth, new_pth)
