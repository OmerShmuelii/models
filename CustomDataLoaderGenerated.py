
import numpy as np

import torch

from torch.utils.data import Dataset as BaseDataset

import os

import albumentations as albu

import scipy

from generate_new_scanfixmulti import  generate_new_scan, get_qmri_data_list, get_img_stt , NormDict  # ChenS' updated script 20210328

# Measure Runtime (for tests)
num_trials = 40
#start_t = time.time()

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

def get_training_augmentation():
    start=0
    train_transform = [

        albu.RandomSizedBBoxSafeCrop(256,256, always_apply=True, p=0.85)
        #albu.tra
        ]
    # train_transform = [
    #
    #     albu.Crop(x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=False, p=0.85),
    #
    #     #albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    #
    #     albu.PadIfNeeded(min_height=512, min_width=512, always_apply=False, border_mode=0),
    #    # albu.RandomCrop(height=512, width=512, always_apply=False),
    #
    #     albu.IAAAdditiveGaussianNoise(p=0.2),
    #     albu.IAAPerspective(p=0.5),
    #
    #     albu.OneOf(
    #         [
    #             albu.CLAHE(p=1),
    #             albu.RandomBrightnessContrast(p=1),
    #             albu.RandomGamma(p=1),
    #         ],
    #         p=0.9,
    #     ),
    #
    #     albu.OneOf(
    #         [
    #             albu.IAASharpen(p=1),
    #             albu.Blur(blur_limit=3, p=1),
    #             albu.MotionBlur(blur_limit=3, p=1),
    #         ],
    #         p=0.9,
    #     ),
    #
    #     albu.OneOf(
    #         [
    #             albu.RandomBrightnessContrast(p=1),
    #             albu.HueSaturationValue(p=1),
    #         ],
    #         p=0.9,
    #     ),
    # ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(512, 512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, wide , high ):
        self.n_holes = n_holes
        self.wide = wide
        self.high = high

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h,w = img.shape[0],img.shape[1]
        # = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            wider = np.random.randint(0, self.wide)
            highr = np.random.randint(0, self.high)
            intensityr=np.random.randint(0, 10)
            y1 = np.clip(y - highr // 2, 0, h)
            y2 = np.clip(y + highr // 2, 0, h)
            x1 = np.clip(x - wider // 2, 0, w)
            x2 = np.clip(x + wider // 2, 0, w)

            mask[y1: y2, x1: x2] *= 1+intensityr/10

        mask = torch.from_numpy(mask)

        mask = mask.expand_as(img[:,:,1].squeeze())
        return mask



class Dataset(BaseDataset):

    def __init__(
            self,
            train_dir,


            split='train',

            augmentation=None,
            preprocessing=None,
            batch_size=32,
            shuffle=True,

            finalSize=448,
            prob=0.5,
            gpu=True,
            ms__flg=False,

            train_dirr='',
            multi=1,
            TST_flag=False,
            Erd_Thc=0,
            ref_flg=False,
            istrain=True,
            sever_per_arr=(0, 21, 25, 30) # #(0, 21, 25, 30), #(0, 6, 59, 12), #(0, 6, 9, 12, 15, 18, 21, 25, 30), #(0, 27, 36, 45, 54, 63, 75, 83, 90),
    ):
        self.isTrain=istrain
        self.ref_flg=ref_flg
        images_dir = get_qmri_data_list(train_dir,ms__flg=ms__flg,tst_flg=TST_flag,ref_flg=self.ref_flg, erd_thc=Erd_Thc)
        scipy.io.savemat(train_dir +os.path.join('samples_list') + '.mat', {'samples_list': images_dir})
        self.sever_per_arr=sever_per_arr
        self.multi=multi
        self.train_dirr=train_dirr
        if train_dirr=='':
           self.train_dirr=train_dir
        self.MS__flag=ms__flg
        self.TST_flag=TST_flag
        self.gpu=gpu
        self.split=split
        self.finalSize=finalSize
        self.batch_size=batch_size
        self.imagepair = images_dir
        self.prob=prob
        self.shuffle=shuffle
        if shuffle==True:
            self.indexpointer=np.random.permutation(len(self.imagepair))
        else:
            self.indexpointer = np.linspace(0, 1, len(self.imagepair))



        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.ids=self.imagepair

        #  #                         |_qT2___|_qT1_____|_T2w______|_T2FLAIR__|_PDm_______|______|____________
        self.Avg_STD_mat =  torch.tensor([
                                        [[122.5048, 1021.6532, 4522.2482, -1, 12245.7118],   # | Avg | Synth 3144.8364
                                         [123.7552,  874.5087, 2331.9195, -1 ,  3660.9177]],  # | STD |______  1240.0415
                                        [[000.0000,  000.0000,  421.8111, -1,     0.0000],   # | Avg | Orig   101.2803
                                         [000.0000,  000.0000,  203.7409,   -1,     0.0000]]   # | STD |______  35.0940
                                        ])

        Pad_flag = True
        Crp_flag = True
        Grd_flag = True

        self.cutoutobj1 = Cutout(7, 10, 6)
        self.cutoutobj2 = Cutout(7, 6, 10)
        self.cutoutobj3 = Cutout(7, 7, 7)

        if self.multi<0:
            num_les= 1 + np.random.randint(10)
        else:
            num_les= self.multi
        self.Dic_flag=False #  and not self.MS__flag
        if self.Dic_flag:
            print ("start dict")
            if TST_flag==False:
                images_dirForDicr = get_qmri_data_list(train_dir, ms__flg=ms__flg, tst_flg=TST_flag, erd_thc=7)
            else:
                images_dirForDicr=images_dir
            self.dict_images = NormDict(
                self.train_dirr + '/' + self.split,
                images_dirForDicr,
                sbj_nme='',
                fin_sze=final_size,
                pad_flg=Pad_flag,
                crp_flg=Crp_flag,
                grd_flg=Grd_flag,
                ms__flg=self.MS__flag,
                tst_flg=TST_flag,
                qt2_flg=True,
                qt1_flg=False,
                t2w_flg=False,
                t2flair_flg=True,
                pd_flg=True,
                cuda_flg=True,
                t2w_t_echo=TE_T2w,
                flair_t_echo=TE_T2FLAIR,
                flair_t_inv=TI_T2FLAIR,
                flair_fa=FA_T2FLAIR,
                flair_etl=ETL_T2FLAIR,
                pe_ax=PE_AX,
                sever_per_arr=self.sever_per_arr,
                les_num=num_les,
                les_prob=self.prob,
                les_size=10,
                lod_flg=True,
                sve_flg=False,
        )
            print("End dict")

    def __getitem__(self, i):
        LES_SIZE=5 + np.random.randint(5)
        # bisfirst=True
        k=i
        if self.multi<0:
            num_les= 1 + np.random.randint(-self.multi)
        else:
            num_les= self.multi

        # read data
        Sbj_Avg_STD_mat = self.Avg_STD_mat.clone()
        if self.Dic_flag:
             #idx = i * (MS__flag or TST_flag)
            Cur_Sbj = (self.ids[i][0], self.ids[i][1])
            MS_Syn_idx = int(self.MS__flag)
            Sbj_Avg_STD_mat[MS_Syn_idx, 0], Sbj_Avg_STD_mat[MS_Syn_idx, 1] = self.dict_images.dict.get(
                 Cur_Sbj,
                 (self.Avg_STD_mat[MS_Syn_idx, 0], self.Avg_STD_mat[MS_Syn_idx, 1])
             )[0:2]
            if self.MS__flag and False:
                print(Sbj_Avg_STD_mat[MS_Syn_idx, 0])
                print(Sbj_Avg_STD_mat[MS_Syn_idx, 1])
            varaug=np.random.uniform(low=0.9, high=1.1)
            meanaug = np.random.uniform(low=0.9, high=1.1)
            Sbj_Avg_STD_mat[MS_Syn_idx, 0]*=varaug
            Sbj_Avg_STD_mat[MS_Syn_idx, 1] *= meanaug
        Pad_flag = True
        Crp_flag = True
        Grd_flag = True
        if self.TST_flag:
            Bnd_Box_Lst, (les_ROI, Brn_msk, qT2, qT1, T2w, T2FLAIR, PDm) = generate_new_scan(  # the relevant function
                dat_flr=self.train_dirr + '/' + self.split,
                fin_sze=final_size,
                sbj_nme=self.ids[i][0],
                scn_idx=self.ids[i][1],
                slc_idx=self.ids[i][2],
                ref_flg=self.ref_flg,
                pad_flg=Pad_flag,
                crp_flg=Crp_flag,
                grd_flg=Grd_flag,
                ms__flg=self.MS__flag,
                tst_flg=self.TST_flag,
                nrm_mat=Sbj_Avg_STD_mat,
                msk_prb=0,
                qt2_flg=True,
                qt1_flg=True,
                t2w_flg=True,
                t2flair_flg=True,
                cuda_flg=self.gpu,
                t2w_t_echo=TE_T2w,  # Scan parameters (predefined)
                flair_t_echo=TE_T2FLAIR,
                flair_t_inv=TI_T2FLAIR,
                flair_fa=FA_T2FLAIR,
                flair_etl=ETL_T2FLAIR,
                les_size=LES_SIZE,
                pd_flg=True,
                pe_ax=PE_AX)
        else:
            if self.MS__flag:
                Bnd_Box_Lst, (les_ROI, Brn_msk, qT2, qT1, T2w, T2FLAIR, PDm) = generate_new_scan(  # the relevant function
                    dat_flr=self.train_dirr + '/' + self.split,
                    sbj_nme=self.ids[i][0],
                    scn_idx=self.ids[i][1],
                    slc_idx=self.ids[i][2],
                    fin_sze=final_size,
                    pad_flg=Pad_flag,
                    crp_flg=Crp_flag,
                    grd_flg=Grd_flag,
                    nrm_mat=Sbj_Avg_STD_mat,
                    msk_prb=0,
                    qt2_flg=True,
                    qt1_flg=True,
                    t2w_flg=True,
                    t2flair_flg=True,
                    cuda_flg=self.gpu, #self.gpu,
                    ms__flg=self.MS__flag,
                    t2w_t_echo=TE_T2w,  # Scan parameters (predefined)
                    flair_t_echo=TE_T2FLAIR,
                    flair_t_inv=TI_T2FLAIR,
                    flair_fa=FA_T2FLAIR,
                    flair_etl=ETL_T2FLAIR,
                    les_prob=0,
                    pd_flg=True,
                    pe_ax=PE_AX)
            else:
                Bnd_Box_Lst, (les_ROI, Brn_msk, qT2, qT1, T2w, T2FLAIR, PDm) = generate_new_scan(  # the relevant function
                    dat_flr=self.train_dirr + '/' + self.split,
                    sbj_nme=self.ids[i][0],
                    scn_idx=self.ids[i][1],
                    slc_idx=self.ids[i][2],
                    fin_sze=final_size,
                    pad_flg=Pad_flag,
                    crp_flg=Crp_flag,
                    grd_flg=Grd_flag,
                    ms__flg=self.MS__flag,
                    nrm_mat=Sbj_Avg_STD_mat,
                    msk_prb=0,
                    qt2_flg=True,
                    qt1_flg=True,
                    t2w_flg=True,
                    t2flair_flg=True,
                    pd_flg=True,
                    sever_per_arr= self.sever_per_arr, #(0, 21, 25, 30), # #(0, 21, 25, 30), #(0, 6, 59, 12), #(0, 6, 9, 12, 15, 18, 21, 25, 30), #(0, 27, 36, 45, 54, 63, 75, 83, 90),
                    les_num=num_les,
                    les_prob=self.prob,
                    cuda_flg=self.gpu,
                    t2w_t_echo=TE_T2w,  # Scan parameters (predefined)
                    flair_t_echo=TE_T2FLAIR,
                    flair_t_inv=TI_T2FLAIR,
                    flair_fa=FA_T2FLAIR,
                    flair_etl=ETL_T2FLAIR,
                    les_size=LES_SIZE,
                    pe_ax=PE_AX)


        numbb=len(Bnd_Box_Lst)
        while len(Bnd_Box_Lst) < 100:
            Bnd_Box_Lst.append((-1,-1,-1,-1))
        Bnd_Box_Lst=torch.FloatTensor(Bnd_Box_Lst)


        if i % 4 != 0 and self.isTrain:
            cutoutmask = (self.cutoutobj1(T2FLAIR) * self.cutoutobj2(T2FLAIR) * self.cutoutobj3(T2FLAIR)).cuda()

            T2FLAIR[:,:,1] *= cutoutmask  # .astype(np.uint8)

        return T2FLAIR ,les_ROI.float(), Bnd_Box_Lst,numbb

    def __len__(self):
        return len(self.ids)