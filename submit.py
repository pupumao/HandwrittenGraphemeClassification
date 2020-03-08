import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from tqdm import tqdm
import torch
from lib.core.model.semodel.SeResnet import se_resnet50


HEIGHT = 137
WIDTH = 236
SIZE = 128


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin, ymax-ymin
    l = max(lx, ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img, (size, size))


x_tot, x2_tot = [], []

result = []
data_root = '/data/old_machine/dataset/bengaliai-cv'

model_path = './model/epoch_55_val_loss0.097817.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HWmodel = se_resnet50()

HWmodel.load_state_dict(torch.load(
    model_path, map_location=device), strict=False)
HWmodel.to(device)
HWmodel.eval()

TEST = [os.path.join(data_root, item) for item in ['test_image_data_0.parquet',
                                                   'test_image_data_1.parquet',
                                                   'test_image_data_2.parquet',
                                                   'test_image_data_3.parquet']]


components = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
# components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']

target = []  # model predictions placeholder
row_id = []  # row_id place holder

id_ = 0

for fname in TEST:
    print('will proc {}'.format(fname))
    df = pd.read_parquet(fname)

    # df.set_index('image_id', inplace=True)
    # the input is inverted
    data = 255 - df.iloc[:,
                         1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    for idx in tqdm(range(len(df))):
        cur_result = {}
        name = df.iloc[idx, 0]
        # normalize each image by its max val
        img = (data[idx]*(255.0/data[idx].max())).astype(np.uint8)
        img = crop_resize(img)

        x_tot.append((img / 255.0).mean())
        x2_tot.append(((img / 255.0) ** 2).mean())
        # img = cv2.imencode('.png', img)[1]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = np.transpose(img, axes=[2, 0, 1])

        img = img.astype(np.float32)

        img = img / 255.
        img = np.expand_dims(img, axis=0)

        images = torch.from_numpy(img)
        images = images.to(device)
        print(images.shape)

        logit1, logit2, logit3 = HWmodel(images)
        res1 = torch.softmax(logit1, 1)
        res2 = torch.softmax(logit2, 1)
        res3 = torch.softmax(logit3, 1)

        res1 = res1.cpu().detach().numpy()[0, :]
        res2 = res2.cpu().detach().numpy()[0, :]
        res3 = res3.cpu().detach().numpy()[0, :]

        pred_dict = {
            'grapheme_root': res1,
            'vowel_diacritic': res2,
            'consonant_diacritic': res3
        }
        print('preds: {}...{}...{}'.format(res1.shape, res2.shape, res3.shape))
        for comp in components:
            id_sample = 'Test_{}_{}'.format(id_, comp)
            row_id.append(id_sample)
            target.append(np.argmax(pred_dict[comp], axis=0))
        id_ += 1


print('pred_dict size: {}, {}, {}'.format(
    len(pred_dict['grapheme_root']),
    len(pred_dict['vowel_diacritic']),
    len(pred_dict['consonant_diacritic'])))

# make predict
print('row_id: {}'.format(row_id))
print('target: {}'.format(target))


df_sample = pd.DataFrame(
    {
        'row_id': row_id,
        'target': target
    },
    columns=['row_id', 'target']
)
df_sample.to_csv('submission.csv', index=False)
df_sample.head()
