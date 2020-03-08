import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from tqdm import tqdm
import torch
import torch.utils.data
import gc

from lib.core.model.semodel.SeResnet import se_resnet50,se_resnext50_32x4d


HEIGHT = 137
WIDTH = 236
SIZE = 128
TEST_BATCH_SIZE=2

data_dir = '/media/lz/ssd_2/kaggle/data'

model_list=['./models/epoch_1_val_loss1.468146.pth']

class BengaliDatasetTest:
    def __init__(self, df):
        self.image_ids = df.image_id.values
        self.img_arr = df.iloc[:, 1:].values



    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image = self.img_arr[item, :]
        img_id = self.image_ids[item]

        image = 255-image.reshape(137, 236).astype(np.uint8)

        image=self.preprocess(image)


        return {
            "image": image,
            "image_id": img_id
        }


    def preprocess(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

        cv2.imshow('ss',img)
        cv2.waitKey(0)
        img = np.transpose(img, axes=[2, 0, 1])
        img = img/255.
        img = img.astype(np.float32)
        return img



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_with_model(model):
    g_pred, v_pred, c_pred = [], [], []
    img_ids_list = []


    TEST = [os.path.join(data_dir, 'test_image_data_0.parquet'),
            os.path.join(data_dir, 'test_image_data_1.parquet'),
            os.path.join(data_dir, 'test_image_data_2.parquet'),
            os.path.join(data_dir, 'test_image_data_3.parquet')
            ]

    for fname in TEST:
        df = pd.read_parquet(fname)

        dataset = BengaliDatasetTest(df=df)
        del df
        gc.collect()

        data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size= TEST_BATCH_SIZE,
                    shuffle=False,
                    num_workers=1
            )



        for bi, d in enumerate(data_loader):
            image = d["image"]
            img_id = d["image_id"]

            image = image.to(device, dtype=torch.float32)

            logit1, logit2, logit3 = model(image)
            g = torch.softmax(logit1, 1)
            v = torch.softmax(logit2, 1)
            c = torch.softmax(logit3, 1)


            for ii, imid in enumerate(img_id):
                g_pred.append(g[ii].cpu().detach().numpy())
                v_pred.append(v[ii].cpu().detach().numpy())
                c_pred.append(c[ii].cpu().detach().numpy())
                img_ids_list.append(imid)


    del data_loader
    del dataset
    gc.collect()

    return g_pred, v_pred, c_pred, img_ids_list




final_g_pred = []
final_v_pred = []
final_c_pred = []
final_img_ids = []

for i in range(len(model_list)):

    model = se_resnext50_32x4d()
    model.load_state_dict(torch.load(model_list[i], map_location=device))
    model.to(device)
    model.eval()
    g_pred, v_pred, c_pred, img_ids_list = predict_with_model(model)

    final_g_pred.append(g_pred)
    final_v_pred.append(v_pred)
    final_c_pred.append(c_pred)
    if i == 0:
        final_img_ids.extend(img_ids_list)


final_g = np.argmax(np.mean(np.array(final_g_pred), axis=0), axis=1)
final_v = np.argmax(np.mean(np.array(final_v_pred), axis=0), axis=1)
final_c = np.argmax(np.mean(np.array(final_c_pred), axis=0), axis=1)



predictions = []
for ii, imid in enumerate(final_img_ids):
    predictions.append((f"{imid}_grapheme_root", final_g[ii]))
    predictions.append((f"{imid}_vowel_diacritic", final_v[ii]))
    predictions.append((f"{imid}_consonant_diacritic", final_c[ii]))


sub = pd.DataFrame(predictions, columns=["row_id", "target"])
sub.to_csv("submission.csv", index=False)
