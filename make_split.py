
import os
import pandas as pd

from train_config import config as cfg

data_dir= os.path.join(cfg.DATA.root_path, 'train_imgs')

train_df_ = pd.read_csv(os.path.join(cfg.DATA.root_path,'train.csv'))

image_list=os.listdir(data_dir)


ratio=0.9
length=len(image_list)

train_set=image_list[:int(ratio*length)]
val_set=image_list[int(ratio*length):]


print(train_df_.head())


index_map=list(train_df_['image_id'])

grapheme_root_map=train_df_['grapheme_root']
vowel_diacritic_map=train_df_['vowel_diacritic']
consonant_diacritic_map=train_df_['consonant_diacritic']


grapheme_root_class=len(set(list(grapheme_root_map)))
vowel_diacritic_class=len(set(list(vowel_diacritic_map)))
consonant_diacritic_class=len(set(list(consonant_diacritic_map)))
print('grapheme_root_map class has %d'%grapheme_root_class)

print('vowel_diacritic_map class has %d'%vowel_diacritic_class)

print('consonant_diacritic_map class has %d'%consonant_diacritic_class)




###

#
# def static_data_distribution(map,classes):
#     map=list(map)
#     cnt_map={}
#     for i in range(len(map)):
#         if map[i]in cnt_map:
#             cnt_map[map[i]]+=1
#         else:
#             cnt_map[map[i]] = 1
#
#     for k,v in cnt_map.items():
#         print(k,v)
#
# static_data_distribution(grapheme_root_map,grapheme_root_class)
# static_data_distribution(vowel_diacritic_map,vowel_diacritic_class)
# static_data_distribution(consonant_diacritic_map,consonant_diacritic_class)


train_f=open('train.txt','w')
for pic in train_set:

    image_id=pic.rsplit('.',1)[0]
    pic_path=os.path.join(data_dir,pic)

    cur_index=index_map.index(image_id)

    grapheme_root_label=grapheme_root_map[cur_index]
    vowel_diacritic_label = vowel_diacritic_map[cur_index]
    consonant_diacritic_label = consonant_diacritic_map[cur_index]

    tmp_str=pic_path+' '+str(grapheme_root_label)+" "+str(vowel_diacritic_label)+" "+str(consonant_diacritic_label)+'\n'
    train_f.write(tmp_str)
train_f.close()

val_f = open('val.txt', 'w')
for pic in val_set:
    image_id = pic.rsplit('.', 1)[0]
    pic_path = os.path.join(data_dir, pic)

    cur_index = index_map.index(image_id)

    grapheme_root_label = grapheme_root_map[cur_index]
    vowel_diacritic_label = vowel_diacritic_map[cur_index]
    consonant_diacritic_label = consonant_diacritic_map[cur_index]

    tmp_str = pic_path + ' ' + str(grapheme_root_label) + " " + str(vowel_diacritic_label) + " " + str(
        consonant_diacritic_label) + '\n'
    val_f.write(tmp_str)
val_f.close()