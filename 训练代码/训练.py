import json
import logging
from pathlib import Path
from collections import Counter
from functools import lru_cache

import numpy as np
from tqdm import tqdm
from PIL import Image
from rimo_storage.cache import disk_cache

import ml_danbooru


def ml_danbooru标签(image: str) -> dict[str, dict[str, float]]:
    tags = ml_danbooru.get_tags_from_image(Image.open(image), threshold=0.4, keep_ratio=True)
    return tags


@disk_cache(path='./_cache/源', serialize='pickle')
def 源(文件夹: Path, 全人: set):
    组 = {}
    for 文件 in 文件夹.iterdir():
        组.setdefault(文件.stem, []).append(文件.suffix)
    res = []
    for k, v in 组.items():
        真标签 = open(文件夹 / f'{k}.txt', encoding='utf8').read().split(', ')
        if '1girl' not in 真标签 or '1boy' in 真标签:
            continue
        人 = {*真标签} & 全人
        if len(人) != 1:
            continue
        人 = 人.pop()
        res.append((k, 人))
    return res


全人 = {*json.load(open('人.json'))}


测试文件夹 = ['data-0000', 'data-0020']
训练文件夹 = [
    'data-0001', 'data-0021', 'data-0031', 'data-0041', 'data-0051', 'data-0071',
    'data-0101', 'data-0121', 'data-0131', 'data-0141', 'data-0151', 'data-0171',
    'data-0201', 'data-0221', 'data-0231', 'data-0241', 'data-0251', 'data-0271',
    'data-0301', 'data-0321', 'data-0331', 'data-0341', 'data-0351', 'data-0371',
    'data-0401', 'data-0421', 'data-0431', 'data-0441', 'data-0451', 'data-0471',
    'data-0501', 'data-0521', 'data-0531', 'data-0541', 'data-0551', 'data-0571',
    'data-0601', 'data-0621', 'data-0631', 'data-0641', 'data-0651', 'data-0671',
    'data-0701', 'data-0721', 'data-0731', 'data-0741', 'data-0751', 'data-0771',
    'data-0801', 'data-0821', 'data-0831', 'data-0841', 'data-0851', 'data-0871',
]

for 文件夹 in 测试文件夹 + 训练文件夹:
    文件夹 = Path(文件夹)
    for k, _ in tqdm(源(文件夹, 全人), desc=f'打标{文件夹}'):
        if (文件夹 / f'{k}.json').exists():
            continue
        try:
            标签 = ml_danbooru标签(文件夹 / f'{k}.jpg')
            with open(文件夹 / f'{k}.json', 'w', encoding='utf8') as f:
                json.dump(标签, f)
        except Exception as e:
            logging.exception(f'{k}.jpg有问题！')


人标签 = ['animal_ears', 'animal_ear_fluff', 'pointy_ears', 'cat_ears', 'fox_ears', 'dog_ears', 'horse_ears', 'multiple_tails', 'rabbit_tail', 'cat_tail', 'fox_tail', 'dog_tail', 'demon_tail', 'blue_eyes', 'hair_between_eyes', 'purple_eyes', 'green_eyes', 'brown_eyes', 'red_eyes', 'closed_eyes', 'pink_eyes', 'yellow_eyes', 'aqua_eyes', 'grey_eyes', 'black_eyes', 'orange_eyes', 'white_eyes', 'glowing_eyes', 'hair_over_eyes', 'hair_over_shoulder', 'long_hair', 'short_hair', 'brown_hair', 'eyebrows_visible_through_hair', 'black_hair', 'blonde_hair', 'very_long_hair', 'blue_hair', 'purple_hair', 'silver_hair', 'white_hair', 'pink_hair', 'grey_hair', 'pubic_hair', 'medium_hair', 'multicolored_hair', 'red_hair', 'two-tone_hair', 'shiny_hair', 'streaked_hair', 'orange_hair', 'gradient_hair', 'aqua_hair', 'green_hair', 'light_purple_hair', 'light_brown_hair', 'tied_hair', 'straight_hair', 'asymmetrical_hair', 'spiked_hair', 'light_blue_hair', 'platinum_blonde_hair', 'eyebrows_behind_hair', 'colored_inner_hair', 'drill_hair', 'wavy_hair', 'low-tied_long_hair', 'antenna_hair', 'medium_breasts', 'large_breasts', 'small_breasts', 'huge_breasts', 'gigantic_breasts', 'flat_chest', 'ponytail', 'high_ponytail', 'ribbon', 'two_side_up', 'twintails', 'short_twintails', 'low_twintails', 'one_side_up', 'double_bun', 'hair_bun', 'bangs', 'blunt_bangs', 'parted_bangs', 'swept_bangs', 'asymmetrical_bangs', 'hair_ornament', 'sidelocks', 'short_hair_with_long_locks', 'virtual_youtuber', 'braid', 'tail', 'hairband', 'hairclip', 'hair_bow', 'hair_ribbon', 'side_ponytail', 'glasses', 'heterochromia', 'elf', 'ahoge', 'halo', 'hair_over_one_eye', 'horns', 'hime_cut', 'hair_intakes', 'headgear', 'short_eyebrows', 'thick_eyebrows', 'mole', 'mole_under_eye', 'bow', 'dark_skin', 'colored_skin', 'wings', 'jewelry', 'necktie', 'coat', 'elbow_gloves', 'hat', 'weapon', 'white_shirt', 'armor', 'black_neckwear', 'yellow_bow', 'emblem', 'hood', 'looking_at_viewer', 'blush', 'breasts', 'simple_background', 'smile', 'white_background', 'open_mouth', 'shirt', 'shiny', 'standing', 'long_sleeves', 'artist_name', 'thighs', 'closed_mouth', 'skirt', 'alternate_costume', 'collarbone', 'cowboy_shot', 'bare_shoulders', 'dress', 'looking_to_the_side', 'eyelashes', 'hand_up', 'head_tilt', 'holding', 'twitter_username', 'cleavage', 'eyebrows', 'parted_lips', 'dated', 'watermark', 'upper_body', 'signature', 'underwear', 'shiny_skin', 'fingernails', 'gloves', ':d', 'navel', 'sleeveless', 'frills', 'eyes_visible_through_hair', 'dutch_angle', 'thighhighs', 'web_address', 'grey_background', 'full_body', 'teeth', 'heart', 'jacket', 'sketch', 'from_side', 'arm_up', 'miniskirt', 'sitting', 'black_legwear', 'lips', 'short_sleeves', 'groin', 'collar', 'collared_shirt', 'blurry', 'open_clothes', 'panties', 'pleated_skirt', 'alternate_hairstyle', 'legs', 'bare_arms', 'midriff', 'leaning_forward', 'puffy_sleeves', 'light_smile', 'hands_up', 'stomach', 'looking_away', 'v-shaped_eyebrows', 'shadow', 'skindentation', 'character_name', 'multicolored', ':o', 'cosplay', 'gradient', 'depth_of_field', 'detached_sleeves', 'floating_hair', 'ass', 'school_uniform', 'half-closed_eyes', 'wide_sleeves', 'shoes', 'bare_legs', 'tareme', 'fang', 'choker', 'feet_out_of_frame', 'looking_down', 'flower', 'buttons', 'black_skirt', 'belt', 'gradient_background', 'sweat', 'sleeves_past_wrists', 'outdoors', 'boots', 'swimsuit', 'covered_nipples', 'wing_collar', 'striped', 'red_ribbon', 'nail_polish', 'one-hour_drawing_challenge', 'blouse', 'expressionless', 'light_blush', 'day', 'frown', 'armpits', 'see-through', 'from_above', 'earrings', 'tsurime', 'no_bra', 'off_shoulder', 'embarrassed', 'bowtie', 'dress_shirt', 'tongue', 'halterneck', 'star_(symbol)', 'blurry_background', 'white_legwear', 'nose_blush', 'sideboob', 'leg_up', 'symbol-shaped_pupils', 'arm_support', 'black_gloves', 'light_particles', 'uniform', 'sky', 'sleeveless_dress', 'open_jacket', 'indoors', 'white_dress', 'black_ribbon', 'neck_ribbon', 'english_text', 'short_dress', 'black_footwear', 'foreshortening', 'sweatdrop', 'looking_up', 'sleeveless_shirt', 'looking_back', 'pantyhose', 'from_below', 'vest', 'contrapposto', 'two-tone_background', 'clothing_cutout', 'sparkle', 'red_bow', 'sunlight', 'black_dress', 'areolae', 'outstretched_arm', 'bikini', 'slit_pupils', 'clothes_lift', 'happy', 'arm_at_side', 'bra', 'patreon_username', 'upper_teeth', 'makeup', 'ass_visible_through_thighs', 'arms_up', 'wind', 'legs_together', 'red_neckwear']


def 标签转特征(t: dict) -> np.array:
    return np.array([t.get(s, 0) for s in 人标签])


l = []
for 文件夹 in tqdm(训练文件夹, desc='好人'):
    文件夹 = Path(文件夹)
    for k, 人 in 源(文件夹, 全人):
        l.append(人)
好人 = Counter(l)

with open('好人.json', 'w') as f:
    json.dump(好人, f)


@lru_cache(maxsize=65536)
def json_load_open(x):
    return json.load(open(x))


高频标签 = []
人特征 = {}
人均值 = {}
人方差 = {}
for 文件夹 in tqdm(训练文件夹, desc='人特征'):
    文件夹 = Path(文件夹)
    for k, 人 in 源(文件夹, 全人):
        if 人 not in 好人:
            continue
        标签 = json_load_open(文件夹 / f'{k}.json')
        高频标签 += [*标签]
        特征 = 标签转特征(标签)
        人特征.setdefault(人, []).append(特征)
with open('高频标签.json', 'w', encoding='utf8') as f:
    json.dump([k for k, v in Counter(高频标签).most_common(1024)], f)
for k, v in [*人特征.items()]:
    人均值[k] = np.array(v).mean(axis=0)
    人方差[k] = np.array(v).std(axis=0)
with open('人均值.json', 'w', encoding='utf8') as f:
    json.dump({k: v.tolist() for k, v in 人均值.items()}, f)
with open('人方差.json', 'w', encoding='utf8') as f:
    json.dump({k: v.tolist() for k, v in 人方差.items()}, f)
人均值方差 = {k: (人均值[k], 人方差[k]) for k in 人均值}

人阵, 人均值阵 = [], []
for k, v in 人均值.items():
    人阵.append(k)
    人均值阵.append(v)
人均值阵 = np.array(人均值阵)


n = 0
top1命中 = 0
top3命中 = 0
for 文件夹 in 测试文件夹:
    文件夹 = Path(文件夹)
    for k, 人 in 源(文件夹, 全人):
        j = json_load_open(文件夹 / f'{k}.json')
        特征 = 标签转特征(j)
        预测人 = [人阵[i] for i in np.argsort(np.linalg.norm((特征 - 人均值阵), axis=1))[:3]]
        n += 1
        预测人 = [i.removesuffix('__test__') for i in 预测人]
        if 人 in 预测人[:1]:
            top1命中 += 1
        if 人 in 预测人[:3]:
            top3命中 += 1
print('特征长度', len([*人均值.values()][0]), '样本数', n, 'top1命中', top1命中 / n, 'top3命中', top3命中 / n, '好人数', len(好人))
with open(f'测试.txt', 'a', encoding='utf8') as f:
    print('特征长度', len([*人均值.values()][0]), '样本数', n, 'top1命中', top1命中 / n, 'top3命中', top3命中 / n, '好人数', len(好人), file=f)
