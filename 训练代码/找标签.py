import json
from pathlib import Path
from bayes_opt import BayesianOptimization

from rimo_storage.cache import disk_cache
import numpy as np
import ml_danbooru


人标签 = ['animal_ears', 'animal_ear_fluff', 'pointy_ears', 'cat_ears', 'fox_ears', 'dog_ears', 'horse_ears', 'multiple_tails', 'rabbit_tail', 'cat_tail', 'fox_tail', 'dog_tail', 'demon_tail', 'blue_eyes', 'hair_between_eyes', 'purple_eyes', 'green_eyes', 'brown_eyes', 'red_eyes', 'closed_eyes', 'pink_eyes', 'yellow_eyes', 'aqua_eyes', 'grey_eyes', 'black_eyes', 'orange_eyes', 'white_eyes', 'glowing_eyes', 'hair_over_eyes', 'hair_over_shoulder', 'long_hair', 'short_hair', 'brown_hair', 'eyebrows_visible_through_hair', 'black_hair', 'blonde_hair', 'very_long_hair', 'blue_hair', 'purple_hair', 'silver_hair', 'white_hair', 'pink_hair', 'grey_hair', 'pubic_hair', 'medium_hair', 'multicolored_hair', 'red_hair', 'two-tone_hair', 'shiny_hair', 'streaked_hair', 'orange_hair', 'gradient_hair', 'aqua_hair', 'green_hair', 'light_purple_hair', 'light_brown_hair', 'tied_hair', 'straight_hair', 'asymmetrical_hair', 'spiked_hair', 'light_blue_hair', 'platinum_blonde_hair', 'eyebrows_behind_hair', 'colored_inner_hair', 'drill_hair', 'wavy_hair', 'low-tied_long_hair', 'antenna_hair', 'medium_breasts', 'large_breasts', 'small_breasts', 'huge_breasts', 'gigantic_breasts', 'flat_chest', 'ponytail', 'high_ponytail', 'ribbon', 'two_side_up', 'twintails', 'short_twintails', 'low_twintails', 'one_side_up', 'double_bun', 'hair_bun', 'bangs', 'blunt_bangs', 'parted_bangs', 'swept_bangs', 'asymmetrical_bangs', 'hair_ornament', 'sidelocks', 'short_hair_with_long_locks', 'virtual_youtuber', 'braid', 'tail', 'hairband', 'hairclip', 'hair_bow', 'hair_ribbon', 'side_ponytail', 'glasses', 'heterochromia', 'elf', 'ahoge', 'halo', 'hair_over_one_eye', 'horns', 'hime_cut', 'hair_intakes', 'headgear', 'short_eyebrows', 'thick_eyebrows', 'mole', 'mole_under_eye', 'bow', 'dark_skin', 'colored_skin', 'wings', 'jewelry', 'necktie', 'coat', 'elbow_gloves', 'hat', 'weapon', 'white_shirt', 'armor', 'black_neckwear', 'yellow_bow', 'emblem', 'hood', 'looking_at_viewer', 'blush', 'breasts', 'simple_background', 'smile', 'white_background', 'open_mouth', 'shirt', 'shiny', 'standing', 'long_sleeves', 'artist_name', 'thighs', 'closed_mouth', 'skirt', 'alternate_costume', 'collarbone', 'cowboy_shot', 'bare_shoulders', 'dress', 'looking_to_the_side', 'eyelashes', 'hand_up', 'head_tilt', 'holding', 'twitter_username', 'cleavage', 'eyebrows', 'parted_lips', 'dated', 'watermark', 'upper_body', 'signature', 'underwear', 'shiny_skin', 'fingernails', 'gloves', ':d', 'navel', 'sleeveless', 'frills', 'eyes_visible_through_hair', 'dutch_angle', 'thighhighs', 'web_address', 'grey_background', 'full_body', 'teeth', 'heart', 'jacket', 'sketch', 'from_side', 'arm_up', 'miniskirt', 'sitting', 'black_legwear', 'lips', 'short_sleeves', 'groin', 'collar', 'collared_shirt', 'blurry', 'open_clothes', 'panties', 'pleated_skirt', 'alternate_hairstyle', 'legs', 'bare_arms', 'midriff', 'leaning_forward', 'puffy_sleeves', 'light_smile', 'hands_up', 'stomach', 'looking_away', 'v-shaped_eyebrows', 'shadow', 'skindentation', 'character_name', 'multicolored', ':o', 'cosplay', 'gradient', 'depth_of_field', 'detached_sleeves', 'floating_hair', 'ass', 'school_uniform', 'half-closed_eyes', 'wide_sleeves', 'shoes', 'bare_legs', 'tareme', 'fang', 'choker', 'feet_out_of_frame', 'looking_down', 'flower', 'buttons', 'black_skirt', 'belt', 'gradient_background', 'sweat', 'sleeves_past_wrists', 'outdoors', 'boots', 'swimsuit', 'covered_nipples', 'wing_collar', 'striped', 'red_ribbon', 'nail_polish', 'one-hour_drawing_challenge', 'blouse', 'expressionless', 'light_blush', 'day', 'frown', 'armpits', 'see-through', 'from_above', 'earrings', 'tsurime', 'no_bra', 'off_shoulder', 'embarrassed', 'bowtie', 'dress_shirt', 'tongue', 'halterneck', 'star_(symbol)', 'blurry_background', 'white_legwear', 'nose_blush', 'sideboob', 'leg_up', 'symbol-shaped_pupils', 'arm_support', 'black_gloves', 'light_particles', 'uniform', 'sky', 'sleeveless_dress', 'open_jacket', 'indoors', 'white_dress', 'black_ribbon', 'neck_ribbon', 'english_text', 'short_dress', 'black_footwear', 'foreshortening', 'sweatdrop', 'looking_up', 'sleeveless_shirt', 'looking_back', 'pantyhose', 'from_below', 'vest', 'contrapposto', 'two-tone_background', 'clothing_cutout', 'sparkle', 'red_bow', 'sunlight', 'black_dress', 'areolae', 'outstretched_arm', 'bikini', 'slit_pupils', 'clothes_lift', 'happy', 'arm_at_side', 'bra', 'patreon_username', 'upper_teeth', 'makeup', 'ass_visible_through_thighs', 'arms_up', 'wind', 'legs_together', 'red_neckwear']


with open(Path(__file__)/'../人均值.json') as f:
    _人均值 = json.load(f)
_人均值 = {k: np.array(v) for k, v in _人均值.items()}

_人阵, _人均值阵 = [], []
for k, v in _人均值.items():
    _人阵.append(k)
    _人均值阵.append(v)
_人均值阵 = np.array(_人均值阵)


def _标签转特征(t: dict) -> np.array:
    return np.array([t.get(s, 0) for s in 人标签])


def predict(image, top_n=3, size=256) -> list:
    tags = ml_danbooru.get_tags_from_image(image, threshold=0.4, keep_ratio=True, size=size)
    特征 = _标签转特征(tags)
    距离 = np.linalg.norm(特征 - _人均值阵, axis=1)
    预测人 = [(_人阵[i], 距离[i]) for i in np.argsort(距离)[:top_n]]
    return 预测人


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


def json_load_open(x):
    return json.load(open(x))

全人 = {*json.load(open('人.json'))}


测试文件夹 = ['data-0000', 'data-0020']


优化记录 = []
def 烙(**原w):
    items = sorted([(int(k), v) for k, v in 原w.items()])
    w = np.array([v for k, v in items])
    n = 0
    top1命中 = 0
    top3命中 = 0
    for 文件夹 in 测试文件夹:
        文件夹 = Path(文件夹)
        for k, 人 in 源(文件夹, 全人):
            j = json_load_open(文件夹 / f'{k}.json')
            特征 = _标签转特征(j)
            预测人 = [_人阵[i] for i in np.argsort(np.linalg.norm(w * (特征 - _人均值阵), axis=1))[:3]]
            n += 1
            预测人 = [i.removesuffix('__test__') for i in 预测人]
            if 人 in 预测人[:1]:
                top1命中 += 1
            if 人 in 预测人[:3]:
                top3命中 += 1
    优化记录.append({
        'w': w.tolist(),
        'top1命中率': top1命中 / n,
    })
    with open('优化记录.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(优化记录, ensure_ascii=False))
    return top1命中 / n, top3命中 / n, 


所有参数 = [str(i) for i in range(len(人标签))]

optimizer = BayesianOptimization(
    f=烙,
    pbounds={k: (0, 1) for k in 所有参数},
    random_state=1,
)
optimizer.probe(params={k: 0 for k in 所有参数 if int(k) >= 130} | {k: 1 for k in 所有参数 if int(k) < 130})
optimizer.probe(params={k: 1 for k in 所有参数 if int(k) >= 130} | {k: 0 for k in 所有参数 if int(k) < 130})
optimizer.probe(params={k: 0.5 for k in 所有参数 if int(k) >= 130} | {k: 1 for k in 所有参数 if int(k) < 130})
optimizer.probe(params={k: 1 for k in 所有参数 if int(k) >= 130} | {k: 0.5 for k in 所有参数 if int(k) < 130})
optimizer.probe(params={k: 1 for k in 所有参数})
optimizer.probe(params={k: 0.5 for k in 所有参数})
optimizer.maximize(
    init_points=4,
    n_iter=10000,
)
