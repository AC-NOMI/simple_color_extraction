import numpy as np
import cv2
import random
import math
from simpel_extract_colors_constant import *


class Area:
    def __init__(self, key, cnt, score, mean_lab, mean_bgr):
        self.key = key
        self.cnt = cnt
        self.keys = set()
        self.score = score
        self.mean_lab = mean_lab
        self.mean_bgr = mean_bgr


class DominantColor:
    def __init__(self, bgr=np.zeros(shape=3), lab=np.zeros(shape=3)):
        self.bgr = bgr
        self.lab = lab


def ge_randint(upper_exclusive=256):
    return random.randint(0, upper_exclusive-1)


def create_key(array):
    b = array[0]
    g = array[1]
    r = array[2]
    return r + g * 256 + b * 256 * 256


def filter_areas(areas, percent=0.005):
    sum_pixel = sum([area.cnt for area in areas])
    res = [area for area in areas if area.cnt / sum_pixel > percent]
    return res


def color_difference(lab1, lab2):
    l1 = lab1[0] / 255 * 100
    l2 = lab2[0] / 255 * 100
    dl, da, db = l1 - l2, lab1[1] - lab2[1], lab1[2] - lab2[2]
    return math.sqrt(dl*dl + da*da + db*db)


def merge_areas(areas: list, tolerance):
    res = list()
    table = set()
    for i in range(len(areas)):
        if i in table:
            continue
        area = areas[i]
        area.keys.add(area.key)

        for j in range(i+1, len(areas)):
            if j in table:
                continue
            diff = color_difference(areas[i].mean_lab, areas[j].mean_lab)

            if diff < tolerance:
                table.add(j)
                area.cnt += areas[j].cnt
                area.keys.add(areas[j].key)

                wj = areas[j].cnt / area.cnt
                w = 1 - wj
                area.mean_bgr = area.mean_bgr * w + areas[j].mean_bgr * wj
                area.mean_lab = area.mean_lab * w + areas[j].mean_lab * wj
        res.append(area)
    return res


def extract(arr: np.array, color_count=None) -> np.array:
    img = arr

    height, width, channel_size = img.shape
    new_height, new_width = int(height / width * 400), 400
    img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # handle RGB-A or gray-scale image
    alpha_channel = None
    if channel_size > 3:
        alpha_channel = img[:, :, 3]
        img = img[:, :, :3]
    elif channel_size < 3:
        img = np.stack(img[:, :, 0], axis=2)

    im_mean_shift = cv2.pyrMeanShiftFiltering(img, spatial_rad, color_rad, maxLevel=max_pyr_level)
    im_mean_shift_seg = im_mean_shift.copy()
    im_seg_height, im_seg_width = im_mean_shift_seg.shape[:2]
    if alpha_channel is None:
        mask = np.zeros(shape=(im_seg_height+2, im_seg_width+2), dtype="uint8")
    else:
        mask = np.full(shape=(im_seg_height+2, im_seg_width+2), fill_value=255, dtype='uint8')
        for y in range(im_seg_height):
            for x in range(im_seg_width):
                if alpha_channel[y, x]:
                    mask[y+1, x+1] = 0

    for y in range(im_seg_height):
        for x in range(im_seg_width):
            if mask[y+1, x+1] == 0:
                new_value = (ge_randint(), ge_randint(), ge_randint())
                cv2.floodFill(im_mean_shift_seg, mask, (x, y), new_value, (5, 5, 5), (5, 5, 5))

    im_mean_shift_lab = cv2.cvtColor(im_mean_shift, code=cv2.COLOR_BGR2Lab)

    table = dict()
    for y in range(im_seg_height):
        for x in range(im_seg_width):
            if alpha_channel is not None and alpha_channel[y, x] == 0:
                continue
            key = create_key(im_mean_shift_seg[y, x])
            if key in table:
                table[key].cnt += 1
                table[key].mean_bgr += im_mean_shift[y, x]
                table[key].mean_lab += im_mean_shift_lab[y, x]
            else:
                new_area = Area(key, cnt=1, score=0, mean_bgr=np.zeros(shape=3), mean_lab=np.zeros(shape=3))
                new_area.mean_bgr += im_mean_shift[y, x]
                new_area.mean_lab += im_mean_shift_lab[y, x]
                table[key] = new_area
    values = [v for v in table.values()]
    for value in values:
        value.mean_bgr /= value.cnt
        value.mean_lab /= value.cnt

    all_areas = values

    all_areas = sorted(all_areas, key=lambda a: a.cnt, reverse=True)
    areas = filter_areas(all_areas, area_filtering_percent_threshold)
    areas = merge_areas(areas, merge_area_tolerance)
    areas = sorted(areas, key=lambda a: a.cnt, reverse=True)
    if color_count is not None:
        areas = areas[:min(len(areas), color_count)]
    results = []
    for area in areas:
        results.append(DominantColor(area.mean_bgr, area.mean_lab))
    return results


def extract_dominant_colors_by_file_path(file_path):
    image = cv2.imread(file_path)
    return test_extract_dominant_colors_by_image(image)


def test_extract_dominant_colors_by_image(image):
    colors_results = extract(image, color_count=4)
    color_platte = None
    for color in colors_results:
        color_array = np.zeros(shape=(50, 400, 3))
        rows, cols = color_array.shape[:2]
        for y in range(rows):
            for x in range(cols):
                color_array[y, x] = np.array(color.bgr)
        color_array = color_array.astype(dtype="uint8")
        if color_platte is None:
            color_platte = color_array
        else:
            color_platte = np.concatenate([color_platte, color_array], axis=0)
    print(color_platte.shape)
    cv2.imwrite("images/color_result.jpg", color_platte)
    return color_platte


def main():
    # set demo image path here
    demo_image_path = "images/demo.jpg"
    color_array = extract_dominant_colors_by_file_path(demo_image_path)


if __name__ == '__main__':
    main()
