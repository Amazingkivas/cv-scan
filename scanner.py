#!/usr/bin/env python3
# Сканер документов на OpenCV
# Делает: находит углы, выравнивает через warpPerspective,
# проверяет параллельность линий, считает строки и символы

import argparse
import cv2
import numpy as np
from pathlib import Path

# какие расширения вообще поддерживаем
exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def take_input(path):
    # если передали файл - берём его, если папку - берём единственную картинку
    if path.is_file():
        if path.suffix.lower() not in exts:
            print("неизвестный формат:", path.suffix)
            exit(1)
        return path
    if not path.exists():
        print("нет такого пути:", path)
        exit(1)
    files = []
    for p in path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    if len(files) != 1:
        print("в папке должно быть ровно одно изображение, а тут", len(files))
        exit(1)
    return files[0]


def sort_corners(pts):
    # надо разложить 4 точки как: левый-верх, правый-верх, правый-низ, левый-низ
    pts = pts.reshape(4, 2).astype(np.float32)
    out = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1).ravel()
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def find_corners(img):
    # ищем границы страницы
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(closed, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_area = img.shape[0] * img.shape[1] * 0.12
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        if cv2.contourArea(c) < min_area:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return sort_corners(approx)

    print("не нашёл 4 угла документа")
    exit(1)


def warp_doc(img, corners):
    tl, tr, br, bl = corners

    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    width = int(round(max(w1, w2)))

    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    height = int(round(max(h1, h2)))

    if width < 2 or height < 2:
        print("что-то не так с размерами, ширина/высота слишком маленькие")
        exit(1)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (width, height))


def get_lines(img):
    # находим все прямые отрезки и считаем их углы
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)

    # порог низкий, чтобы поймать заголовок и короткие строки
    thr = max(15, min(img.shape[:2]) // 15)
    minlen = max(20, min(img.shape[:2]) // 15)

    # maxLineGap побольше, чтобы слова в заголовке сливались в одну линию
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=thr,
                            minLineLength=minlen, maxLineGap=25)

    horiz = []
    vert = []
    segs = []

    if lines is None:
        return segs, horiz, vert

    lines = np.asarray(lines).reshape(-1, 4)

    # минимальная длина "настоящей" линии - 15% от меньшей стороны картинки
    # ВАЖНО: img.shape[:2] - только высота и ширина, без каналов!
    real_len = min(img.shape[:2]) * 0.15

    for x1, y1, x2, y2 in lines:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

        is_horiz = (ang <= 20 or ang >= 160)
        is_vert = (70 <= ang <= 110)

        if is_horiz:
            if ang > 90:
                ang_h = ang - 180
            else:
                ang_h = ang
            segs.append((int(x1), int(y1), int(x2), int(y2)))
            if length >= real_len:
                horiz.append(ang_h)
        elif is_vert:
            segs.append((int(x1), int(y1), int(x2), int(y2)))
            if length >= real_len:
                vert.append(ang - 90)

    return segs, horiz, vert


def max_dev(angles):
    # максимальное отклонение угла от медианы
    if len(angles) < 2:
        return None
    med = np.median(angles)
    m = 0.0
    for a in angles:
        if abs(a - med) > m:
            m = abs(a - med)
    return float(m)


def fix_skew(img):
    # проверяем параллельность и если надо - подкручиваем
    _, horiz, vert = get_lines(img)

    h_dev = max_dev(horiz)
    v_dev = max_dev(vert)

    # средний угол наклона линий
    cands = []
    if len(horiz) > 0:
        cands.append(np.median(horiz))
    if len(vert) > 0:
        cands.append(np.median(vert))
    if len(cands) > 0:
        angle = float(np.median(cands))
    else:
        angle = 0.0

    tol = 3.0
    bad = False
    if h_dev is not None and h_dev > tol:
        bad = True
    if v_dev is not None and v_dev > tol:
        bad = True

    result_img = img
    result_stats = {
        "h_dev": h_dev,
        "v_dev": v_dev,
        "angle": 0.0
    }

    if bad and abs(angle) > 0.05:
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # проверяем, стало ли лучше
        _, h2, v2 = get_lines(rotated)
        h_dev2 = max_dev(h2)
        v_dev2 = max_dev(v2)

        before = 0
        if h_dev is not None:
            before = h_dev
        if v_dev is not None and v_dev > before:
            before = v_dev

        after = 0
        if h_dev2 is not None:
            after = h_dev2
        if v_dev2 is not None and v_dev2 > after:
            after = v_dev2

        if after < before:
            result_img = rotated
            result_stats["h_dev"] = h_dev2
            result_stats["v_dev"] = v_dev2
            result_stats["angle"] = angle
        else:
            result_stats["angle"] = 0.0

    ok = True
    for d in (result_stats["h_dev"], result_stats["v_dev"]):
        if d is not None and d > tol:
            ok = False
    result_stats["parallel"] = ok

    return result_img, result_stats


def count_text(img):
    # считаем строки и символы без всякого OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bin_img = cv2.adaptiveThreshold(gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 12)

    # убираем длинные горизонтальные линии - они мешают считать строки
    h_len = max(25, img.shape[1] // 8)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, h_kernel)
    bin_img = cv2.subtract(bin_img, h_lines)

    # то же самое для вертикальных
    v_len = max(25, img.shape[0] // 8)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, v_kernel)
    bin_img = cv2.subtract(bin_img, v_lines)

    proj = (bin_img > 0).sum(axis=1)
    threshold = max(2, img.shape[1] * 0.008)

    rows = []
    start = None
    for i in range(len(proj)):
        if proj[i] > threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= 2:
                    rows.append((start, i))
                start = None
    if start is not None and len(proj) - start >= 2:
        rows.append((start, len(proj)))

    total_chars = 0
    for top, bot in rows:
        strip = bin_img[top:bot]
        n, _, stats, _ = cv2.connectedComponentsWithStats(strip, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= 3 and h >= 2 and w >= 1:
                total_chars += 1

    return len(rows), total_chars


def draw_lines(img):
    # рисуем только длинные отрезки, чтобы не замазать текст
    overlay = img.copy()
    segs, _, _ = get_lines(img)

    real_len = min(img.shape[:2]) * 0.15

    for x1, y1, x2, y2 in segs:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length >= real_len:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="сканер документов")
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()

    src = take_input(args.input)

    img = cv2.imread(str(src))
    if img is None:
        print("не смог открыть картинку:", src)
        return

    # 1. ищем углы
    corners = find_corners(img)

    # 2. выравниваем перспективу
    warped = warp_doc(img, corners)

    # 3. правим наклон по линиям
    aligned, stats = fix_skew(warped)

    # 4. считаем строки и символы
    lines_count, chars_count = count_text(aligned)

    args.output.mkdir(parents=True, exist_ok=True)

    corner_vis = img.copy()
    cv2.polylines(corner_vis, [corners.astype(np.int32)], True, (0, 255, 0), 4)
    cv2.imwrite(str(args.output / "corners.png"), corner_vis)

    cv2.imwrite(str(args.output / "scanned.png"), aligned)

    cv2.imwrite(str(args.output / "lines.png"), draw_lines(aligned))

    # краткий отчёт
    print()
    print("=" * 40)
    print("РЕЗУЛЬТАТ")
    print("=" * 40)
    print("файл:", src)
    print("размер после выравнивания:", aligned.shape[1], "x", aligned.shape[0])
    print("строк найдено:", lines_count)
    print("символов найдено:", chars_count)
    print("линии параллельны:", stats["parallel"])
    print("поворот при выравнивании:", stats["angle"], "град.")
    print()
    print("файлы сохранены в", args.output)


if __name__ == "__main__":
    main()