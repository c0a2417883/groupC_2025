import cv2  # openCVライブラリのインポート
import numpy as np  # numpyライブラリのインポート
from cv2 import aruco, imread, imwrite

##　↓↓↓↓↓↓↓inRangeWrap, calc_centroidは変更しないでください↓↓↓↓↓↓
# inRangeを色相が0付近や180付近の色へ対応する形へ修正
def inRangeWrap(hsv, lower, upper):
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)
    else:
        # 180をまたぐ場合
        lower1 = np.array([0, lower[1], lower[2]])
        upper1 = np.array([upper[0], upper[1], upper[2]])
        lower2 = lower
        upper2 = np.array([179, upper[1], upper[2]])
        return cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )
    
def calc_centroid(mask):
    M = cv2.moments(mask)
    if M["m00"] != 0:
        # 重心座標を計算SS
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        s = np.count_nonzero(mask)/(mask.shape[0]*mask.shape[1])
        return cx, cy, s
    else:
        return None   
##　↑↑↑↑↑↑↑inRangeWrap, calc_centroidは変更しないでください↑↑↑↑↑↑↑

def o_ball(img):
    # 画像の読み込み
    o_draw_img = img.copy() # 元データを書き換えないようにコピーを作成
    # HSVに変換（色指定はRGBよりHSVの方が扱いやすい）
    o_hsv_img = cv2.cvtColor(o_draw_img, cv2.COLOR_BGR2HSV)

    # BGR空間での抽出範囲
    ## ボール
    o_lower = np.array([0, 220, 170]) # 色相, 彩度, 明度 の下限
    o_upper = np.array([10, 240, 255]) # 色相, 彩度, 明度 の上限

    # 指定範囲に入る画素を抽出（白が該当部分）
    o_mask = inRangeWrap(o_hsv_img, o_lower, o_upper)
    
    try:
        o_x, o_y, o_s = calc_centroid(o_mask)
        print(f"{o_s=}")
        return o_x, o_y
    except TypeError:
        return None


def p_ball(img):
    # 画像の読み込み
    p_draw_img = img.copy() # 元データを書き換えないようにコピーを作成
    # HSVに変換（色指定はRGBよりHSVの方が扱いやすい）
    p_hsv_img = cv2.cvtColor(p_draw_img, cv2.COLOR_BGR2HSV)

    # BGR空間での抽出範囲
    ## ボール
    # p_lower = np.array([199, 0, 0]) # 色相, 彩度, 明度 の下限
    # p_upper = np.array([220, 50, 162]) # 色相, 彩度, 明度 の上限

    p_lower = np.array([110, 100, 100]) # 色相, 彩度, 明度 の下限
    p_upper = np.array([160, 255, 250]) # 色相, 彩度, 明度 の上限

    # 指定範囲に入る画素を抽出（白が該当部分）
    p_mask = inRangeWrap(p_hsv_img, p_lower, p_upper)
    
    try:
        p_x, p_y, p_s = calc_centroid(p_mask)
        print(f"{p_s=}")
        return p_x, p_y
    except TypeError:
        return None

def d_circle(img):
    # 画像読み込み
    draw_img = img.copy()

    # 前処理（グレースケール＋ぼかし）
    gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # ノイズ低減

    # 円検出（HoughCircles）
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=30,
        param1=100, param2=50,   
        minRadius=10, maxRadius=40  
    )
    # マスク作成（検出円を塗りつぶし）
    mask = np.zeros(gray.shape, dtype=np.uint8)
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for x, y, r in circles:
            cv2.circle(mask, (x, y), r, 255, -1)     # マスク（白塗り）
            break

    x, y, s = calc_centroid(mask)
    print(f"{s=}")
    if x and y:
        return x, y
    else:
        return None

