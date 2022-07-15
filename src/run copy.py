import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# カメラから画像を取得して,リアルタイムに手書き数字を判別させる。
# 動画表示
cap = cv2.VideoCapture(0)

model = load_model("MNIST.h5") # 学習済みモデルをロード

while(True):

    Xt = []
    Yt = []

    ret, frame = cap.read()

    # 画像のサイズを取得,表示。グレースケールの場合,shape[:2]
    h, w, _ = frame.shape[:3]

    # 画像の中心点を計算
    w_center = w//2
    h_center = h//2

    # 画像の真ん中に142×142サイズの四角を描く
    cv2.rectangle(frame, (w_center-71, h_center-71), (w_center+71, h_center+71),(255, 0, 0))

    # カメラ画像の整形
    im = frame[h_center-70:h_center+70, w_center-70:w_center+70] # トリミング
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # グレースケールに変換
    _, th = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU) # 2値化
    th = cv2.bitwise_not(th) # 白黒反転
    th = cv2.GaussianBlur(th,(9,9), 0) # ガウスブラーをかけて補間
    th = cv2.resize(th,(28, 28), cv2.INTER_CUBIC) # 訓練データと同じサイズに整形

    Xt.append(th)
    Xt = np.array(Xt)/255

    print(th.shape)
    exit()

    result = model.predict(Xt, batch_size=1) # 判定,ソート
    for i in range(10):
        r = round(result[0,i], 2)
        Yt.append([i, r])
        Yt = sorted(Yt, key=lambda x:(x[1]))

    # 判定結果を上位3番目まで表示させる
    cv2.putText(frame, "1:"+str(Yt[9]), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, "2:"+str(Yt[8]), (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, "3:"+str(Yt[7]), (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
    cv2.imshow("frame",frame) # カメラ画像を表示

    k =  cv2.waitKey(1) & 0xFF # キーが押下されるのを待つ。1秒置き。64ビットマシンの場合,& 0xFFが必要

    if k == ord("q"): # 終了処理
        break

cap.release() # カメラを解放
cv2.destroyAllWindows() # ウィンドウを消す