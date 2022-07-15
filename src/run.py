import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import copy

# カメラから画像を取得して,リアルタイムに手書き数字を判別させる。
# 動画表示
cap = cv2.VideoCapture(0)

model = load_model("MNIST.h5") # 学習済みモデルをロード

while(True):
    ret, frame = cap.read()
    h_, w_, _ = frame.shape[:3]

    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # グレースケールに変換
    _, th = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU) # 2値化

    # 輪郭抽出
    contours = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    for moji in contours:
        x, y, w, h = cv2.boundingRect(moji)
        if h < 40 or w < 40: continue
        if h > h_/2 or w > w_/2: continue
        red = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)
        l = max(w, h)
        # cv2.rectangle(frame, (max(0, x-int(l*0.5)), max(0, y-int(l*0.5))), (x+int(l*1.5), y+int(l*1.5)), red, 2)

        im = frame[max(0, y-int(l*0.2)):y+int(l*1.2), max(0, x-int(l*0.2)):x+int(l*1.2)]

        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # グレースケール
        _, im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU) # 2値化
        im = cv2.bitwise_not(im) # 白黒反転
        im = cv2.GaussianBlur(im,(9,9), 0) # ガウスブラーをかけて補間
        im = cv2.resize(im,(28, 28), cv2.INTER_CUBIC) # 訓練データと同じサイズに整形
        Xt = []
        Yt = []

        Xt.append(im)
        Xt = np.array(Xt)/255

        

        result = model.predict(Xt, batch_size=1) # 判定,ソート
        for i in range(10):
            r = round(result[0,i], 2)
            Yt.append([i, r])
        Yt = sorted(Yt, key=lambda x:(x[1]))

        cv2.putText(frame, str(Yt[-1]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("frame",frame) # カメラ画像を表示

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # カメラを解放
cv2.destroyAllWindows() # ウィンドウを消す