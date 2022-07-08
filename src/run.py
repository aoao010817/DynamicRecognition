import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# カメラから画像を取得して,リアルタイムに手書き数字を判別させる。
# 動画表示
cap = cv2.VideoCapture(0)

model = load_model("MNIST.h5") # 学習済みモデルをロード

while(True):
    ret, frame = cap.read()
    h, w, _ = frame.shape[:3]

    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # グレースケールに変換
    _, th = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU) # 2値化

    # 輪郭抽出
    contours = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for moji in contours:
        x, y, w, h = cv2.boundingRect(moji)
        if h < 30 or w < 30: continue
        red = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)
        if h >= w:
            l = h
        else:
            l = w
        l = l + (28-l%28)

        
        im = frame[y:y+l, x:x+l]
        im = cv2.bitwise_not(th) # 白黒反転
        im = cv2.GaussianBlur(th,(9,9), 0) # ガウスブラーをかけて補間
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
        cv2.putText(frame, "1:"+str(Yt[9]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("frame",frame) # カメラ画像を表示

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # カメラを解放
cv2.destroyAllWindows() # ウィンドウを消す