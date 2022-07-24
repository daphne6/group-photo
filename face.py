import cv2
import mediapipe as mp
import numpy as np
import math
import time
import datetime
import subprocess
import sys

face_square = frozenset([(284, 365), (365, 136), (136, 54), (54, 284)])

#=======================================================================#
##目の開き具合の検出に使う


def get_position(landmarks, index):
    return (landmarks.landmark[index].x, landmarks.landmark[index].y)


def is_inside(position): #カメラの中に入っているかどうか
    return (0 < position[0] < 1) and (0 < position[1] < 1)


def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


class Eye:
    def __init__(self, face_landmarks, side): #side: left→True, right→False
        if side: #left eye
            self.horizontal = (get_position(face_landmarks, 263),
                               get_position(face_landmarks, 362))
            self.vertical_1 = (get_position(face_landmarks, 380),
                               get_position(face_landmarks, 385))
            self.vertical_2 = (get_position(face_landmarks, 373),
                               get_position(face_landmarks, 387))
        else: #right eye
            self.horizontal = (get_position(face_landmarks, 33),
                               get_position(face_landmarks, 133))
            self.vertical_1 = (get_position(face_landmarks, 153),
                               get_position(face_landmarks, 158))
            self.vertical_2 = (get_position(face_landmarks, 144),
                               get_position(face_landmarks, 160))

    def is_inside(self): #目がカメラの中に入っているかどうか
        return is_inside(self.horizontal[0]) and is_inside(self.horizontal[1]) and is_inside(self.vertical_1[0]) and is_inside(self.vertical_1[1]) and is_inside(self.vertical_2[0]) and is_inside(self.vertical_2[1])

    def is_open(self):
        #eyes aspect ratio
        ear = distance(self.vertical_1[0], self.vertical_1[1]) + distance(self.vertical_2[0], self.vertical_2[1]) / (2 * distance(self.horizontal[0], self.horizontal[1]))
        if ear <= 0.1:
            return False
        return True
        

class Eyes: #pair of eyes
    def __init__(self, face_landmarks):
        self.left_eye = Eye(face_landmarks, True)
        self.right_eye = Eye(face_landmarks, False)

    def is_inside(self): #両目がカメラの中に入っているかどうか
        return self.left_eye.is_inside() and self.right_eye.is_inside()

    def both_open(self):
        return self.left_eye.is_open() and self.right_eye.is_open()

    def either_open(self):
        return self.left_eye.is_open() or self.right_eye.is_open()

    
#===============================================================#

#===============================================================#
##入力


tmp = input('debug? (y/n)\n')
if tmp == 'y':
    debug = True
elif tmp == 'n' or tmp == '':
    debug = False
else:
    print('invalid input')
    sys.exit()

num = input('何人で撮る？\n')
if num == '': #デフォルトは1人
    num = 1    
try:
    num = int(num)
except ValueError:
    print('invalid input')
    sys.exit()

if num <= 0:
    print('1以上の整数を入れてください')
    sys.exit()

    
max_pic = input('何枚撮る？\n')
if max_pic == '': #デフォルトは1枚
    max_pic = 1
try:
   max_pic = int(max_pic)
except ValueError:
    print('invalid input')
    sys.exit()

if max_pic <= 0:
    print('1以上の整数を入れてください')
    sys.exit()

    
wait_sec = input('何秒待つ？\n')
if wait_sec == '': #デフォルトは3秒
    wait_sec = 3

try:
    wait_sec = int(wait_sec)
except ValueError:
    print('invalid input')
    sys.exit()
    
#===============================================================#

#===============================================================#
##initialize


#mediapipeの画像認識、検知した特徴点の描画関連
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=1,  color=(0,0,255))


cap = cv2.VideoCapture(0) #カメラ画像
first = True #カメラ画像のサイズを1度だけ取得するのに使う
num_pic = 0 #撮影した枚数
original_alert_frame_in = cv2.imread('alert_frame_in.png') #全員フレームに収まるよう呼びかける表示

#規定枚数を取り終わった後の処理に使う画像たち
restart_tytle = cv2.imread('restart_tytle.png')
restart_yes_default = cv2.imread('restart_yes_default.png')
restart_yes_selected = cv2.imread('restart_yes_selected.png')
restart_no_default = cv2.imread('restart_no_default.png')
restart_no_selected = cv2.imread('restart_no_selected.png')
restart_check = cv2.imread('restart_check.png')
#まとめる
restart_images = (restart_tytle, restart_yes_default, restart_yes_selected,
                  restart_no_default, restart_no_selected, restart_check)

run = time.time() #開始時刻。最初の3秒間は写真を撮らないようにする
size = None #キャプチャ画像のサイズ
restart_image = None #リスタート画面
first_restart = True #restartを最初に呼び出すときに使う


#写真を撮るかどうかの判定に使う
start = 0 #全員の目が開き始めた時間
tmp_recorded = False #写真を記録したかどうか
shuttered = 0 #シャッターを切った時間


#===============================================================#

#===============================================================#
##表示する画像の処理をする関数


def scale_to_resolation(img, resolation):
    #指定した解像度になるように、アスペクト比を固定して、リサイズする。
    h, w = img.shape[:2]
    scale = (resolation / (w * h)) ** 0.5
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


'''
#シャッターを切る処理をこのように書くとうまくいかなかった
def shutter(tmp_picture):
    global tmp_recorded
    if tmp_recorded:
        shuttered = time.time()
        cv2.namedWindow('face_mesh', cv2.WINDOW_NORMAL)
        while True:
            t = time.time() - shuttered
            
            if t < 0.1:
                cv2.imshow('face_mesh', adjust(tmp_picture, beta=100.0))
            elif t < 0.3:
                cv2.imshow('face_mesh', adjust(tmp_picture, beta=75.0))
            elif t < 0.5:
                cv2.imshow('face_mesh', adjust(tmp_picture, beta=50.0))
            else:
                cv2.imshow('face_mesh', tmp_picture)
            if t > 1:
                break
            
            cv2.imshow('face_mesh', adjust(tmp_picture, beta=100.0))
            if t > 1:
                break
        tmp_recorded = False
        time.sleep(1)
'''


def adjust(img, alpha=1.0, beta=0.0):
    # alpha → 画像のコントラストを変更
    # beta → 画像の明るさを変更
    dst = alpha * img + beta
    # [0, 255] でクリップし、uint8 型にする。
    return np.clip(dst, 0, 255).astype(np.uint8)


def alert(img, alert_image, darkness = 0.5):
    #imgにalert_imageを透かして重ねる
    return cv2.addWeighted(img, darkness, alert_image, 1.0, 0)


def insert(background, target, area): #backgroundの指定範囲に画像を挿入する
    insert_image = cv2.resize(target, dsize=(area['w'][1] - area['w'][0], area['h'][1] - area['h'][0]))
    res = background.copy()
    res[area['h'][0]:area['h'][1], area['w'][0]:area['w'][1]] = insert_image
    return res


def restart(yes, no, restart_images, _background, first_called):
    global size
    if size == None:
        return None

    
    tytle_area = {'h': (size[0] // 12, size[0] // 4), 'w': (0, size[1])} #タイトルを表示する範囲
    yes_area = {'h': (size[0] // 3, size[0] * 2 // 3), 'w': (size[1] // 8, size[1] * 3 // 8)} #選択肢"yes"を表示する範囲
    no_area = {'h': (size[0] // 3, size[0] * 2 // 3), 'w': (size[1] * 5 // 8, size[1] * 7 // 8)} #選択肢"no"を表示する範囲
    check_area = {'h': (size[0] * 3 // 4, size[0] * 11 // 12), 'w': (size[1] * 3 // 4, size[1] * 7 // 8)} #チェックボタンを表示する範囲

    if first_called:
        background = np.zeros((size[0], size[1], 3), dtype='uint8')
        background = insert(background, restart_images[0], tytle_area) #タイトルを挿入
        background = insert(background, restart_images[5], check_area) #チェックボタンを挿入
    else:
        background = _background

    #選択肢"yes"を挿入
    if yes:
        background = insert(background, restart_images[2], yes_area)
    else:
        background = insert(background, restart_images[1], yes_area)

    #選択肢"no"を挿入
    if no:
        background = insert(background, restart_images[4], no_area)
    else:
        background = insert(background, restart_images[3], no_area)

    return background, yes_area, no_area, check_area


def is_pointed(position, area): #restart時、指がどこをさしているか判断する
    return area['w'][0] < position[0] < area['w'][1] and area['h'][0] < position[1] < area['h'][1]


#===============================================================#



while cap.isOpened():
    if debug:
            tick = cv2.getTickCount() #for fps

    success, original_image = cap.read()
    image_for_picture = original_image.copy()

    image = scale_to_resolation(original_image, 400*400)

    if not success:
        print("empty camera frame")
        first = True
        continue

    if first:
        first = False
        size = original_image.shape[:2]
        alert_frame_in = cv2.resize(original_alert_frame_in, dsize=(size[1], size[0]))

    if num_pic < max_pic:
        if time.time() - run >= wait_sec: #開始から3秒以上経過していたら実行
            with mp_face_mesh.FaceMesh(
                    static_image_mode=False, #False→トラックする
                    max_num_faces=num,
                    refine_landmarks=True, #True→目や唇の周りをより細かく
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5 #ignored if static_image_mode=True
            ) as face_mesh:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                if shuttered == 0: #写真を撮る前の画像処理
                    if results.multi_face_landmarks: #顔を検出している場合
                        eyes = []

                        #検出した顔に対して1つずつ処理
                        for face_landmarks in results.multi_face_landmarks:
                            '''
                            #顔のメッシュ
                            mp_drawing.draw_landmarks(
                                image=original_image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_tesselation_style())

                            #目と眉の強調表示
                            mp_drawing.draw_landmarks(
                                image=original_image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                            '''

                            eyes.append(Eyes(face_landmarks))
                            if len(results.multi_face_landmarks) < num: #検出された顔の数が足りない場合、警告
                                original_image = alert(original_image, alert_frame_in)

                            else: #全員の顔が検出されている場合
                                all_inside = True
                                all_open = True
                                for eye in eyes:
                                    if not eye.both_open():
                                        all_open = False
                                    if not eye.is_inside():
                                        all_inside = False


                                if not all_inside: #目が写っていない場合、警告
                                    original_image = alert(original_image, alert_frame_in)

                                else: #全員の目が写っている場合
                                    if all_open: #目がすべて開いている場合
                                        if start == 0: #開始時間を記録
                                            start = time.time()
                                        else: #経過時間を確認
                                            duration = time.time() - start
                                            if duration > 0.25 and (not tmp_recorded): #0.25秒経過時点で一旦画像を記録(撮影)
                                                tmp_picture = image_for_picture
                                                tmp_recorded = True

                                            if duration > 0.5: #0.5秒経過でシャッターを切る(ような振る舞いをする)。その後の処理は一番下で
                                                shuttered = time.time()
                                                start = 0
                                                subprocess.Popen(['see','shutter.mp3'])

                                            if debug:
                                                cv2.putText(original_image,
                                                            "duration:{:.2f} ".format(duration),
                                                            (300, 50),
                                                            cv2.FONT_HERSHEY_PLAIN,
                                                            3, (0, 0, 255),
                                                            2, cv2.LINE_AA)

                                    else: #目が開いていない場合、経過時間をリセット
                                        start = 0
                                        tmp_recorded = False


                            if debug:
                                mp_drawing.draw_landmarks(
                                    image=original_image,
                                    landmark_list=face_landmarks,
                                    connections=face_square,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mesh_drawing_spec)


                        if debug:
                            fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
                            cv2.putText(original_image, "FPS:{} ".format(int(fps)), (10, 50),
                                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_AA)

                        cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty('camera', cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
                        cv2.imshow('camera', original_image)

                    else: #顔を検出していない場合、リセットして警告
                        shuttered = 0
                        tmp_recorded = False
                        start = 0
                        original_image = alert(original_image, alert_frame_in)
                        cv2.imshow('camera', original_image)

                else: #シャッターを切ったあとの処理
                    if not tmp_recorded: #写真が取れていなければやめる(ここは実行されないはず)
                        shuttered = 0
                    else:
                        t = time.time() - shuttered
                        #0.5秒間のフラッシュ演出
                        if t < 0.1:
                            cv2.imshow('camera', adjust(tmp_picture, beta=100.0)) #かなり白い画像
                        elif t < 0.3:
                            cv2.imshow('camera', adjust(tmp_picture, beta=75.0)) #まぁまぁ白い画像
                        elif t < 0.5:
                            cv2.imshow('camera', adjust(tmp_picture, beta=50.0)) #少し白い画像
                        #その後1秒間は撮影した写真を表示
                        else:
                            cv2.imshow('camera', tmp_picture)
                        if t > 1.5:
                            now = datetime.datetime.now()
                            cv2.imwrite('pictures/' + now.strftime("%Y%m%d %H:%M:%S") + '.jpg', tmp_picture)
                            num_pic += 1
                            tmp_recorded = False
                            shuttered = 0

                            
        else: #3秒経過するまでは、キャプチャを少し暗くして表示
            cv2.putText(original_image, "waiting...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
            cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('camera', cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.imshow('camera', adjust(original_image, beta=-30))

    else: #規定枚数の撮影後は、その後どうするかを選択(手の画像認識を用いる)
        if first_restart: #initialize
            yes = False
            no = False
            finish = False

        try:
            restart_image, yes_area, no_area, check_area = restart(yes, no, restart_images, restart_image, first_restart)
            
        except TypeError:
            break

        first_restart = False
        original_image = cv2.flip(original_image, 1)
        original_image = alert(original_image, restart_image, 0.3)
        with mp_hands.Hands(
                static_image_mode=False, #False→トラックする
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5 #ignored if static_image_mode=True
        ) as hands_detection:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands_detection.process(cv2.flip(image, 1))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    position = (int(size[1] * hand_landmarks.landmark[8].x), int(size[0] * hand_landmarks.landmark[8].y))
                    if debug:
                        print(hand_landmarks.landmark[8])

                    if hand_landmarks.landmark[8].z < -0.05: #手が画面に近い時は操作を受け付ける
                        cv2.circle(original_image, position, 3, (0, 255, 0), -1)
                        if is_pointed(position, check_area): #確定ボタンを押している場合
                            if debug:
                                print('check')
                            if yes:
                                num_pic = 0
                                run = time.time()
                                first_restart = True
                            elif no:
                                finish = True
                        elif is_pointed(position, yes_area): #yesボタンを押している場合
                            if debug:
                                print('yes')
                            yes = True
                            no = False
                        elif is_pointed(position, no_area): #noボタンを押している場合
                            if debug:
                                print('no')
                            yes = False
                            no = True

                    else:
                        cv2.circle(original_image, position, 3, (0, 0, 255), -1)

        cv2.imshow('camera', original_image) #操作しやすいよう、鏡に映して表示
        if finish:
            break

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
