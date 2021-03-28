
from PIL import ImageGrab
import cv2 as cv
import pygetwindow as pgw
import numpy as np
from time import time
import pyautogui

#haystack_img = cv.imread('img/window.png', cv.IMREAD_UNCHANGED)

pawn_b_img = cv.imread('img/pawn_b.png', cv.IMREAD_UNCHANGED) #0.82
pawn_w_img = cv.imread('img/pawn_w.png', cv.IMREAD_UNCHANGED) #0.42 -> 0.39

king_b_img = cv.imread('img/king_b.png', cv.IMREAD_UNCHANGED) #0.78
king_w_img = cv.imread('img/king_w.png', cv.IMREAD_UNCHANGED) #0.52

queen_b_img = cv.imread('img/queen_b.png', cv.IMREAD_UNCHANGED)
queen_w_img = cv.imread('img/queen_w.png', cv.IMREAD_UNCHANGED)

bishop_b_img = cv.imread('img/bishop_b.png', cv.IMREAD_UNCHANGED) #0.76 -> 0.73
bishop_w_img = cv.imread('img/bishop_w.png', cv.IMREAD_UNCHANGED) #0.5 -> 0.45

knight_b_img = cv.imread('img/knight_b.png', cv.IMREAD_UNCHANGED) #0.75
knight_w_img = cv.imread('img/knight_w.png', cv.IMREAD_UNCHANGED) #0.4 -> 0.36

rook_b_img = cv.imread('img/rook_b.png', cv.IMREAD_UNCHANGED) #0.75 -> 0.72
rook_w_img = cv.imread('img/rook_w.png', cv.IMREAD_UNCHANGED) #0.51 -> 0.47

#piece = [[imgb, threshold], [imgw, threshold]]
rooks = [[rook_b_img, 0.75, "Rook"], [rook_w_img, 0.51, "Rook"]]
bishops = [[bishop_b_img, 0.76, "Bishop"], [bishop_w_img,0.5, "Bishop"]]
knights = [[knight_b_img,0.75, "Knight"], [knight_w_img,  0.4, "Knight"]]
pawns = [[pawn_b_img, 0.8, "Pawn"],[pawn_w_img, 0.4, "Pawn"]]
king = [[king_b_img, 0, "King"], [king_w_img, 0, "King"]]
queen = [[queen_b_img, 0, "Queen"], [queen_w_img, 0, "Queen"]]

pieces=[rooks, bishops, knights, pawns]
single_pieces = [king, queen]

def findPieces(img_in, img_out=None):
    for piece in pieces:
        for color in piece:
            result = cv.matchTemplate(img_in, color[0], cv.TM_CCOEFF_NORMED)

            
            width = color[0].shape[1]
            height = color[0].shape[0]

            locations = np.where( result >= color[1])
            locations = list(zip(*locations[::-1]))

            #rectangles are [x,y,w,h]
            rectangles = []
            for loc in locations:
                rect = [int(loc[0]), int(loc[1]), width, height]
                #done twice to make sure there will always be atleasty one overlap to stop groupRectangles from throwing them out
                rectangles.append(rect)
                rectangles.append(rect)

            rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)

            if len(rectangles):
                for (x, y, w, h) in rectangles:
                    top_left = (x,y)
                    bot_right = (x + w, y + h)
                    
                    cv.putText(img_in, color[2], top_left, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
                    cv.rectangle(img_in, top_left, bot_right , color=(0, 0, 255), thickness=2, lineType=cv.LINE_4)

    for kq in single_pieces:
        for piece in kq:
            result = cv.matchTemplate(img_in, piece[0], cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            

            width = piece[0].shape[1]
            height = piece[0].shape[0]

            top_left = max_loc
            bot_right = (max_loc[0]+ width, max_loc[1] + height)

            cv.putText(img_in, piece[2], top_left, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
            cv.rectangle(img_in, top_left, bot_right , color=(0, 0, 255), thickness=2, lineType=cv.LINE_4)

    #cv.imshow('Result', img_in)
    #cv.waitKey()

loop_time = time()

chess_size = (7,7)
while True:

    screenshot = np.array(pyautogui.screenshot())
    
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGBA2BGRA)
    board = cv.cvtColor(screenshot, cv.COLOR_BGRA2GRAY)


    board = np.float32(board)
    dst = cv.cornerHarris(board, 3, 3, 0.05)
    kernel= np.ones((2,2), np.uint8)
    dst = cv.dilate(dst, kernel, iterations=2)

    findPieces(screenshot)
    screenshot[dst>0.01*dst.max()] = [0,0,255, 0]

    cv.imshow('CV', screenshot)

    print('FPS {}'.format( 1 / (time() - loop_time)))
    loop_time = time()

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print("Exited")