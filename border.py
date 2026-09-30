import cv2 as cv
from google.colab.patches import cv2_imshow
import numpy as np

img = cv.imread("./assets/eagle_mog.png")
video = cv.VideoCapture("./assets/eagle.mp4")
fps = video.get(cv.CAP_PROP_POS_FRAMES)
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

codec = cv.VideoWriter_fourcc(*'mp4v')
gravador = cv.VideoWriter(
    "aguia_lobotomizada.mp4", codec, fps, (width, height)
)

if video.isOpened() == False:
  print("Video nao abriu")

while(video.isOpened()):
  ret, frame = video.read()
  
  if ret == True:
    frame_cinza = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    frame_blackNwhite = cv.threshold(frame_cinza, 50, 255, cv.THRESH_BINARY_INV)[1]

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    frame_sem_ruido = cv.morphologyEx(frame_blackNwhite, cv.MORPH_OPEN, kernel)
    frame_sem_buraco = cv.morphologyEx(frame_sem_ruido, cv.MORPH_CLOSE, kernel)

    contornos, _ = cv.findContours(frame_sem_buraco, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    result = frame.copy()
    contorno = max(contornos, key=cv.contourArea)
    box = cv.contourArea(contorno)
    x, y, box_width, box_height = cv.boundingRect(contorno)

    cv.rectangle(
        result,
        (x, y),
        (x + box_width, y + box_height),
        (0, 255, 0),
        4
    )
    eagle_face = cv.resize(img, (box_width, box_height))

    result_filtrado = cv.bitwise_not(result)
    result_filtrado[y : y + box_height, x : x + box_width] = eagle_face

    #gravador.write(result)
    cv2_imshow(result_filtrado)

    if cv.waitKey(25) & 0xFF == ord('q'):
      break

  else:
    break

video.release()
gravador.release()
cv.destroyAllWindows()
