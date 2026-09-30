import numpy as np
import cv2 as cv

############################################
#EXPLICACAO:
# esse código pega a imagem da câmera
# do computador, reconhece onde tá
# o seu rosto e coloca uma imagem
# por cima.
############################################





# !!!! RECOMENDO LER ESSA FUNCAO DPS DE LER O LOOP PRINCIPAL !!!!

#coloca imagem na frente do rosto
def sobrepor(frame, img, x, y, w, h):
    # garante q a imagem vai ficar dentro do frame da camera
    frame_h, frame_w = frame.shape[:2]
    x1, y1 = max(x, 0), max(y, 0) #inicio do retangulo
    x2, y2 = min(x + w, frame_w), min(y + h, frame_h) #fim do retangulo
    if x1 >= x2 or y1 >= y2:
        return

    # muda o tamanho da imagem q vai sobrepor pro tamanho do retangulo de onde está o rosto
    img = cv.resize(img, (w, h))

    """
      troca os pixels da camera pelos pixels da imagem (fazendo com que "sobreponha")

      lembrem-se que imagens na vdd são só uma matriz dos pixels, entao basta trocar a cor de cada pixel pra fazer a imagem
      parecer que está sobrepondo.

      lembrem tb a sintaxe do numpy: quando escrevemos array[x1:x2] é o mesmo que escrever 'para todo valor do array indo de array[x1] até array[x2], faça tal coisa'
                                                                                                    (e escrever apenas [:x] é igual a [0:x] - do inicio do array até x)

      nesse caso seria: para todo valor no início de onde está o rosto (x1, y1) ate o fim de onde está o rosto (x2,y2), troque o pixel para um pixel da imagem q vai sobrepor
    """
    frame[y1:y2, x1:x2] = img[y1 - y:y2 - y, x1 - x:x2 - x]





# ----- SETUP ----- #

#medidas de tamanho da janela q vai abrir
CAM_WIDTH = 1280
CAM_HEIGHT = 720


#serve pra capturar as imagens da camera.
#pra videos, o opencv trata eles como
# um monte de imagens (frames)
# e aplica as mudanças frame a frame
cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)



# isso aqui é usado pra detectar rostos. nao vai cair na prova,
# mas eu achei q seria divertido usar !!!
detector = cv.FaceDetectorYN.create(
    './assets/face_detection_yunet_2026may.onnx',
    '',
    (CAM_WIDTH, CAM_HEIGHT),
    score_threshold=0.8,
)

#imagem que vai cobrir o rosto!!
imagem_que_tampa = cv.imread('./assets/hampster_happy.jpg')





# ----- MAIN LOOP ----- #
while True:
    #fica pegando cada frame da camera (como explicado la em cima)
    ret, frame = cam.read()

    #detecta rostos no frame (e retorna um retangulo com coordenadas de onde ele achou o rosto)
    _, faces = detector.detect(frame)

    # se encontrar pelo menos 1 rosto na imagem
    if faces is not None:
        # pra cada rosto na imagem (pode ter mais de 1)
        for face in faces:

            #pega as medidas
            x, y, w, h = face[:4].astype(int)

            #desenha retangulo na area do rosto
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

            #bota uma imagem na frente do rosto
            sobrepor(frame, imagem_que_tampa, x, y, w, h)

    #mostra resultado final
    cv.imshow('cam', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()





