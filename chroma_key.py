import cv2 as cv
import numpy as np

frente = cv.imread('assets/rabbit_green.png')
fundo = cv.imread('assets/jpg')

# O fundo precisa ter o mesmo tamanho da imagem da frente
altura, largura = frente.shape[:2]
fundo = cv.resize(fundo, (largura, altura))

# 1. Máscara do verde (em HSV o verde fica numa faixa estreita de matiz)
hsv = cv.cvtColor(frente, cv.COLOR_BGR2HSV)
verde_min = np.array([40, 150, 60])
verde_max = np.array([85, 255, 255])
mascara = cv.inRange(hsv, verde_min, verde_max)  # 255 = verde, 0 = coelho

# 2. Erosão + dilatação (abertura) remove pontos verdes soltos dentro do coelho;
#    dilatação + erosão (fechamento) fecha buracos no verde do fundo
kernel = np.ones((5, 5), np.uint8)
mascara = cv.erode(mascara, kernel)
mascara = cv.dilate(mascara, kernel)
mascara = cv.dilate(mascara, kernel)
mascara = cv.erode(mascara, kernel)

# 3. Substituição: onde a máscara é verde, usa o pixel do fundo
resultado = frente.copy()
resultado[mascara == 255] = fundo[mascara == 255]

cv.imshow('mascara', mascara)
cv.imshow('chroma key', resultado)
cv.waitKey(0)
cv.destroyAllWindows()
