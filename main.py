import numpy as np
import cv2 as cv

# Carregue a imagem em cores
img = cv.imread('bocha.jpg', cv.IMREAD_COLOR)

assert img is not None, "O arquivo não pôde ser lido, verifique com os.path.exists()"

# Converta a imagem para tons de cinza
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detecte os círculos na imagem
circles = cv.HoughCircles(
    gray,
    cv.HOUGH_GRADIENT,
    dp=1,            # Resolução inversa da matriz acumuladora
    minDist=20,      # Distância mínima entre os centros dos círculos
    param1=50,       # Limiar do detector de bordas
    param2=30,       # Limiar do acumulador para a detecção de círculos
    minRadius=0,
    maxRadius=0
)

#Aplique uma binarização para segmentar a bola branca da imagem
_, imagem_binaria = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

cv.imshow('Imagem binarizada', imagem_binaria)

#Encontre os contornos na imagem binarizada
contornos, _ = cv.findContours(imagem_binaria, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

white_ball = None
maior_area = 0

# Encontre o contorno com a maior área, que deve ser a bola branca
for contorno in contornos:
    area = cv.contourArea(contorno)
    if area > maior_area:
        maior_area = area
        white_ball = contorno

if white_ball is not None:
    # Desenhe o contorno da bola branca
    x, y, largura, altura = cv.boundingRect(white_ball)
    cv.rectangle(img, (x, y), (x + largura, y + altura), (255, 0, 0), 2)
 
    if circles is not None:
        # Converta o raio e o centro do círculo para inteiros
        circles = np.uint16(np.around(circles))
    
        closest_ball = None
        min_distance = float('inf')
        
        for i in circles[0, :]:
            # Desenhe o círculo externo
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Desenhe o centro do círculo
            cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

            if np.array_equal(white_ball, i):
                continue  # Pule o cálculo da distância se forem iguais

            # Calcule a distância entre os centros dos círculos usando numpy
            center1 = np.array([white_ball[0][0][0], white_ball[0][0][1]])    
            center2 = np.array([i[0], i[1]])
            distance = np.linalg.norm(center1 - center2)
                
            # Verifique se esta é a bola mais próxima até agora e se não é a própria bola branca que está sendo comparada
            if distance < min_distance and distance > 100:
                closest_ball = i
                min_distance = distance

        # Identifique a bola mais próxima
        if closest_ball is not None:
            cv.putText(img, "Closer ball", (closest_ball[0] - 50, closest_ball[1] - 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# Mostre a imagem com os círculos detectados
cv.imshow('Circulos detectados', img)
cv.waitKey(0)
cv.destroyAllWindows()
