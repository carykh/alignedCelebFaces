# import the pygame module
import pygame

import tensorflow as tf
import numpy as np
from scipy import misc
import random
import math
import html

eigenvalues = np.load("eigenvalues.npy")
eigenvectors = np.load("eigenvectors.npy")
eigenvectorInverses = np.linalg.pinv(eigenvectors)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_COUNT = 13016
DENSE_SIZE = 300
learning_rate = 0.0002  # Used to be 0.001
settings = np.zeros((DENSE_SIZE))
approachSettings = np.zeros((2))
approachSettings.fill(1)
denseData = np.load("denseArray27K.npy")
shouldICalculateImage = True

f = open('names/allNames.txt', 'r+')
allNames = f.read()
f.close()
allPeople = html.unescape(allNames).split('\n')
f = open('eigenvalueNames.txt', 'r+')
eigenvalueNames = f.read().split('\n')
f.close()
nearestPerson = 0

meanData = denseData.mean(axis=0)

inputs_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, 3), name='targets')




### Encoder
conv0 = tf.layers.conv2d(inputs=inputs_, filters=120, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 64x64x25
maxpool0 = tf.layers.max_pooling2d(conv0, pool_size=(2,2), strides=(2,2), padding='same')
# Now 32x32x25
conv1 = tf.layers.conv2d(inputs=maxpool0, filters=160, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 32x32x40
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now 16x16x40
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=200, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 16x16x60
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# Now 8x8x60
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=240, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 8x8x80
maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
# Now 4x4x80

maxpool3_flat = tf.reshape(maxpool3, [-1,4*4*240])

W_fc1 = weight_variable([4*4*240, 300])
b_fc1 = bias_variable([300])
tesy = tf.matmul(maxpool3_flat, W_fc1)
encoded = tf.nn.relu(tf.matmul(maxpool3_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([300, 4*4*240])
b_fc2 = bias_variable([4*4*240])
predecoded_flat = tf.nn.relu(tf.matmul(encoded, W_fc2) + b_fc2)

predecoded = tf.reshape(predecoded_flat, [-1,4,4,240])

### Decoder
upsample1 = tf.image.resize_images(predecoded, size=(8,8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 8x8x80
conv4 = tf.layers.conv2d(inputs=upsample1, filters=200, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 8x8x60
upsample2 = tf.image.resize_images(conv4, size=(16,16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 16x16x60
conv5 = tf.layers.conv2d(inputs=upsample2, filters=160, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 16x16x40
upsample3 = tf.image.resize_images(conv5, size=(32,32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 32x32x40
conv6 = tf.layers.conv2d(inputs=upsample3, filters=120, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 32x32x25
upsample4 = tf.image.resize_images(conv6, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 64x64x25
conv7 = tf.layers.conv2d(inputs=upsample4, filters=15, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 64x64x10


logits = tf.layers.conv2d(inputs=conv7, filters=3, kernel_size=(3,3), padding='same', activation=None)
#Now 64x64x1

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess,  "models/model27674.ckpt")

def getCelebSliders(i):
    traits = denseData[i]-meanData
    return np.matmul(traits,eigenvectorInverses)/eigenvalues

celebSliders = np.zeros(denseData.shape)
for i in range(denseData.shape[0]):
     celebSliders[i] = getCelebSliders(i)

def calculateImage(settings):
    realSettings = meanData.copy()
    for i in range(DENSE_SIZE):
        realSettings += settings[i]*eigenvalues[i]*eigenvectors[i]
    realSettings = realSettings.reshape((1,DENSE_SIZE))
    reconstructedImage = sess.run([decoded], feed_dict={encoded: realSettings})
    ri_np = np.array(reconstructedImage).reshape((64,64,3))
    ri_np = np.swapaxes(ri_np,0,1)
    closestCeleb = np.argmin(np.linalg.norm((settings-celebSliders)*eigenvalues,axis=1))
    return ri_np*255, closestCeleb

def createSpecialChildren(index, parentCount, fileName):
    totalImage = np.zeros((240+264*parentCount,300,3))
    totalImage.fill(255)

    parents = [-1]*parentCount

    childSettings = np.zeros((celebSliders[0].shape))
    for i in range(0,parentCount):
        while parents[i] == -1 or int(allPeople[parents[i]].split(",")[1]) > 10: # fame rank must be 5 or better
            parents[i] =  np.random.randint(IMAGE_COUNT)
        parents[i] = 13015
        childSettings += celebSliders[parents[i]]
    childSettings /= parentCount

    for i in range(0,parentCount+1):
        img = np.zeros((192,192,3))
        if i == parentCount:
            img, _ = calculateImage(childSettings)
        else:
            img, _ = calculateImage(celebSliders[parents[i]])
            #img = np.swapaxes(misc.imread("data/dataFace"+str(parents[i])+".png"),0,1)

        totalImage[24+i*264:216+i*264,24:216] = misc.imresize(img,size=[192,192],interp='nearest')

    blah = pygame.surfarray.make_surface(totalImage)
    for i in range(0,parentCount+1):
        name = "CHILD"
        if i < parentCount:
            name = allPeople[parents[i]].split(",")[0]
        font = pygame.font.SysFont("Helvetica", 22)
        textSurface = font.render(name, 1, (0,0,0))
        blah.blit(textSurface, [24+264*i,220])
        if i < parentCount-1:
            font = pygame.font.SysFont("Helvetica", 48)
            blah.blit(font.render('^', 1, (0,0,0)), [240+264*i,100])
    font = pygame.font.SysFont("Helvetica", 48)
    blah.blit(font.render('=', 1, (0,0,0)), [240+264*(parentCount-1),100])
    pygame.image.save(blah,"spesh/"+fileName+"{:03d}".format(index)+".png")

def createChildGrid(index, parentCount, fileName):
    totalImage = np.zeros((264+264*parentCount,264+264*parentCount,3))
    totalImage[264:,264:,:] = 255

    parents = [-1]*parentCount
    for i in range(0,parentCount):
        parents[i] =  np.random.randint(IMAGE_COUNT)
        img, _ = calculateImage(celebSliders[parents[i]])
        #img = np.swapaxes(misc.imread("data/dataFace"+str(parents[i])+".png"),0,1)
        big_img = misc.imresize(img,size=[192,192],interp='nearest')
        totalImage[24+(i+1)*264:216+(i+1)*264,24:216] = big_img
        totalImage[24:216,24+(i+1)*264:216+(i+1)*264] = big_img
        totalImage[264*(i+1):264*(i+2),264*(i+1):264*(i+2)] = [0,255,0]
    
    childSettings = np.zeros((parentCount,parentCount,celebSliders[0].shape[0]))
    for i in range(0,parentCount):
        for j in range(0,parentCount):
            childSettings = (celebSliders[parents[i]]+celebSliders[parents[j]])/2
            img, _ = calculateImage(childSettings)
            totalImage[24+(i+1)*264:216+(i+1)*264,24+(j+1)*264:216+(j+1)*264] = misc.imresize(img,size=[192,192],interp='nearest')
    blah = pygame.surfarray.make_surface(totalImage)
    for i in range(0,parentCount):
        name = allPeople[parents[i]].split(",")[0]
        font = pygame.font.SysFont("Helvetica", 22)
        textSurface = font.render(name, 1, (255,255,255))
        blah.blit(textSurface, [24+264*(i+1),220])
        blah.blit(textSurface, [24,220+264*(i+1)])
    pygame.image.save(blah,"spesh/"+fileName+"{:03d}".format(index)+".png")


def createFamilyTree(index, parentCount, fileName):
    totalImage = np.zeros((264*parentCount,264*parentCount,3))
    totalImage.fill(255)
    
    parents = [-1]*parentCount

    allSettings = np.zeros((parentCount,parentCount,celebSliders[0].shape[0]))
    for i in range(0,parentCount):
        parents[i] =  np.random.randint(IMAGE_COUNT)
        allSettings[0,i] = celebSliders[parents[i]]
        img, _ = calculateImage(celebSliders[parents[i]])
        #img = np.swapaxes(misc.imread("data/dataFace"+str(parents[i])+".png"),0,1)
        big_img = misc.imresize(img,size=[192,192],interp='nearest')
        totalImage[24+i*264:216+i*264,40:232] = big_img

    for level in range(1, parentCount):
        for i in range(0,parentCount-level):
            allSettings[level,i] = (allSettings[level-1,i]+allSettings[level-1,i+1])*0.5
            img, _ = calculateImage(allSettings[level,i])
            xStart = 24+i*264+level*132
            yStart = 40+level*264
            totalImage[xStart:xStart+192,yStart:yStart+192] = misc.imresize(img,size=[192,192],interp='nearest')
            totalImage[xStart+92:xStart+100,yStart-32:yStart] = 0
            totalImage[xStart:xStart+192,yStart-40:yStart-32] = 0
            totalImage[xStart:xStart+8,yStart-72:yStart-40] = 0
            totalImage[xStart+184:xStart+192,yStart-72:yStart-40] = 0

    blah = pygame.surfarray.make_surface(totalImage)
    for i in range(0,parentCount):
        name = allPeople[parents[i]].split(",")[0]
        font = pygame.font.SysFont("Helvetica", 22)
        textSurface = font.render(name, 1, (0,0,0))
        blah.blit(textSurface, [20+264*i,14])
    pygame.image.save(blah,"spesh/"+fileName+"{:03d}".format(index)+".png")
    

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)
BLUE = (50, 50, 255)
GREY = (200, 200, 200)
ORANGE = (200, 100, 50)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
TRANS = (1, 1, 1)

VISIBLE_COMPONENTS = 20

enteringName = False
isShiftPressed = False
enteredName = ""
frameTimer = 0
misspelledTimer = 0
scrollPosition = 0
stringToNameDict = {}

transitionTimes = np.zeros((2))
transitionKeyFrames = np.zeros((2,DENSE_SIZE))

for i in range(len(allPeople)):
    line = allPeople[i]
    pieces = line.split(",")
    name = pieces[0]
    alphaOnlyName = ''.join(x for x in name if (x.isalpha() or x == ' '))
    lowerName = alphaOnlyName.lower()
    if len(lowerName) >= 1:
        stringToNameDict[lowerName] = i

def stringToCeleb(st):
    alphaOnlyName = ''.join(x for x in st if (x.isalpha() or x == ' '))
    lowerName = alphaOnlyName.lower()
    if lowerName not in stringToNameDict:
        return -1
    return stringToNameDict[lowerName]
    

oopsImage = pygame.image.load("oops.png")
imagerect = oopsImage.get_rect()

calculatedImage, nearestPerson = calculateImage(settings)
from pygame.locals import *

class Slider():
    def __init__(self, i, maxi, mini, x, y, w, h):
        self.maxi = maxi
        self.mini = mini
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.surf = pygame.surface.Surface((w,h))
        self.hit = False
        self.i = i 

    def trueI(self):
        return self.i+scrollPosition

    def draw(self):
        j = self.trueI()
        eigen = "%.4f" % eigenvalues[j]
        name = "PCA #"+str(self.trueI()+1)+" ("+eigen+")"
        if j < len(eigenvalueNames)-1:
            name = eigenvalueNames[j]

        self.font = pygame.font.SysFont("Helvetica", 16)
        self.txt_surf = self.font.render(name, 1, WHITE)
        self.txt_rect = self.txt_surf.get_rect(center = (self.w/2,13))

        s = 70
        if self.i%2+(self.i//2)%2 == 1:
            s = 100
        self.surf.fill((s,s,s))
        pygame.draw.rect(self.surf, (220,220,220), [10,30,self.w-20,5], 0)
        for g in range(7):
            pygame.draw.rect(self.surf, (s+50,s+50,s+50), [9+(self.w-20)/6*g,40,2,5], 0)


        self.surf.blit(self.txt_surf, self.txt_rect)

        self.button_surf = pygame.surface.Surface((10,20))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.rect(self.button_surf, WHITE, [0,0,10,20])

        surf = self.surf.copy()

        v = min(max(settings[j],-9999),9999)
        pos = (10+int((v-self.mini)/(self.maxi-self.mini)*(self.w-20)), 33)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.x, self.y)

        screen.blit(surf, (self.x, self.y))

    def move(self):
        j = self.trueI()
        settings[j] = (pygame.mouse.get_pos()[0] - self.x - 10) / 130 * (self.maxi - self.mini) + self.mini
        if settings[j] < self.mini:
            settings[j] = self.mini
        if settings[j] > self.maxi:
            settings[j] = self.maxi

class ApproachSlider():
    def __init__(self, i, maxi, mini, x, y, w, h):
        self.i = i
        self.maxi = maxi
        self.mini = mini
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.surf = pygame.surface.Surface((w,h))
        self.hit = False

    def draw(self):
        self.font = pygame.font.SysFont("Helvetica", 16)
        if self.i == 0:
            st = "Go "+"%.1f" % (100*approachSettings[self.i])+"% the way to this celeb."
        else:
            st = "Speed of travel: "+"%.2f" % (100*(1-approachSettings[self.i]))+" frames"
        self.txt_surf = self.font.render(st, 1, WHITE)
        self.txt_rect = self.txt_surf.get_rect(center = (self.w/2,13))
        
        s = 70+30*self.i
        self.surf.fill((s,s,s))
        pygame.draw.rect(self.surf, (220,220,220), [10,35,self.w-20,5], 0)
        self.surf.blit(self.txt_surf, self.txt_rect)

        self.button_surf = pygame.surface.Surface((10,30))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.rect(self.button_surf, WHITE, [0,0,10,30])

        surf = self.surf.copy()

        v = min(max(approachSettings[self.i],-9999),9999)
        pos = (10+int((v-self.mini)/(self.maxi-self.mini)*(self.w-20)), 38)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.x, self.y)

        screen.blit(surf, (self.x, self.y))

    def move(self):
        approachSettings[self.i] = (pygame.mouse.get_pos()[0] - self.x - 10) / (self.w-20) * (self.maxi - self.mini) + self.mini
        if approachSettings[self.i] < self.mini:
            approachSettings[self.i] = self.mini
        if approachSettings[self.i] > self.maxi:
            approachSettings[self.i] = self.maxi

def drawButtons():
    enb_shade = 200
    if enteringName:
        enb_shade = math.sin(frameTimer*0.06)*40+200

    enterNameButton = pygame.surface.Surface((300,120))
    pygame.draw.rect(enterNameButton, (enb_shade,enb_shade,enb_shade), [5,5,290,110], 0)
    font = pygame.font.SysFont("Helvetica", 30)

    st = "[Enter celeb name]"
    if len(enteredName) >= 1:
        st = enteredName

    buttonData = [
    [(0,0),(300,60),(230,30,30),44,"RANDOMIZE",WHITE],
    [(0,540),(300,60),(30,30,230),44,"GO TO MEAN",WHITE],
    [(800,0),(300,120),(230,170,30),44,"INVERT",WHITE],
    [(300,500),(500,100),(0,0,0),24,"Hey! You look like "+allPeople[nearestPerson].split(",")[0]+".",WHITE],
    [(800,120),(300,120),(enb_shade,enb_shade,enb_shade),30,st,BLACK],
    [(800,360),(300,120),(30,170,30),44,"GO TO THEM",WHITE],
    [(800,480),(300,120),(30,170,30),24,"GO TO RANDOM CELEB",WHITE]]

    for button in buttonData:
        buttonSurface = pygame.surface.Surface(button[1])
        pygame.draw.rect(buttonSurface, button[2], [5,5,button[1][0]-10,button[1][1]-10], 0)
        font = pygame.font.SysFont("Helvetica", button[3])
        b_text = font.render(button[4], 1, button[5])
        b_text_rect = b_text.get_rect(center = (button[1][0]/2,button[1][1]/2))
        buttonSurface.blit(b_text, b_text_rect)
        screen.blit(buttonSurface, button[0])

    if transitionTimes[0] >= 0:
        w = 290*(frameTimer-transitionTimes[0])/(transitionTimes[1]-transitionTimes[0])
        progressBarSurface = pygame.surface.Surface((w,25))
        progressBarSurface.fill((0,150,0))
        screen.blit(progressBarSurface, (805,125))

    image_surface = pygame.surfarray.make_surface(calculatedImage)
    bigger = pygame.transform.scale(image_surface,(500,500))
    screen.blit(bigger,(300,0))

    if misspelledTimer >= 1:
        y = 5
        if misspelledTimer < 60:
            y = -115+120*(0.5+math.cos((misspelledTimer-60)/60.0*math.pi)*0.5)
        screen.blit(oopsImage, (805,y))

# return misspelledTimer. how many frames the misspelled warning should show up. I know, it's weird and dumb.
def goToCeleb(c):
    celebChoice = stringToCeleb(enteredName)
    if c >= 0:
        celebChoice = c
    if celebChoice == -1:
        return 800
    else:
        sliderSettings = celebSliders[celebChoice]
        if approachSettings[1] == 1:
            for i in range(DENSE_SIZE):
                settings[i] += (sliderSettings[i]-settings[i])*approachSettings[0]
        else:
            transitionKeyFrames[0] = settings.copy()
            transitionKeyFrames[1] = settings.copy()
            for i in range(DENSE_SIZE):
                transitionKeyFrames[1,i] += (sliderSettings[i]-settings[i])*approachSettings[0]
            transitionTimes[0] = frameTimer-1
            transitionTimes[1] = frameTimer-1 + 100*(1-approachSettings[1]) # really bad magic numbers oh well
    return 0

pygame.init()
slides = []
for i in range(VISIBLE_COMPONENTS):
    eigen = "%.4f" % eigenvalues[i]
    slides.append(Slider(i,3,-3,(i%2)*150,(i//2)*48+60,150,48))

approachSlides = []
for i in range(2):
    approachSlides.append(ApproachSlider(i,1,0,800,240+60*i,300,60))

screen = pygame.display.set_mode((1100, 600))

running = True

# OPTIONAL SPECIAL CHILD CREATION
#createSpecialChildren(0,1,"speshCaryD")
#for i in range(0,2):
#    createSpecialChildren(i,2,"speshTwo")
#createChildGrid(0,6,"speshGrid")
#createFamilyTree(0,12,"speshFamilyHuge")
# END OF OPTIONAL SPECIAL CHILD CREATION

while running:
    shouldICalculateImage = False
    frameTimer += 1
    misspelledTimer = max(0,misspelledTimer-1)
    for event in pygame.event.get():
        # Check for KEYDOWN event; KEYDOWN is a constant defined in pygame.locals, which we imported earlier
        if event.type == KEYDOWN:
            # If the Esc key has been pressed set running to false to exit the main loop
            if event.key == K_LSHIFT or event.key == K_RSHIFT:
                isShiftPressed = True
            elif event.key == K_ESCAPE:
                running = False
            elif enteringName:
                k = event.key
                isLetter = k >= ord('a') and k <= ord('z')
                if isLetter or k == ord('-') or k == ord(' ') or k == ord('\''):
                    ch = chr(k)
                    if isShiftPressed and isLetter:
                        ch = chr(k-32)
                    enteredName = enteredName+ch
                if len(enteredName) >= 1 and (k == K_BACKSPACE or k == K_DELETE):
                    enteredName = enteredName[0:-1]
                if k == K_RETURN:
                    enteringName = False
                    misspelledTimer = goToCeleb(-1)
                    shouldICalculateImage = True
        # Check for QUIT event; if QUIT, set running to false
        elif event.type == KEYUP:
            if event.key == K_LSHIFT or event.key == K_RSHIFT:
                isShiftPressed = False
        elif event.type == QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_loc = pygame.mouse.get_pos()
            if event.button == 4 or event.button == 5:
                dire = (event.button-4.5)*2
                if mouse_loc[0] < 300 and mouse_loc[1] >= 60 and mouse_loc[1] < 540:
                    i = (mouse_loc[0]//150)+((mouse_loc[1]-60)//48)*2+scrollPosition
                    settings[i] -= 0.2*dire
                    shouldICalculateImage = True
                else:
                    scrollPosition = min(max(scrollPosition+2*int(dire),0),denseData.shape[1]-VISIBLE_COMPONENTS)
                    for i in range(VISIBLE_COMPONENTS):
                        slides[i].val = settings[i+scrollPosition]
            else:
                enteringName = False
                if mouse_loc[0] < 300:
                    if mouse_loc[1] < 60:
                        for i in range(DENSE_SIZE):
                            settings[i] = np.random.normal(0,1,1)
                        shouldICalculateImage = True
                        enteredName = ""
                    elif mouse_loc[1] >= 540:
                        for i in range(DENSE_SIZE):
                            settings[i] = 0
                        shouldICalculateImage = True
                        enteredName = ""
                    else:
                        i = (mouse_loc[0]//150)+((mouse_loc[1]-60)//48)*2
                        slides[i].hit = True
                elif mouse_loc[0] >= 800:
                    if mouse_loc[1] < 120:
                        for i in range(DENSE_SIZE):
                            settings[i] *= -1
                        shouldICalculateImage = True
                        misspelledTimer = 0
                        enteredName = ""
                    elif mouse_loc[1] < 240:
                        enteringName = True
                        misspelledTimer = 0
                        enteredName = ""
                    elif mouse_loc[1] >= 240 and mouse_loc[1] < 360:
                        i = ((mouse_loc[1]-240)//60)
                        approachSlides[i].hit = True
                    elif mouse_loc[1] >= 480:
                        c = np.random.randint(denseData.shape[0])
                        goToCeleb(c)
                        shouldICalculateImage = True
                        enteredName = allPeople[c].split(",")[0]
                    elif mouse_loc[1] >= 360:
                        misspelledTimer = goToCeleb(-1)
                        shouldICalculateImage = True

        elif event.type == pygame.MOUSEBUTTONUP:
            for s in slides:
                s.hit = False
            for a_s in approachSlides:
                a_s.hit = False
    
    if transitionTimes[0] >= 0:
        proportionThrough = min(max((frameTimer-transitionTimes[0])/(transitionTimes[1]-transitionTimes[0]),0),1)
        if frameTimer >= transitionTimes[1]:
            proportionThrough = 1
            transitionTimes[:] = -1

        settings = transitionKeyFrames[0] + proportionThrough*(transitionKeyFrames[1]-transitionKeyFrames[0])
        shouldICalculateImage = True
    else:
        for s in slides:
            if s.hit:
                s.move()
                shouldICalculateImage = True
    for a_s in approachSlides:
        if a_s.hit:
            a_s.move()

    if shouldICalculateImage:
        calculatedImage, nearestPerson = calculateImage(settings)

    screen.fill(BLACK)
    for s in slides:
        s.draw()
    for a_s in approachSlides:
        a_s.draw()
    drawButtons()

    pygame.display.flip()
