import pygame
import time 
import os
import random
from pygame import *
import numpy as np
import operator

pygame.font.init()

WIDTH = 500
HEIGHT= 800

bird_imgs = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]
pipe_img= pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))
base_img =pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))
bg_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans",50)


class Bird:
	IMGS = bird_imgs
	MAX_ROTATION = 25
	ROT_VELOCITY = 20
	ANIMATION_TIME = 5

	def __init__(self,x,y):
		self.x=x
		self.y=y
		self.weights1 = []
		self.weights2 = []
		self.biases = []
		self.tilt =0
		self.tick_count = 0
		self.vel = 0
		self.height = self.y
		self.img_count = 0
		self.img = self.IMGS[0]
		self.score = 0
		self.fitness = 0

	def jump(self):
		self.vel = -11.5
		self.tick_count = 0
		self.height = self.y

	def move(self):
		self.tick_count+=1

		d = self.vel*self.tick_count+1.5*self.tick_count**2

		if d>=16:
			d = 16

		if d < 0:
			d-=2

		self.y =self.y + d

		if d<0 or self.y < self.height +50:
			if self.tilt < self.MAX_ROTATION:
				self.tilt = self.MAX_ROTATION
		else:
			if self.tilt > -90:
				self.tilt -= self.ROT_VELOCITY

	def draw(self,win):
		self.img_count += 1

		if self.img_count < self.ANIMATION_TIME:
			self.img = self.IMGS[0]
		elif self.img_count < self.ANIMATION_TIME * 2:
			self.img = self.IMGS[1]
		elif self.img_count < self.ANIMATION_TIME * 3:
			self.img = self.IMGS[2]
		elif self.img_count < self.ANIMATION_TIME * 4:
			self.img = self.IMGS[1]
		elif self.img_count < self.ANIMATION_TIME * 4+1:
			self.img = self.IMGS[0]
			self.img_count =0

		if self.tilt <= -80:
			self.img = self.IMGS[1]
			self.img_count = self.ANIMATION_TIME*2

		rotated_image = pygame.transform.rotate(self.img, self.tilt)
		new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x,self.y)).center)
		win.blit(rotated_image, new_rect.topleft)

	def get_mask(self):
		return pygame.mask.from_surface(self.img)

class Pipe:
	GAP = 200
	VEL = 5

	def __init__(self , x):
		self.x = x
		self.height = 0

		self.top = 0
		self.bottom = 0
		self.PIPE_TOP = pygame.transform.flip(pipe_img, False , True)
		self.PIPE_BOTTOM = pipe_img

		self.passed = False 
		self.set_height()

	def set_height(self):
		self.height = random.randrange(50,450)
		self.top = self.height - self.PIPE_TOP.get_height()
		self.bottom = self.height + self.GAP

	def move(self):
		self.x -= self.VEL


	def draw(self,win):
		win.blit(self.PIPE_TOP,(self.x,self.top))
		win.blit(self.PIPE_BOTTOM,(self.x,self.bottom))

	def collide(self, bird):
		bird_mask = bird.get_mask()
		top_mask = pygame.mask.from_surface(self.PIPE_TOP)
		bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

		top_offset= (self.x - bird.x, self.top -round(bird.y))
		bottom_offset = (self.x - bird.x,self.bottom- round(bird.y))

		bpoint = bird_mask.overlap(bottom_mask, bottom_offset)
		tpoint = bird_mask.overlap(top_mask, top_offset)

		if tpoint or bpoint:
			return True 

		return False

class Base:
	VEL = 5
	WIDTH = base_img.get_width()
	IMG = base_img

	def __init__(self,y):
		self.y = y
		self.x1 = 0
		self.x2 = self.WIDTH

	def move(self):
		self.x1 -= self.VEL
		self.x2 -= self.VEL

		if self.x1 +self.WIDTH < 0:
			self.x1 = self.x2 + self.WIDTH

		if self.x2 +self.WIDTH < 0:
			self.x2 = self.x1 +self.WIDTH

	def draw(self, win):
		win.blit(self.IMG, (self.x1, self.y))
		win.blit(self.IMG, (self.x2, self.y))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def initialize_bird(bird):
	weights1 = []
	weights2 = []
	weight = []

	for i in range(3):
		for j in range(4):
			weight.append(random.randint(-30,30))
		weights1.append(weight)
		weight =[]
	for i in range(4):
		weights2.append(random.randint(-30,30))

	bias = random.randint(-10,10)
	biases =[]
	for i in range(4):
		biases.append(bias)
	biases = np.asarray(biases)[np.newaxis]

	bird.weights1 = weights1
	bird.weights2 = weights2
	bird.biases = biases

def decide(bird,pipe_x,pipe_y):
	weights1 = bird.weights1
	weights2 = bird.weights2
	biases = bird.biases
	parameters = np.array([bird.y,pipe_x,pipe_y])[np.newaxis]
	weights2 = np.asarray(weights2)[np.newaxis]
	weights1 = np.asarray(weights1)
	first_pass = np.dot(weights1.T,parameters.T)
	second_pass= np.dot(weights2,first_pass+biases)
	return round(sigmoid(second_pass[0][0]))


def draw_window(win , birds, pipes, base,score ,gen):
	win.blit(bg_img,(0,0))
	for pipe in pipes:
		pipe.draw(win)

	text = STAT_FONT.render("Best Score: "+ str(score), 1, (255,255,255))
	win.blit(text, (WIDTH -10 -text.get_width(),10))

	text = STAT_FONT.render("Gen: "+ str(gen), 1, (255,255,255))
	win.blit(text, (10,10))
	base.draw(win);
	for bird in birds:
		bird.draw(win)
	pygame.display.update()



def mutate(bird,gen):
	lr = 2
	for weight in bird.weights1:
		for w in weight:
			w=w+(random.randint(-2,2)*lr)/(gen+1)
	for weight in bird.weights2:
		weight = weight+(random.randint(-2,2)*lr)/(gen+1)

	for bias in bird.biases:
		bias = bias+(random.randint(-2,2)*lr/(gen+1))
	return bird


def create_generation(birds,gen):
	sorted_birds = sorted(birds, key=operator.attrgetter("fitness"), reverse = True)
	best_birds = sorted_birds[:30]
	new_gen=[]
	for i,bird in enumerate(best_birds):
		best_birds[i].x = 230
		best_birds[i].y = 350
		best_birds[i].vel = 0
		best_birds[i].score = 0
		best_birds[i].tick_count = best_birds[i].img_count = 0
		new_gen.append(bird)
		if (i==0):
			additionalChild = mutate(bird,gen)
			new_gen.append(additionalChild)
		firstchild= mutate(bird,gen)
		secondchild = mutate(bird,gen)
		new_gen.append(firstchild)
		new_gen.append(secondchild)
	return new_gen




def simulate_game(birds,gen):
	print('playing these birds:', len(birds))
	for i in birds:
		print(i.score)


	base = Base(730)
	pipes = [Pipe(600)]
	win = pygame.display.set_mode((WIDTH,HEIGHT))
	clock = pygame.time.Clock()



	best_score =0
	grace = 6

	dead = []
	run=True
	while run:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

		pipe_ind =0

		if len(birds) > 0:
			if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
				pipe_ind = 1

		for bird in birds:
			bird.move()
			bird.fitness +=0.1
			if(decide(bird,abs(bird.y - pipes[pipe_ind].height),abs(bird.y - pipes[pipe_ind].bottom)) == 1):
				bird.jump()


		add_pipe=False
		rem = []
		
		for pipe in pipes:
			for bird in birds:
				if pipe.collide(bird):
					bird.fitness -=1
					dead.append(bird)
					birds.remove(bird)

					
				if not pipe.passed and pipe.x < bird.x:
					pipe.passed =True
					add_pipe= True

			if pipe.x + pipe.PIPE_TOP.get_width()< 0:
				rem.append(pipe)



			pipe.move()      

		if add_pipe:
			best_score+=1
			for bird in birds:
				bird.score+=1
				bird.fitness+=5
			pipes.append(Pipe(600))

		for r in rem:
			pipes.remove(r);

		for bird in birds:
			if (bird.y + bird.img.get_height() >= 730) | (bird.y + bird.img.get_height() <= 0 ):
				bird.fitness-=1
				dead.append(bird)
				birds.remove(bird)
				

		if(len(birds)==0):
			return create_generation(dead,gen)

		base.move()
		draw_window(win,birds,pipes ,base,best_score, gen)




def main():
	birds = []
	gen =0

	for i in range(100):
		birds.append(Bird(230,350))

	for bird in birds:
		initialize_bird(bird)


	while True:

		best_birds = simulate_game(birds,gen)
		gen+=1
		birds = best_birds



	pygame.quit()
	quit()

main()
