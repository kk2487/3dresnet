import os 
import random
train_list = open('data/1103/trainlist01.txt','w')
test_list = open('data/1103/testlist01.txt','w')
#folders = ['blink','normal','test','test1']
folders = ['blink','normal']
for fold in folders:
	file_path = '/home/rvl/KaiChun/CEO/Eye_Classfication/traindata/1013_eye/'+fold
	files = os.listdir(file_path)
	for file in files :
		print(file_path+'/'+file+'/n_frames')
		if os.path.isfile(file_path+'/'+file+'/n_frames'):
			os.remove(file_path+'/'+file+'/n_frames')
			print('remove'+file_path+'/'+file+'/n_frames')
	for file in files :
		if fold == 'blink' :
			if random.randint(0,10) >= 1 :
				test_list.write(fold+'/'+file+'.avi 1\n')			
			else :
				train_list.write(fold+'/'+file+'.avi 1\n')
		elif fold == 'normal' :
			if(random.randint(0,100)<=20):
				if random.randint(0,10) >= 1 :
					test_list.write(fold+'/'+file+'.avi 2\n')			
				else :
					train_list.write(fold+'/'+file+'.avi 2\n')
		images = os.listdir(file_path+'/'+file)
		min_val = 10000
		for image in images :
			num = int(image[6:len(image)-4])
			if num < min_val :
				min_val = num
		min_val -= 1
		for image in images :
			#print(image)
			zero = ''
			num = int(image[6:len(image)-4])
			for i in range(0,5-len(str(num))):
				zero += '0'
			#print(file_path+'/'+file+'/'+image[0:6]+zero+str(int(image[6:len(image)-4])-min_val)+'.jpg')
			os.rename(file_path+'/'+file+'/'+image,file_path+'/'+file+'/'+image[0:6]+zero+str(int(image[6:len(image)-4])-min_val)+'.jpg')

		#for i in range(len(1,images)):
			#os.rename(file_path+'/'+file+'/'+'image_0000'+str(i)+'.jpg',file_path+'/'+file+'/'+image[0:6]+zero+str(i)+'.jpg')