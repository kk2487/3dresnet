import os
dirs = os.listdir('traindata/1013/')
for adir in dirs :
	files = os.listdir('traindata/1013/'+adir+'/')
	for file in files :
		os.rename('traindata/1013/'+adir+'/'+file,'traindata/1013/'+adir+'/'+adir+'_'+file)