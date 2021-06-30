#!/usr/bin/python
'''
zip -r fmriprep_TATA_task.zip ../nabaruns.github.io/img/fmriprep_TATA_task/

'''
import os,sys,json
# if sys.version[0]=='3':
#     raw_input = lambda(x): input(x)

##############################
#Owner information goes here!#
##############################
name = 'corr'
client_id= '38449805356-k5t4f7l8ijct0pvoc8of0eski7aecgtb.apps.googleusercontent.com'
client_secret='rvJ3kLoYoYXfWBD-UQFA-AC7'

##############################

cmd1 = json.loads(os.popen('curl -d "client_id=%s&scope=https://www.googleapis.com/auth/drive.file" https://oauth2.googleapis.com/device/code'%client_id).read())
str(input('\n Enter %(user_code)s\n\n at %(verification_url)s \n\n Then hit Enter to continue.'%cmd1))
str(input('(twice)'))
cmd2 = json.loads(os.popen(('curl -d client_id=%s -d client_secret=%s -d device_code=%s -d grant_type=urn~~3Aietf~~3Aparams~~3Aoauth~~3Agrant-type~~3Adevice_code https://accounts.google.com/o/oauth2/token'%(client_id,client_secret,cmd1['device_code'])).replace('~~','%')).read())
print(cmd2)# zip files
cmd3 = os.popen('zip -r %s.zip %s'%(name,' '.join(sys.argv[1:]))).read
print(cmd3)
cmd4 = os.popen('''
curl -X POST -L \
    -H "Authorization: Bearer %s" \
    -F "metadata={name :\'%s\'};type=application/json;charset=UTF-8" \
    -F "file=@%s.zip;type=application/zip" \
    "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    '''%(cmd2["access_token"],name,name)).read()
print(cmd4)
print('end')
