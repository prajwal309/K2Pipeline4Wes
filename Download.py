#importing the libraries
import pandas as pd
import glob
import subprocess
import os

StatusFile = open('ExtractStatus.txt','a')
CSVFiles = glob.glob('*.csv')
print (CSVFiles)

def FileCheck(filename):
    #check if the file is already downloaded
    FitsFile = glob.glob('*.fits')
    return filename[:-3] in FitsFile

def DownloadFile(EPICID,OutputFile,CampaignStr,StatusFile):
    if not(FileCheck(OutputFile)):
        FirstFolder = str(EPICID-EPICID%1e5)[0:9]
        SecondFolder = str(EPICID-EPICID%1e3)[4:9]
        #CampaignStr='c15'
        URL = ('https://archive.stsci.edu/missions/k2/target_pixel_files/%s/%s/%s/%s' %(CampaignStr,FirstFolder,SecondFolder,OutputFile))
        print ("The URL is ", URL)
        if not(os.path.exists(OutputFile)):
            command = ('curl -f -R -o %s %s' %(OutputFile,URL))
            os.system(command)
            Status = os.system('gunzip %s' %(OutputFile))
            if Status==0:
                StatusFile.write('%s Successful \n' %(OutputFile))
            else:
                StatusFile.write('%s Failed \n' %(OutputFile))
        else:
            print ("The file already exists")
            StatusFile.write('%s Already Exists \n' %(OutputFile))
    else:
         print ("File Already present")
         pass

for file in CSVFiles:
    CSVFile = pd.read_csv(file,skiprows=1,names=['EPICID','RA', 'DEC', 'MAG','INVESTID'])
    #Extract information from Investigation ID

    for i in range(len(CSVFile['EPICID'])):
        EPICID = CSVFile['EPICID'][i]
        ProposalID = CSVFile['INVESTID'][i].split("|")[0]
        if len(file) == 19:
            Campaign = int(file[2:4])
        else:
            Campaign = int(file[2])
        #to avoid the headers
        try:
            EPICID = int(EPICID)
        except:
            continue

        #see if short cadence data is available
        SC_Flag = False
        if "SC" in CSVFile['INVESTID'][i]:
            SC_Flag = True

        if Campaign>8 and Campaign<12:
            #Two separate files for Campaign9 and Campaign10
            for j in [1,2]:
                CampaignStr = 'c'+str(Campaign)+str(j)
                OutputFile = ('ktwo%s-%s_llc.fits' %(EPICID,CampaignStr))
                if not(os.path.exists(OutputFile)):
                    DownloadFile(EPICID,OutputFile,CampaignStr,StatusFile)
                else:
                    print ("The file already exists")


            if SC_Flag:
                for j in [1,2]:
                    CampaignStr = 'c14'
                    OutputFile = ('ktwo%s-%s_spd-targ.fits.gz' %(EPICID,CampaignStr))
                    if not(os.path.exists(OutputFile)):
                        DownloadFile(EPICID,OutputFile,CampaignStr,StatusFile)
                    else:
                        print ("The file already exists")

        else:
            if Campaign<2:
                CampaignStrTemp = 'c0'+str(Campaign)
            else:
                CampaignStrTemp = 'c'+str(Campaign)
            CampaignStr = 'c'+str(Campaign)
            OutputFile = ('ktwo%s-%s_lpd-targ.fits.gz' %(EPICID,CampaignStrTemp))
            if not(os.path.exists(OutputFile)):
                DownloadFile(EPICID,OutputFile,CampaignStr,StatusFile)
            else:
                print ("The file already exists")
            if SC_Flag:
                OutputFile = ('ktwo%s-%s_spd-targ.fits.gz' %(EPICID,CampaignStrTemp))
                if not(os.path.exists(OutputFile)):
                    DownloadFile(EPICID,OutputFile,CampaignStr,StatusFile)
                else:
                    print ("The file already exists")


#close the files
StatusFile.close()
