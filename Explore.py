import glob
from os import system
import re

Folder = "PhaseCurves_MyMethod"
GirishFolder = "PhaseCurves_SFF_300"
Spitzer = "PhaseCurves_Spitzer_300"
Folders = glob.glob(Folder)

for folder in Folders:
    path_L = folder+"/figs/low_sn/*.png"
    path_H = folder+"/figs/high_sn/*.png"
    file_L = glob.glob(path_L)
    file_H = glob.glob(path_H)
    filenames = file_L+file_H
    for filename in filenames:
        starname= re.search('[0-9]{9}',filename).group(0)
        print starname

        if "spd" in filename:
            starname=starname+"_spd"
        FilePath = folder+"/"+starname
        system("eog %s" %(FilePath+"/*BLS*.png"))


        #system("eog %s/figs/high_sn/*%s*.png" %(GirishFolder,starname))
        #system("eog %s/figs/low_sn/*%s*.png" %(GirishFolder,starname))

        #system("eog %s/figs/high_sn/*%s*.png" %(Spitzer,starname))
        #system("eog %s/figs/low_sn/*%s*.png" %(Spitzer,starname))
        input("Continue to the next target?")
