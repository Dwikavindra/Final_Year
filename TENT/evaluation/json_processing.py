#Process json 
import os 
import json
#Process json
"""1. Open the json file 
    2. Process  the slicedImage for each candidate 
    3. check each digit in the digits folder 
    4. Check each digit folder after openinng the folder if digit1 folder is empty then return undefined else return an image path """

def getImageFilePath(filePath):
   fileNames=os.listdir(filePath)
   if(fileNames):
    # print(f"This is filePath {filePath}")
    # print(f"This is fileNames[0] {fileNames[0]}")
    fileNamesPNGFiltered=[fileName for fileName in fileNames if fileName.endswith(".png")]
    if(len(fileNamesPNGFiltered)>0):
        return filePath+f"/{fileNamesPNGFiltered[0]}"
   return None

def fill_image_os_path(jsonFolderPath,digitsFolderPath):
    jsonFolderFiles=os.listdir(jsonFolderPath)
    for fileName in jsonFolderFiles:
        if(fileName.endswith(".json")):
            jsonFilePath=f"{jsonFolderPath}/{fileName}"
            jsonData= json.load(open(jsonFilePath))
            slicedImage=jsonData["slicedImage"]
            balotId=jsonData["id"]
            for candidate, digits in slicedImage.items():
            #    print(candidate)
            #    print(digitsFolderPath)
                digitsFolderPathNew=f"{digitsFolderPath}/{balotId}/{candidate}"
                digits["digit1"]=getImageFilePath(digitsFolderPathNew+"/digit1")
                digits["digit2"]=getImageFilePath(digitsFolderPathNew+"/digit2")
                digits["digit3"]=getImageFilePath(digitsFolderPathNew+"/digit3")
            
                with open(jsonFilePath, "w") as file:
                    #writes changes to json file
                    json.dump(jsonData, file, indent=4)



def predict_images(jsonFolderPath,model):
    jsonFolderFiles=os.listdir(jsonFolderPath)
    for fileName in jsonFolderFiles:
        if(fileName.endswith(".json")):
            jsonFilePath=f"{jsonFolderPath}/{fileName}"
            jsonData= json.load(open(jsonFilePath))
            slicedImage=jsonData["slicedImage"]
            predictedResult=jsonData["predictedResult"]
            
            balotId=jsonData["id"]
            #get the image File Path and let it predict 

            for candidate, digits in slicedImage.items():
                digit1FilePath=digits["digit1"]
                digit2FilePath=digits["digit2"]
                digit3FilePath=digits["digit3"]
                predictedDigit1=predict_image(new_image_preprocessing(digit1FilePath),model)
                predictedDigit2=predict_image(new_image_preprocessing(digit2FilePath),model)
                predictedDigit3=predict_image(new_image_preprocessing(digit3FilePath),model)
                predictedResult[candidate]["digit1"]= None if predictedDigit1 == None else predictedDigit1["predicted"]
                predictedResult[candidate]["digit2"]= None if predictedDigit2 == None else predictedDigit2["predicted"]
                predictedResult[candidate]["digit3"]= None if predictedDigit3 == None else predictedDigit3["predicted"]
                with open(jsonFilePath, "w") as file:
                    #writes changes to json file
                    json.dump(jsonData, file, indent=4)

def calculate_amount_of_digit(jsonFolderPath,digitName):
    digit=0
    jsonFolderFiles=os.listdir(jsonFolderPath)
    for fileName in jsonFolderFiles:
        if(fileName.endswith(".json")):
            jsonFilePath=f"{jsonFolderPath}/{fileName}"
            jsonData= json.load(open(jsonFilePath))
            realResult=jsonData["realResult"]
            for _, digits in realResult.items():
                if digits[digitName]!= None :
                    digit+=1
    return digit

def calculate_correct_digit(jsonFolderPath,digitName):
    #digitName for id 
    my_dict = {}
    correct_digit=0
    jsonFolderFiles=os.listdir(jsonFolderPath)
    for fileName in jsonFolderFiles:
        if(fileName.endswith(".json")):
            jsonFilePath=f"{jsonFolderPath}/{fileName}"
            jsonData= json.load(open(jsonFilePath))
            realResult=jsonData["realResult"]
            id=jsonData['id']
            predictedResult=jsonData["predictedResult"]
            for _, digits in realResult.items():
                 if digits[digitName]!= None :
                    my_dict[id]=digits[digitName]
            for _, digits in predictedResult.items():
                 if digits[digitName]!= None :
                    if digits[digitName]== my_dict[id]:
                        correct_digit+=1
            
    return correct_digit


fill_image_os_path(f"/workspace/Dwika/fyp/data_election/json",f"/workspace/Dwika/fyp/data_election/digits")