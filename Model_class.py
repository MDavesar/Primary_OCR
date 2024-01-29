import pytesseract
import cv2
import pandas as pd
import os

class OcrModel:
    def __init__(self):
        self.custom_config = r'--oem 3 --psm 4'
    
    def ocr_predictions(self, filename):
        img = cv2.imread(filename)
        text = pytesseract.image_to_string(img, config=self.custom_config)
        return text

    def words(self, text):
        my_string = ''.join(text)
        words = my_string.split()
        return words
    
    def to_lowercase(self, words):
        lowercase_list = [string.lower() for string in words]
        return lowercase_list
    
    def accuracy_check(self, keywords, text, filename):
        keywords_match_cnt = 0
        wrds = self.words(text)
        aa = self.to_lowercase(wrds)
        for i in keywords:
            if i.lower() in aa:
                keywords_match_cnt += 1
        accuracy = (keywords_match_cnt / len(keywords)) * 100
        data = {'filename': filename[:-1], 'keywords': len(keywords), 'Matched': keywords_match_cnt, 'Accuracy': accuracy}
        return data

    def full(self, directory, keywords_list):
        database = pd.DataFrame()
        filenames = os.listdir(directory)
        for fileno in range(1, len(filenames) + 1):
            filename = f'{directory}/ocr{fileno}.jpg'
            text = self.ocr_predictions(filename)
            temp_data = self.accuracy_check(keywords_list[fileno - 1], text, filename)
            database = pd.concat([database, pd.DataFrame([temp_data])])
        return database
