import pytesseract
import cv2
import pandas as pd
import os

# Set the path to the Tesseract executable
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

def analyze_image(filtered_image, keywords):
    custom_config = r'--oem 3 --psm 4'

    def words(text):
        my_string = ''.join(text)
        return my_string.split()

    def to_lowercase(words):
        return [string.lower() for string in words]

    def ocr_predictions(image):
        text = pytesseract.image_to_string(image, config=custom_config)
        return text

    def match_multi_word_keyword(keyword, wrds):
        keyword_tokens = keyword.lower().split()
        for i in range(len(wrds) - len(keyword_tokens) + 1):
            if wrds[i:i+len(keyword_tokens)] == keyword_tokens:
                return True
        return False

    def accuracy_check(keywords, text, filename):
        keywords_match_cnt = 0
        wrds = to_lowercase(words(text))
        for keyword in keywords:
            if match_multi_word_keyword(keyword, wrds):
                keywords_match_cnt += 1
        accuracy = (keywords_match_cnt / len(keywords)) * 100
        data = {'filename': filename[:-1], 'keywords': len(keywords), 'Matched': keywords_match_cnt, 'Accuracy': accuracy}
        return pd.DataFrame([data])

    # Since there's no filename associated with the filtered image, you can pass an empty string
    filename = ''
    text = ocr_predictions(filtered_image)
    result = accuracy_check(keywords, text, filename)
    return result

keywords = ['haematology','blood','differential','platelet','patient','r.b.c','corpuscular','count','biological','authorize']

# Read the image
image_path = 'images\ocr8.jpg'
image = cv2.imread(image_path)

# Apply median filtering
median_filtered = cv2.medianBlur(image, 1)  # 5 is the kernel size

# Display the result
cv2.imshow('Median Filtered Image', median_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Analyze the filtered image
result = analyze_image(median_filtered, keywords)
print(result)
