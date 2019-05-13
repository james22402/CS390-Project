# Project Check Marks

Project Check Marks is a project which applies OpenCV and tesseract techniques in order to read different parts of a check. Note that we cannot show pictures as they would compromise security.

## Purpose

We wanted to read a check such that we could grab information like:
- Account Number
- Routing Number
- Check Number
- Amount
- Name that check is made out to

## Results

All in all the program was able to find the account number, routing number, and check number reliably. However, since we used tesseract - primarily an OCR library it was not able to effectively read handwriting. This made it so that trying to read both the name and the amount was difficult since they were not typed in, but instead written in.

## License
[MIT](https://choosealicense.com/licenses/mit/)
