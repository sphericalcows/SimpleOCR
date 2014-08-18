SimpleOCR
=========

Project to allow simple OCR for the marking of multiple choice and numeric tests. It is assumed that writing is constrained
to allow for sufficiently high accuracy for the purpose.

These programs are a proof of concept. 
They are rather crude programs, with may parameters currently hard-coded, though I hope to make a lot of this user-input
driven once I figure out how to properly use GUI etc.

There are two major programs:
ExtractNumericData.py       Reads a template document and identifies rectangles to identify where digits are stored
      The data documents are then read and converted into a simplified matrix for data recognition. This is written to a
      file in order to be used for data recognition.
      This program has been used to produce the data in the .txt data files (described after)
  Files: 
    Input:
      NumericTrainingV3_Template.png  The template file for the numeric training OCR
      NumericTrainingV3-*.png These are png files containing the data used for training.
      FutherScans-*.png Additional png files containing independent data for testing
      The input png file to use is hard-coded at present
    Output:
      NumericData.txt A pickle file containing the numeric data in the form of a digit value and matrix data, 
        to be read by other programs
        
TestDigitRecognition.py     Reads files produced by the ExtractNumericData program and compares the accuracy of various 
      recognition models. 
  Files:
    Input:
      TrainingData.txt  Renamed NumericData.txt from the ExtractNumericData program, used for training the OCR
      TestData.txt  Also renamed output from the ExtractNumericData program, this used for testing
    Output:
      no output files

