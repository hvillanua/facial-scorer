# facial-scorer
Facial scorer applying eigenfaces and linear regression.

The images and the faces folders contain the compressed images, so you don't need to run the fetch_data_facial_scorer.py script.

The given image dataset is incomplete, as the scores only range from 7.8 to 8.3. This will yield unfair results
since the regression doesn't fit the whole range (0-10).

Used packages:
  - PIL
  - sklearn
  - cPickle
  - numpy
  - pylab
  - selenium (only if using the fetch_data_facial_scorer.py script)

Opencv is also required to work.

To simply test the model:
  - Unpack the images.tar.gz inside the images folder, and the faces.tar.gz inside the faces folder.
  - Call the predict function found in predict_score.py. I.e.: predict("local\\repository\\path\\faces\\91924.jpg").
Be careful not to delete the "data_dict.pkl" file inside the scores folder if you're just testing it for the first time.

To generate a new model based on your own image database:
  - Delete the content from all the folders.
  - Download your pictures and place them on the pictures folder.
  - Generate a python dictionary using the name of the pictures (without the file extension) as keys and their score as the value, i.e.:{91924 : 7.87288}.
      Name it "data_dict.pkl", pickle it and place it on the scores folder. You can use the fetch_data_facial_scorer.py file
      as a guide to make your own script.
  - Call the preprocess function found in preprocess_images.py
  - Perform a manual deletion of the incorrect classified faces (false positives).
    Make sure that there is only one face image for each original image. When there is more than one face from the original image
    you will see the name is "originalName_X" where X is the detected face.
  - Call the unifyFaceFilenames function found in preprocess_images.py.
  - Call the predict function found in predict_score.py.
