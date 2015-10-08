import sys, os, glob, PIL
import cPickle as pickle
import numpy as np
import pylab as pl

from time import time
from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA
from sklearn import linear_model

def predict(pred_img_path, scores_folder=u'scores', faces_folder=u'faces', model_path=u'model', color=False, base_width=62, n_components=20):
    """ function to predict the score of a face
        if the prediction model model doesn't exist in the
        model_path, it will create a new one and save it for future use

    Parameters
    ----------
    pred_img_path : string, mandatory
        path to the image used to predict
    scores_folder : string, optional, default u'scores'
        path to the folder containing the score file
    faces_folder : string, optional, default u'faces'
        path to the folder containing the faces cropped from the original images
    model_path : string, optional, default u'faces'
        path to the folder containing both the pca and the regression model
    color : boolean, optional, default False
        whether to use images with color or use a gray level representation
    base_width : int, optional, default 62
        target width for the resized square shaped of the input image
    n_components: int, optional, default 20
        maximum number of components to keep. When not given or None, this is set to n_features (the second dimension of the training data).
    """

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # if the pca or the model file don't exist, train a new pca and model and save it
    regression_path = os.path.join(model_path, 'regression.pkl')
    pca_path = os.path.join(model_path, 'pca.pkl')
    """
        Set n_components to a high value (i.e.: 200) and use
        plot_pca_components(pca.explained_variance_ratio_)
        after training the pca to visually appreaciate the values.
        Can apply Kaiser or Scree plot criterions to approximately
        determine a good value.
    """
    if (not os.path.exists(pca_path)) or (not os.path.exists(regression_path)):
        faces_filename = os.path.join(faces_folder, '*.jpg')
        images = glob.glob(faces_filename)
        images = map(lambda x: x[x.rfind("\\")+1:], images)
        scores_filename = os.path.join(scores_folder, 'data_dict.pkl')
        if not os.path.exists(scores_filename):
            print 'The file "data_dict.pkl" was not found in the scores folder.'
            return
        scores_dict = pickle.load( open(scores_filename, 'rb') )
        n_faces = len(images)
        # allocate some contiguous memory to host the decoded image slices and the scores
        # heigth and width are the same since we're using square shaped face images
        if not color:
            X = np.zeros((n_faces, base_width, base_width), dtype=np.float32)
        else:
            X = np.zeros((n_faces, base_width, base_width, 3), dtype=np.float32)
        Y = np.zeros(n_faces, dtype=np.float32)
        for i, img_name in enumerate(images):
            path = os.path.join(faces_folder, img_name)
            img = np.asarray(Image.open(path), dtype=np.float32)
            img /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
            if not color:
                # average the color channels to compute a gray levels representation
                img = img.mean(axis=2)
                
            X[i,...] = img
            Y[i] = scores_dict[img_name[:-4]]

        # reshape the two-dimensional pictures to one dimension
        X = X.reshape(len(X), -1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

        # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
        # dataset): unsupervised feature extraction / dimensionality reduction
        print "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
        t0 = time()
        pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
        plot_pca_components(pca.explained_variance_ratio_)
        joblib.dump(pca, pca_path)
        print "done in %0.3fs" % (time() - t0)
        
        print "Projecting the input data on the eigenfaces orthonormal basis"
        t0 = time()
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print "done in %0.3fs" % (time() - t0)

        # fit the regression and save it
        reg = linear_model.LinearRegression()
        reg.fit(X_train_pca,Y_train)
        joblib.dump(reg, regression_path)
        print "Printing model precision:"
        #print "Slope:", reg.coef_
        print "Intercept:", reg.intercept_
        print "Score of the regression:", reg.score(X_test_pca,Y_test)
        
    else:
        #load the pca and the model
        pca = joblib.load(pca_path)
        reg = joblib.load(regression_path)

    if not os.path.exists(pred_img_path):
        print "The given path does not exist."
        return

    img_ori = Image.open(pred_img_path)
    img_cut = img_ori.resize((base_width, base_width), Image.ANTIALIAS)
    img = np.asarray(img_cut, dtype=np.float32)
    img /= 255.0  # scale uint8 coded colors to the [0.0, 1.0] floats
    if not color:
        # average the color channels to compute a gray levels representation
        img = img.mean(axis=2)
    img = img.reshape(-1)
    X_pred = pca.transform(img)
    Y_pred = reg.predict(X_pred)
    print "Score for {}:".format(pred_img_path[pred_img_path.rfind("\\")+1:]), Y_pred


    # plot the gallery of the input picture and the most significative eigenfaces
    if not color:
        eigenfaces = pca.components_.reshape((n_components, base_width, base_width))
    else:
        eigenfaces = pca.components_.reshape((n_components, base_width, base_width, 3))

    result_images = np.zeros((n_components, base_width, base_width), dtype=np.float32)
    result_images[0] = img.reshape((1, base_width, base_width))
    result_images[range(1,n_components)] = np.delete(eigenfaces,eigenfaces.shape[0]-1, 0)
    eigenface_titles = [pred_img_path[pred_img_path.rfind("\\")+1:]] + ["eigenface %d" % i for i in range(eigenfaces.shape[0]-1)]
    plot_gallery(result_images, eigenface_titles, color, base_width, base_width)

    pl.show()

###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, color, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(min((n_row * n_col), images.shape[0])):
        pl.subplot(n_row, n_col, i + 1)
        if not color:
            pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        else:
            pl.imshow(images[i].reshape((h, w, 3)))
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

def plot_pca_components(components):
    """Helper function to plot the value of the pca components to visually adjust n_components"""
    pl.plot(range(len(components)),components, marker='o')
    pl.show()
