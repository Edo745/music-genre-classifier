# Music Genre Classification

## Introduction and Objective
- Music classification is the process of categorizing songs into specific genres based on their musical attributes.
- Many popular platforms like Spotify and YouTube employ classification algorithms to enhance user experiences.
- The project's objective is to build a robust classification model capable of distinguishing among 8 distinct musical genres: Rock, Blues, Jazz, Country, EDM, DnB, Drill, Trap.

## The idea
The idea is to divide the songe into 30-second chunks and extract features from each chunk before passing them to the classifier. If we have $n$ chunks, we will obtain $n$ predictions. The final prediction will be the genre that receives the highest number of votes from the classifier.

_Note_: the intro and outro of each musical piece have been removed, assuming they are less informative.

<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/fa4aa8ce-3fd9-456a-a1f0-a0ca520e79ba">
</p>

## Data collection
- To assemble the dataset, I download from Youtube the playlists related to the 8 musical genres of interest. In particular, 100 songs per genre.
- I develop a custom code to automate the data collection process. This code allowed the user to simply retrieve the desired playlists and their respective songs.
- There is no limit, the user can download any playlist of any existing genre.


## Data preprocessing
- Splitting songs into 30-second segments
- Removing the first and last segments: this focused our model's attention on the central portions of the songs, where the genre characteristics are typically more pronounced.

## Feature Extraction
I utilized the powerful [Librosa](https://librosa.org) library, specifically designed for music and audio analysis.
I extracted a comprehensive set of features to represent each 30-second musical segment:
- Spectral features: chroma_stft, chroma_cqt, chroma_cens, chroma_vqt, melspectrogram, mfcc, rms,
spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_flatness, spectral_rolloff,
poly_features, tonnetz, zero_crossing_rate
- Rhythm features: tempo, tempogram, fourier_tempogram, tempogram_ratio
- Beat and tempo features: beat_track, plp

For the array-type features I calculated both the mean and variance.

## Dataset Creation
The dataset creation process began by gathering playlists from YouTube, covering a diverse array of 9 musical genres, ranging from established to emerging ones: blues, country, dnb, drill, edm, jazz, rock, trap and hiphop. Each of these playlists was curated to include 100 songs, resulting in a rich collection of musical compositions.

The next step involved dividing each song into segments, each lasting 30 seconds.

Librosa library is specialized in audio analysis and allowed for the extraction of pertinent features from each individual track.

The features extracted from these 30-second segments were then organized into a Pandas dataframe and used for the models' training.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7aff53d9-d761-4ded-85c9-fffd71baaba1">
</p> 

1. Spectral features:
  
 * **chroma_stft(*[, y, sr, S, norm, n_fft, ...])**: Compute a chromagram from a waveform or power spectrogram.

 * **chroma_cqt(*[, y, sr, C, hop_length, fmin, ...]**: Constant-Q chromagram

 * **chroma_cens(*[, y, sr, C, hop_length, fmin, ...]**): Compute the chroma variant "Chroma Energy Normalized" (CENS)

 * **chroma_vqt(*[, y, sr, V, hop_length, fmin, ...])**: Variable-Q chromagram

 * **melspectrogram(*[, y, sr, S, n_fft, ...])**: Compute a mel-scaled spectrogram.

 * **mfcc(*[, y, sr, S, n_mfcc, dct_type, norm, ...])**: Mel-frequency cepstral coefficients (MFCCs)

 * **rms(*[, y, S, frame_length, hop_length, ...])**: Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.

 * **spectral_centroid(*[, y, sr, S, n_fft, ...])**: Compute the spectral centroid.

 * **spectral_bandwidth(*[, y, sr, S, n_fft, ...])**: Compute p'th-order spectral bandwidth.

 * **spectral_contrast(*[, y, sr, S, n_fft, ...])**: Compute spectral contrast

 * **spectral_flatness(*[, y, S, n_fft, ...])**: Compute spectral flatness

 * **spectral_rolloff(*[, y, sr, S, n_fft, ...])**: Compute roll-off frequency.

 * **poly_features(*[, y, sr, S, n_fft, ...])**: Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.

 * **tonnetz(*[, y, sr, chroma])**: Compute the tonal centroid features (tonnetz)

 * **zero_crossing_rate(y, *[, frame_length, ...])**: Compute the zero-crossing rate of an audio time series.



2. Rhythm features

 * **tempo**(*[, y, sr, onset_envelope, tg, ...]): Estimate the tempo (beats per minute)

 * **tempogram**(*[, y, sr, onset_envelope, ...]): Compute the tempogram: local autocorrelation of the onset strength envelope.

 * **fourier_tempogram**(*[, y, sr, ...]): Compute the Fourier tempogram: the short-time Fourier transform of the onset strength envelope.

 * **tempogram_ratio**(*[, y, sr, onset_envelope, ...]): Tempogram ratio features, also known as spectral rhythm patterns.



3. Beat and tempo

 * **beat_track(*[, y, sr, onset_envelope, ...])**: Dynamic programming beat tracker.

 * **plp(*[, y, sr, onset_envelope, hop_length, ...])**: Predominant local pulse (PLP) estimation.




Here the [official documentation](https://librosa.org/doc/latest/feature.html) for more informations about the features.


## Dataset Balancing - undersampling
<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/0847fe02-4fb8-465b-ade5-f4d37d2b06cc">
</p> 

## Metrics used for the model evaluation
<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/fc626cab-85ac-4db1-92d5-c3e3cb26a17f">
</p> 

## Training set, Test set & Feature scaling
```python
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
joblib.dump(scaler, 'scaler.pkl')
```

## Models
<p align="center">
  <img width="800" src="https://github.com/user-attachments/assets/132aacc2-ea32-4758-ad16-9e615ab3e6b5">
</p> 

## One vs Rest Classifier
One vs Rest Classifier is a strategy used in multiclass problems and it consists to construct one classifier per class.
In this case eight distinct classifiers are trained, each aiming to distinguish a specific genre from the other seven.
I use the function OneVsRestClassifier() provided by the scikit-learn library

## Fit and predict
```python
clf_list = [
  RidgeClassifier(),
  KNeighborsClassifier(),
  tree.DecisionTreeClassifier(),
  RandomForestClassifier(),
  AdaBoostClassifier(
    estimator=DecisionTreeClassifier(),
     algorithm="SAMME",
     n_estimators=100
  )
]

for clf in clf_list:
classifier =
OneVsRestClassifier(eval(clf))
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)
```
## Confusion Matrixes
<p align="center">
  <img src="https://github.com/user-attachments/assets/a6d423ca-c119-4bef-9ac6-821514c00e75" width="400" alt="Ridge" title="Ridge"/> 
  <img src="https://github.com/user-attachments/assets/81c3ab16-b00c-4226-89c7-b4585cf2b91d" width="400" alt="KNN" title="KNN"/>
  <img src="https://github.com/user-attachments/assets/d0dac5d6-ef3a-46c6-ba08-8f87325ce78e" width="400" alt="Decision Tree" title="Decision Tree"/>
  <img src="https://github.com/user-attachments/assets/5ce83d5c-22e1-48be-bc4d-14a1cfe2248d" width="400" alt="Random Forest" title="Random Forest"/> 
  <img src="https://github.com/user-attachments/assets/203e9393-4165-4610-9268-e59a76c91453" width="400" alt="AdaBoost " title="AdaBoost "/>
</p> 

## Accuracies and Execution times
| Model               | Accuracy | Time  |
|---------------------|----------|-------|
| **rf_classifier**        | **0.75**     | 13.31 |
| **knn_classifier**       | 0.73     | 0.92  |
| **ridge_classifier**     | 0.64     | 0.23  |
| **adaBoost_classifier**  | 0.51     | 46.65 |
| **dt_classifier**        | 0.49     | 1.11  |

## ROC Curves
<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/71f7fc69-a776-43b1-9b84-2f79339ae0ad">
</p> 

## K-fold Cross Validation
I implemented the k-fold cross-validation in order to prevent overfitting and to have a more comprehensive understanding of the model's performance.
<p align="center">
  <img width="700" src="https://github.com/user-attachments/assets/07c106a4-faba-4b5e-a626-129dac597514">
</p> 

## Feature Importance
Down below the more important features for each musical genre.
<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9d7609-2981-4949-907a-5e2bba649501" width="400"/>
  <img src="https://github.com/user-attachments/assets/f3f0be34-330a-4627-95c2-f613bfe69e5a" width="400"/>
  <img src="https://github.com/user-attachments/assets/ac128c26-5989-4924-8d98-d4c0daa851d7" width="400"/>
  <img src="https://github.com/user-attachments/assets/531929f3-8e76-46b0-bc60-b674fc4c7d30" width="400"/>
  <img src="https://github.com/user-attachments/assets/84600022-4e1b-45ef-ad24-0700f7dbf632" width="400"/>
  <img src="https://github.com/user-attachments/assets/664976af-6df8-40c7-8b50-da0e3f111d32" width="400"/>
  <img src="https://github.com/user-attachments/assets/a13a4ad2-4815-4fe3-a04d-024fbfa343a5" width="400"/>
  <img src="https://github.com/user-attachments/assets/85dcd923-f3a8-41b0-b391-d8643db93f11" width="400"/>
  <img src="https://github.com/user-attachments/assets/93b9100c-4c7c-4f87-ab76-6ac2e508fbc5" width="400"/>
</p>

## Fine-Tuning
In this section, we perform hyperparameter tuning to optimize the performance of our Random Forest classifier. We utilize GridSearchCV from the scikit-learn library to systematically explore a range of hyperparameter values, including the number of estimators, maximum depth of the trees, and the minimum number of samples required to split an internal node. The tuning process employs cross-validation to ensure that we obtain robust estimates of model performance. Below is the implementation:

```python
param_grid = {
   'estimator__n_estimators': [100, 500],
   'estimator__max_depth': [5, 10, 20, 30],
   'estimator__min_samples_split': [5, 10, 20, 30],
}
scores = [ 'f1_macro' ]

for score in scores:
  grid_search = GridSearchCV(
   ovr_rf_classifier,
   param_grid=param_grid,
   cv=5,
   scoring=score,
   refit=True
  )
  
  grid_search.fit(X_train_scaled, y_train)

grid_search_models[score]=grid_search
```
## Let's try the model
<p align="center">
  <img width="500" src="https://github.com/user-attachments/assets/ef9baeb7-7625-4e2e-8185-fbc911bd6539">
</p> 









