# COVER

TITLE: HUMAN ACTIVITY RECOGNITION FROM SMARTPHONE MOTION SENSORS USING A CONV1D–BILSTM–ATTENTION MODEL
DEPARTMENT: Department of Computer Science and Engineering
STUDENT: Devansh Shankhadhar (24U020027)
STUDENT: Hemant Nagar (24U020015)
SUPERVISOR: Dr. Yatendra Sahu|Assistant Professor
SUPERVISOR: Dr. Ridhi Arora|Assistant Professor
SUPERVISOR_DEPARTMENT: Computer Science and Engineering
DATE: November 2026

## CERTIFICATE

This is to certify that the work embodied in this report entitled **“HUMAN ACTIVITY RECOGNITION FROM SMARTPHONE MOTION SENSORS USING A CONV1D–BILSTM–ATTENTION MODEL”** has been satisfactorily completed by **Devansh Shankhadhar (24U020027)** and **Hemant Nagar (24U020015)**. It is a bonafide piece of work, carried out under our guidance in the **Department of Computer Science and Engineering, Indian Institute of Information Technology, Bhopal** for the partial fulfillment of the Bachelor of Technology during the academic year 2026-2027.

Date: ____________

{gap}

| {plain} | | |
|:--|:--|:--|
| **Assistant Professor,** | **Assistant Professor,** | **Department Coordinator,** |
| Dr. Yatendra Sahu, | Dr. Ridhi Arora, | Dr. Rahul Chaurasia |
| Computer Science and Engineering, | Computer Science and Engineering, | Department of CSE, |
| IIIT Bhopal (M.P.) | IIIT Bhopal (M.P.) | IIIT Bhopal (M.P.) |

## DECLARATION

We hereby declare that the following project report entitled **“HUMAN ACTIVITY RECOGNITION FROM SMARTPHONE MOTION SENSORS USING A CONV1D–BILSTM–ATTENTION MODEL”** is presented in partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in Computer Science and Engineering. It is an authentic documentation of our work carried out under the able guidance of Dr. Yatendra Sahu and Dr. Ridhi Arora. The work has been carried out at the Indian Institute of Information Technology, Bhopal. The project work presented has not been submitted in part or whole for the award of any degree or professional diploma in any other institute or organization.

We hereby declare that the facts mentioned above are true to the best of our knowledge. In case of any discrepancy that may occur, we will be the ones to take responsibility.

| {plain} | |
|:--|--:|
| Devansh Shankhadhar (24U020027) | Signature: ____________ |
| Hemant Nagar (24U020015) | Signature: ____________ |

## ACKNOWLEDGEMENT

We, Devansh Shankhadhar and Hemant Nagar, would like to express our sincere gratitude to everyone who supported and guided us throughout the completion of our project titled **“Human Activity Recognition from Smartphone Motion Sensors Using a Conv1D–BiLSTM–Attention Model”**.

First and foremost, we would like to thank our Director, **Prof. Dr. Ashutosh Kumar Singh**, for providing the resources and facilities needed to carry out this project.

We are deeply grateful to our Project Guides, **Dr. Yatendra Sahu** and **Dr. Ridhi Arora**, Assistant Professors, Department of Computer Science and Engineering, for regular guidance and for the suggestions that shaped the evaluation of the model.

Our sincere thanks to the Department Coordinator, **Dr. Rahul Chaurasia**, for the support and encouragement provided during the project, and to the faculty of the Department of Computer Science and Engineering for their teaching and support.

We also thank our parents and families for their patience and moral support.

{sig}Devansh Shankhadhar
{sig}24U020027
{sig}Hemant Nagar
{sig}24U020015

## AREA OF WORK

Our project focuses on **Human Activity Recognition (HAR)** using **deep learning for time-series classification**. The aim is to decide automatically, from the accelerometer and gyroscope of an ordinary smartphone, which of six activities a person is performing.

The work covers signal processing of accelerometer and gyroscope data, the combination of several public datasets into one consistent training set, a hybrid convolutional and recurrent network with temporal attention, and its evaluation on people and recordings not used for training, with attention to data leakage, units and sampling rates. The trained model is served by a Flask REST API to a browser client that reads a phone's motion sensors, with predictions stored in SQLite.

## TABLE OF CONTENTS

## LIST OF FIGURES

## LIST OF TABLES

## ABSTRACT

Human Activity Recognition (HAR) is the task of identifying what a person is doing, such as walking or climbing stairs, from sensor data. This project builds a HAR system that uses the accelerometer and gyroscope of a smartphone. A web page on the phone samples the sensors at 20 Hz and sends a three-second window of six signals to a Flask server every 1.5 seconds. The server filters the window, removes gravity, adds two magnitude channels, and classifies the result with a neural network made of one-dimensional convolutional layers, two bidirectional Long Short-Term Memory (BiLSTM) layers and a temporal attention layer. The six output classes are Hand Activity, Jogging, Sports, Stairs, Still and Walking. Rule-based post-processing and per-session smoothing stabilise the label, and every prediction is stored in an SQLite database with CSV and PDF export.

The model is trained on three public datasets (WISDM, UCI HAR and Heterogeneity) and a small set of phone recordings. Every source is first converted to the units, sampling rate and window length of the live input, and the data are split by person and by recording before augmentation, so that no test subject or recording contributes to training. On 27,352 windows from public-dataset subjects excluded from training, the model reaches an accuracy of 77.9 percent, a macro precision of 0.807, a macro recall of 0.798 and a macro F1-score of 0.796; Stairs recall is 0.928, while Hand Activity, which is often confused with Still, is the weakest class (recall 0.473). On the only held-out phone stair-climbing session, 54 of 117 windows are classified as Stairs. This session is the only labelled phone test data for Stairs, so the result does not establish performance on phones in general, and the accuracy target of 80 percent was not reached.

**Keywords:** Human Activity Recognition, smartphone sensors, Conv1D, Bidirectional LSTM, temporal attention, data leakage, Flask, React

## CHAPTER 1: INTRODUCTION

### 1.1 Background

Almost every smartphone contains a three-axis accelerometer, which measures linear acceleration, and a three-axis gyroscope, which measures the rate of rotation. Walking, running or climbing stairs with a phone in a pocket or in the hand produces characteristic patterns in these signals, and Human Activity Recognition (HAR) uses such patterns to identify the activity automatically, with applications in fitness tracking, health monitoring and context-aware services [1], [2]. Early HAR systems computed hand-designed features from short windows and classified them with methods such as Support Vector Machines or decision trees [1], [2]. Deep neural networks instead learn features directly from the signal: convolutional layers detect short local patterns, recurrent layers such as the Long Short-Term Memory (LSTM) model their temporal order, and hybrid networks combine both [3], [4], [5], [6].

### 1.2 Motivation

Most published HAR results are obtained offline on a single benchmark dataset, with one device, one body position and a fixed protocol. A phone-browser application does not share these conditions. Its input comes from the DeviceMotion interface of a mobile browser, sampled at 20 Hz in m/s² with gravity included, from a phone held in the hand or carried in a pocket. No public dataset was recorded in exactly this format, so a model can only be trained by combining datasets whose units, sampling rates and labels differ. Every mismatch between training data and live input lowers recognition quality, and even differences between devices alone measurably reduce performance [7].

Two activities make this problem concrete. Walking and stair climbing produce similar periodic acceleration patterns and are easily confused [2], and hand-based activities produce little motion at a phone in the pocket. A useful system must therefore (i) convert all training sources to the live-input format, (ii) evaluate on people and recordings that were never used for training, because overlapping windows and augmented copies inflate accuracy when they cross the train–test boundary (Section 2.7), and (iii) turn per-window predictions into a stable, stored record of activity. This project addresses these three requirements with a Conv1D–BiLSTM–attention model served by a Flask API to a browser dashboard.

### 1.3 Scope of the Project

1. A Conv1D–BiLSTM–attention model with six output classes (Hand Activity, Jogging, Sports, Stairs, Still and Walking), trained on the WISDM [8], UCI HAR [1] and Heterogeneity [9] datasets and a few phone recordings.
2. A data-preparation pipeline that converts all sources to the live-input format and splits the data by person and by recording before augmentation to avoid the evaluation bias described in [10], and an evaluation of the model on held-out subjects and phone recordings, supported by reproducible evaluation scripts and 25 automated tests.
3. A Flask REST API that preprocesses each window with the same function used for training, runs the model, applies post-processing and per-session smoothing, similar in purpose to the filtering of consecutive outputs in real-time smartphone HAR [11], and stores the predictions in SQLite with CSV and PDF export.
4. A lightweight browser client (React and Vite) that streams 20 Hz sensor windows, the sampling rate also used by the WISDM dataset [8], to the API and displays the predicted activity. It serves as the test and demonstration front end of the model rather than as a research contribution. Google sign-in, MongoDB logging and Gemini-generated reports exist in the code as an optional module that is disabled without credentials; it was not tested and is outside the scope of this report.

### 1.4 Organisation of the Report

Chapter 2 reviews the relevant datasets [1], [8], [9] and deep-learning methods [4], [6], [12], and Chapter 3 defines the problem and objectives. Chapter 4 describes the data pipeline, model, training and real-time prediction; Chapter 5 gives the algorithms and Chapter 6 the main diagrams. Chapter 7 summarises the implementation, Chapter 8 presents the results and limitations, and Chapter 9 concludes the report.

## CHAPTER 2: BACKGROUND DETAILS AND LITERATURE REVIEW

### 2.1 Introduction

A typical HAR system divides the continuous sensor signal into short, fixed-length windows and assigns one activity label to each window. Early systems passed hand-designed window features to classical classifiers [1], [2], while later research showed that deep networks can learn features directly from raw or lightly processed signals [3], [4], [5]. Sections 2.2 to 2.4 review the three public datasets used in this project and how their labels relate to our six classes, Sections 2.5 and 2.6 review the relevant deep-learning methods, and Section 2.7 summarises the research gaps.

### 2.2 Research on the UCI HAR Dataset

Anguita et al. introduced the UCI Human Activity Recognition dataset in 2013 [1]. Thirty volunteers performed six activities, namely walking, walking upstairs, walking downstairs, sitting, standing and lying, while a smartphone was fixed to their waist. The accelerometer and gyroscope signals were divided into windows, and 561 time-domain and frequency-domain features were calculated from each window. A multiclass Support Vector Machine (SVM) trained on these features reached an overall accuracy of 96 percent on 2,947 test windows [1]. The dataset is distributed by the UCI Machine Learning Repository as "Human Activity Recognition Using Smartphones".

Official dataset page [1]: https://archive.ics.uci.edu/dataset/240/

Reyes-Ortiz et al. extended this work to transitions between activities, such as standing up or lying down, with a real-time system that combines consecutive SVM outputs with heuristic filtering [11]. The dataset became a common benchmark because of its fixed division into training and test subjects [13], [14], [15], although its strong results were obtained with hand-designed features, a single body position and a controlled recording procedure.

In our project, the raw inertial signals of this dataset [1] are part of the training data, mapped to Walking, Stairs and Still (Table 3). They differ from our phone input: they are sampled at 50 Hz in units of g from a phone worn on the waist, whereas our input is sampled at 20 Hz in m/s² with gravity included from a phone held in the hand or kept in a pocket. The pipeline therefore uses the total-acceleration signals, converted to m/s² and resampled to 20 Hz (Section 4.3).

### 2.3 Research on the WISDM Dataset

The Wireless Sensor Data Mining (WISDM) laboratory at Fordham University has produced two well-known activity datasets. In the earlier study, Kwapisz et al. collected accelerometer data from 29 users who carried a phone in their pocket while walking, jogging, climbing up and down stairs, sitting and standing [2]. They calculated 43 features from 10-second segments and compared decision trees, logistic regression and a multilayer perceptron. The multilayer perceptron achieved the highest overall accuracy of 91.7 percent, and climbing up and down stairs were the hardest activities to recognise [2]. This earlier dataset contains accelerometer data only and is not the WISDM dataset used in our project.

In 2019, Weiss et al. released the WISDM Smartphone and Smartwatch Activity and Biometrics Dataset [8], [16]. It contains accelerometer and gyroscope data from a smartphone and a smartwatch, recorded at 20 Hz from 51 subjects who each performed 18 activities for 3 minutes [8]. The activities include basic movements such as walking, jogging and climbing stairs, hand-based activities such as brushing teeth and folding clothes, and several eating activities. The phone was kept in the subject's pocket and the watch was worn on the dominant hand [8], [17]. The main aim of the original study was biometrics: Weiss et al. showed that a person can be identified from the way they perform everyday activities [16]. The dataset is available from the UCI Machine Learning Repository.

Official dataset page [8]: https://archive.ics.uci.edu/dataset/507/

Oluwalade et al. used the same dataset for activity recognition [17]. They found a statistically significant difference between the data recorded by the phone and by the watch and explained it by the different places where the two devices are worn. After combining the eating activities into one class, which left 15 activities, they trained LSTM, bidirectional LSTM, CNN and convolutional LSTM models on watch accelerometer data. Activities not performed with the hands, such as standing and stairs, were generally classified better than hand-oriented activities [17]. Alam et al., who used the dataset in a federated-learning benchmark, note that the phone and watch data were not recorded at the same time for most subjects [18].

Our project uses only the smartphone accelerometer and gyroscope files of this dataset [8]; the mapping of its activity codes to our classes is given in Table 3, and the eating and drinking activities are not used. WISDM is our only source of Hand Activity and Jogging data and, together with the "bike" activity of the Heterogeneity dataset, of Sports data. Hand Activity describes hand-based tasks recorded while the phone was in the pocket, so the phone sees only the small body movement that accompanies the hand movement; because hand-oriented activities were harder to classify even with a watch on the moving hand [17], this class was expected to be difficult (Chapter 8).

### 2.4 Research on the Heterogeneity Activity Recognition Dataset

Stisen et al. studied how differences between devices affect activity recognition [7]. They collected data with 36 smartphones and smartwatches of 13 different models from four manufacturers, used by nine people, and showed that differences in sensor hardware and in the way each device handles its sensors measurably reduce recognition performance [7]. The data were released as the Heterogeneity Activity Recognition dataset [9], with accelerometer and gyroscope readings for biking, sitting, standing, walking, stair up and stair down, plus a null class. Each device was sampled at the highest frequency it allowed, and the documentation does not state where the phones were carried [9].

Official dataset page [9]: https://archive.ics.uci.edu/dataset/344/

Following the label names of the dataset [9], its activities are mapped to Walking, Still, Stairs and Sports (Table 3), and each device's readings are aligned by timestamp and resampled to 20 Hz (Section 4.3). Our live input comes from yet another source, the motion sensors exposed by a mobile web browser; device differences are known to reduce recognition performance [7], but the effect of this particular difference has not been measured.

### 2.5 Convolutional, Recurrent and Attention Models

**Convolutional networks.** LeCun et al. showed that networks built from local receptive fields, shared weights and sub-sampling learn features directly from data instead of relying on hand-designed features [19]. Applied in one dimension along the time axis, a convolutional filter responds to short sensor patterns such as the peak of a footstep. Zeng et al. were among the first to apply CNNs to mobile-sensor activity recognition and outperformed the methods of the time on the Skoda, Opportunity and Actitracker datasets [4]; Yang et al. learned features from raw multichannel signals jointly with the classifier [3], and Ha and Choi combined several accelerometers and gyroscopes in one network [20]. Because a convolutional layer sees only a limited stretch of the signal, hybrid models add recurrent layers after it [6].

**Recurrent networks.** The LSTM of Hochreiter and Schmidhuber uses gated memory cells to keep information over long sequences [21], and the bidirectional network of Schuster and Paliwal reads a sequence forwards and backwards so that each output uses both earlier and later context [22]; in window-based HAR the whole window is available, so this needs no future data. Hammerla et al. compared fully connected, convolutional and recurrent models on three wearable datasets and gave guidelines for their use [5]. Guan and Plötz improved on single LSTMs with LSTM ensembles [23], Zhao et al. proposed a residual bidirectional LSTM with reported gains on Opportunity and UCI HAR [13], and Chen et al. reported 92.1 percent accuracy with an LSTM on WISDM accelerometer data [24].

**Temporal attention.** Bahdanau et al. replaced the fixed-length encoding of a sequence with learned soft weights over its parts [25], and Murahari and Plötz introduced attention layers into DeepConvLSTM for wearable HAR, reporting improved performance and interpretable temporal weights [12]. The attention layer of our model (Section 4.7) follows this general idea of learned soft weights over time steps, but it is not the same design as [25] or [12].

### 2.6 Hybrid Models and Data Augmentation

Ordóñez and Roggen proposed DeepConvLSTM, with four convolutional layers followed by two LSTM layers [6]. On the Opportunity dataset it outperformed deep non-recurrent networks by 4 percent on average and reached a weighted F1-score of 0.915 for gesture recognition including the Null class [6]. Mutegeki and Han reported 92 percent accuracy on UCI HAR with a CNN-LSTM model [14], and Xia et al. placed two LSTM layers before the convolutional layers and reported 95.78 percent on UCI HAR, 95.85 percent on WISDM and 92.63 percent on Opportunity [15]. Um et al. compared augmentation methods for wearable sensor data and found rotation to be the best single method; combining rotation, permutation and time warping raised accuracy from 77.54 to 86.88 percent [26]. Our model belongs to this hybrid family and our pipeline uses rotation-based augmentation (Section 4.5). The quoted figures come from different datasets and protocols and are not directly comparable.

### 2.7 Research Gaps and Relevance to Our Project

The way a model is evaluated strongly affects the reported accuracy. Hammerla and Plötz showed that random cross-validation on segmented time-series data is biased, because neighbouring windows are not statistically independent, and that performance drops when this bias is removed [10]. With 50 percent window overlap and augmented copies, a random window-level split would place near-duplicates of the same recording in both the training and the test data. The pipeline of this project therefore splits the data by person and by recording before augmentation (Section 4.5).

A second gap is generalisation across devices and body positions. Stisen et al. showed that differences between devices reduce recognition performance [7], and Oluwalade et al. found that a phone and a watch record the same activities differently because of where they are worn [17]. The datasets used in our project were recorded with a phone on the waist [1], in the pocket [8], or with many different devices [7], while our users may hold the phone in the hand or keep it in a pocket.

Most of the reviewed deep-learning studies report results on recorded benchmark datasets [3], [5], [13], [6], [15]; real-time phone-based recognition has been addressed [11], but less attention is given to the full path from live browser sensor data to a stored record of activity. This project therefore contributes a consistent data pipeline, a working system and a leakage-free evaluation rather than a more accurate model, and its measured results do not exceed the published methods. Table 1 summarises the reviewed studies that are most relevant to the methodology of this project.

**Table 1: Comparison of HAR Approaches in Literature**

| Approach | Representative Work | Main Contribution | Relevance to This Project |
|----------|-------------------|-------------------|---------------------------|
| Features with classical classifiers | Kwapisz et al. [2] | Pocket phone accelerometer; 91.7% with a multilayer perceptron | Earlier WISDM study; stairs were hardest to recognise |
| Device heterogeneity | Stisen et al. [7] | Device differences reduce recognition performance | Heterogeneity dataset is part of our training data |
| 1-D CNN | Zeng et al. [4], Yang et al. [3] | Features learned from raw sensor signals | Basis of our convolutional blocks |
| LSTM and BiLSTM | Hammerla et al. [5], Zhao et al. [13] | Modelling of temporal order in sensor windows | Basis of our BiLSTM layers |
| CNN-LSTM hybrid | Ordóñez and Roggen [6], Xia et al. [15] | Local features combined with temporal modelling | Same family as our model |
| Temporal attention | Murahari and Plötz [12] | Learned weights over temporal context | Basis of our attention layer |
| Data augmentation | Um et al. [26] | Rotation was the most useful single augmentation | Rotation is used in our training pipeline |
| Evaluation bias | Hammerla and Plötz [10] | Random splits of overlapping windows inflate results | Reason for our person-level split before augmentation |

## CHAPTER 3: PROBLEM DEFINITION AND OBJECTIVES

### 3.1 Problem Statement

The problem addressed in this project is to recognise six physical activities (Hand Activity, Jogging, Sports, Stairs, Still and Walking) in real time from the accelerometer and gyroscope of an ordinary smartphone, using only the phone's web browser as the sensing client, and to measure the recognition performance on people and recordings that were not used for training.

The problem has three technical parts. First, the training data come from three public datasets and a few phone recordings that differ in units, sampling rate, body position and activity labels, whereas the live input has a fixed format (m/s² including gravity, rad/s, 20 Hz, 3-second windows); all sources must be brought into this format. Second, the evaluation must be free of leakage between overlapping windows and augmented copies, which requires a split by person and by recording made before augmentation. Third, per-window predictions must be stabilised and stored so that a user can read and review the recognised activity.

### 3.2 Objectives

1. To design a six-class Conv1D–BiLSTM–temporal-attention classifier for smartphone accelerometer and gyroscope windows, with a target accuracy of at least 80 percent on subjects excluded from training.
2. To build a data-preparation pipeline that converts the WISDM, UCI HAR and Heterogeneity datasets and the project's phone recordings to the units, sampling rate and window format of the live input.
3. To evaluate the model without data leakage, using person- and recording-level splits made before augmentation, and to report per-class and overall metrics.
4. To implement a real-time inference service that shares its preprocessing with training and stabilises predictions with rule-based post-processing and per-session smoothing.
5. To develop a browser dashboard for live sensing, CSV analysis and replay, with SQLite session storage and CSV and PDF export.

Section 9.2 reviews which of these objectives were met.
