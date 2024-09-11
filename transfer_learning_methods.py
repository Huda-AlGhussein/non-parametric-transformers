import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold 
from sklearn.feature_selection import mutual_info_classif

def get_project_indices(df, project_column='Project'):
    # Group the dataframe by the project column and get the indices
    grouped = df.groupby(project_column)
    project_indices = {name: group.index for name, group in grouped}
    return project_indices

def split_dataframe_by_project_indices(df, project_indices):
    # Create a dictionary to store each project's dataframe
    project_dfs = [{'name':name,'df':df.loc[indices]} for name, indices in project_indices.items()]
    return project_dfs

def get_split_train_dataset (train_data):
    #X_train = pd.read_csv(train_path, index_col=0)
    #X_test = pd.read_csv(test_path, index_col=0)
    #X_test = pd.read_csv('Train-Test Data/ant/Test/ant_test_1.csv', index_col=0)
    Y_train = train_data["label"]
    train_data = train_data.drop(columns=["bugs","label"])
    #X_test.drop(columns=["bugs","label"],inplace=True)

    X_train_indices = get_project_indices(train_data)
    X_train = train_data.iloc[:,3:]
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X_train)
    #X_test = X_test.iloc[:,3:]
    X_train_ps = split_dataframe_by_project_indices(X_train,X_train_indices)
    Y_train_ps = split_dataframe_by_project_indices(Y_train,X_train_indices)

    
    return X_train_ps, Y_train_ps


def watanabe_transform(X_train, X_test,epsilon=1e-10):
    #print(X_train) 
    mean_train = X_train.mean()
    mean_test = X_test.mean()
    mean_test_safe = np.where(np.abs(mean_test) < epsilon, np.sign(mean_test) * epsilon, mean_test)

    X_test_transformed = X_test * (mean_train / mean_test_safe)
    # Replace any inf values in the result with 0
    X_test_transformed = np.nan_to_num(X_test_transformed, nan=0.0, posinf=0.0, neginf=0.0)
    return X_test_transformed

def cruz_transform(X_train, X_test):
    #print(X_train)
    median_train = np.median(X_train,axis=0) #X_train.median()
    median_test = np.median(X_test,axis=0)#X_test.median()
    X_test_transformed = X_test + (median_train - median_test)
    return X_test_transformed

def knn_filter(X_train, X_test, y_train, k=10):
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(X_train)
    distances, indices = knn.kneighbors(X_test)
    # To get the filtered training set without duplicates
    #X_train_filtered = X_train[np.unique(indices.flatten())]
    #X_train_filtered = np.unique(X_train[indices.flatten()], axis=0)
    indices = np.unique(indices.flatten())
    X_train_filtered = X_train[indices]
    Y_train_filtered = y_train.iloc[indices]
    return X_train_filtered,Y_train_filtered

def he_method(X_train_projects, X_test, y_train_projects,model, N=10, FSS=0.8):
    distances = []
    SAMS=[]
    # Step 1: Calculate distances between each training set and the test set
    for i in range(len(X_train_projects)):
        p_df = X_train_projects[i]['df']
        p_name = X_train_projects[i]['name']
        print(f'HE_Method Processing {p_name} Project')
        K = min(500,len(p_df), len(X_test))
        SAMtrain = p_df.sample(n=K, random_state=42)
        #SAMtest = X_test[np.random.randint(X_test.shape[0], size=K), :]
        SAMtest = X_test.sample(n=K, random_state=42)
        
        SAMtrain['label'] = 1
        SAMtest['label'] = -1
        
        SAM = pd.concat([SAMtrain, SAMtest])
        SAMS.append(SAM)
        y_SAM = SAM['label']
        X_SAM = SAM.drop('label', axis=1)
        kf = KFold(n_splits=5)
        classifier = LogisticRegression(random_state=42)
        classifier.fit(X_SAM, y_SAM)
        
        accuracy = accuracy_score(y_SAM, classifier.predict(X_SAM))
        distances.append(2 * (accuracy - 0.5))  # Use 1 - accuracy as distance measure
    # Step 2: Select top_N training sets with lowest distance
    top_N_indices = np.argsort(distances)[:N]
    selected_train_sets = [X_train_projects[i]['df'] for i in top_N_indices]
    selected_train_labels = [y_train_projects[i]['df'] for i in top_N_indices]

    # Step 3: Remove unstable features using Information Gain
    def remove_unstable_features(X, y, ratio):
        # Calculate information gain for each feature
        info_gains = mutual_info_classif(X, y)
        
        # Sort features by information gain (descending order)
        sorted_features = pd.Series(info_gains, index=X.columns).sort_values(ascending=False)
        
        # Select stable features (those with lower information gain)
        n_keep = int(len(sorted_features) * (1 - ratio))
        stable_features = sorted_features.index[-n_keep:]
        
        return X[stable_features]
    
    # Apply feature selection to each training set (maybe SAMS should be passed here instead, not sure)
    selected_train_sets = [remove_unstable_features(X, y, FSS) 
                           for X, y in zip(selected_train_sets, selected_train_labels)]
    
    # Apply feature selection to test set (using the first training set for consistency)
    #X_test_stable = remove_unstable_features(X_test, y_train_projects[top_N_indices[0]]['df'], FSS)
    # Step 4: Learn prediction models and ensemble
    #cls_predictions=[]
    
    print(f'Predicting with {type(model).__name__} classifier..')
    predictions , probabilities = [], []
    for X_train, y_train in zip(selected_train_sets, selected_train_labels):
        #model = model(random_state=42)
        model.fit(X_train, y_train)
        # Use same features as X_train
        X_test_stable = X_test[X_train.columns] 
        pred = model.predict(X_test_stable)
        proba = model.predict_proba(X_test_stable)[:, 1]
        predictions.append(pred)
        if np.isnan(np.min(proba)):
            #print(X_test_stable)
            #print(pred)
            #print(model.predict_proba(X_test_stable))
            #print(proba)
            proba = np.nan_to_num(proba,nan=0.0)
        probabilities.append(proba)
    
    # Calculate average prediction scores
    predictions_array = np.array(predictions)
    final_predictions = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)),
        axis=0,
        arr=predictions_array
    )
    final_probabilities = np.mean(probabilities, axis=0)
    #cls_predictions.append({'model':type(cls).__name__,'final_predictions':final_predictions})
        
    return final_predictions,final_probabilities

def he_method_multiple(X_train_projects, X_test, y_train_projects,classifiers=[RandomForestClassifier], N=10, FSS=0.8):
    distances = []
    SAMS=[]
    # Step 1: Calculate distances between each training set and the test set
    for i in range(len(X_train_projects)):
        p_df = X_train_projects[i]['df']
        p_name = X_train_projects[i]['name']
        print(f'Processing {p_name} Project')
        K = min(500,len(p_df), len(X_test))
        SAMtrain = p_df.sample(n=K, random_state=42)
        SAMtest = X_test.sample(n=K, random_state=42)
        
        SAMtrain['label'] = 1
        SAMtest['label'] = -1
        
        SAM = pd.concat([SAMtrain, SAMtest])
        SAMS.append(SAM)
        y_SAM = SAM['label']
        X_SAM = SAM.drop('label', axis=1)
        kf = KFold(n_splits=5)
        classifier = LogisticRegression(random_state=42)
        classifier.fit(X_SAM, y_SAM)
        
        accuracy = accuracy_score(y_SAM, classifier.predict(X_SAM))
        distances.append(2 * (accuracy - 0.5))  # Use 1 - accuracy as distance measure
    # Step 2: Select top_N training sets with lowest distance
    top_N_indices = np.argsort(distances)[:N]
    selected_train_sets = [X_train_projects[i]['df'] for i in top_N_indices]
    selected_train_labels = [y_train_projects[i]['df'] for i in top_N_indices]

    # Step 3: Remove unstable features using Information Gain
    def remove_unstable_features(X, y, ratio):
        # Calculate information gain for each feature
        info_gains = mutual_info_classif(X, y)
        
        # Sort features by information gain (descending order)
        sorted_features = pd.Series(info_gains, index=X.columns).sort_values(ascending=False)
        
        # Select stable features (those with lower information gain)
        n_keep = int(len(sorted_features) * (1 - ratio))
        stable_features = sorted_features.index[-n_keep:]
        
        return X[stable_features]
    
    # Apply feature selection to each training set (maybe SAMS should be passed here instead, not sure)
    selected_train_sets = [remove_unstable_features(X, y, FSS) 
                           for X, y in zip(selected_train_sets, selected_train_labels)]
    
    # Apply feature selection to test set (using the first training set for consistency)
    #X_test_stable = remove_unstable_features(X_test, y_train_projects[top_N_indices[0]]['df'], FSS)
    # Step 4: Learn prediction models and ensemble
    cls_predictions=[]
    for cls in classifiers:
        print(f'Predicting with {cls} classifier..')
        predictions = []
        for X_train, y_train in zip(selected_train_sets, selected_train_labels):
            model = cls(random_state=42)
            model.fit(X_train, y_train)
            # Use same features as X_train
            X_test_stable = X_test[X_train.columns] 
            pred = model.predict_proba(X_test_stable)[:, 1]
            predictions.append(pred)
    
        # Calculate average prediction scores
        final_predictions = np.mean(predictions, axis=0)
        cls_predictions.append({'model':type(cls).__name__,'final_predictions':final_predictions})
        
    return cls_predictions

def herbold_method(X_train_projects,y_train_projects, X_test, k=0.5):
    distances = []
    SAMS=[]
    X_test_mean = X_test.mean(axis=0)
    X_test_std = X_test.std(axis=0)
    distances = []
    
    for i in range(len(X_train_projects)):
        p_df = X_train_projects[i]['df']
        p_name = X_train_projects[i]['name']

        project_means = p_df.mean(axis=0)
        project_stds = p_df.std(axis=0)

        sim = np.sqrt(((project_means - X_test_mean) ** 2).mean() + ((project_stds - X_test_std) ** 2).mean())
        #print(sim)
        distances.append(sim)
    
    indices = np.argsort(distances)
    selected_projects = [X_train_projects[i] for i in indices]
    
    X_train_filtered = pd.concat([s['df'] for s in selected_projects])
    y_train_filtered = pd.concat([y_train_projects[i]['df'] for i in indices])
    
    return selected_projects, X_train_filtered, y_train_filtered

def ma_transfer_naive_bayes(X_train, X_test, y_train):
    def calculate_weights(X_train, X_test):
        weights = []
        #for i, row in X_train.iterrows():
        for row in X_train:
            h = ((X_test.min() <= row) & (row <= X_test.max())).astype(int)
            si = h.sum()
            weight = 1 / ((len(row) - si + 1) ** 2)
            weights.append(weight)
        return np.array(weights)

    weights = calculate_weights(X_train, X_test)
    clf = GaussianNB()
    clf.fit(X_train, y_train, sample_weight=weights)
    predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    return predictions,proba