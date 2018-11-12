from sklearn import feature_selection as fs
from sklearn import preprocessing as prep

import pandas as pd


def main():
    choice = '6'
    print("+==============================================================+")
    print("| Welcome to Bonnie's Master Project:                          |")
    print("| Python Tool for Log Analysis with integrated Machine Learning|")
    print("+==============================================================+")
    print("Please select an option:")
    print("1. Load Data")
    print("2. Select Features")
    print("3. Neural Network")
    print("4. Support VectPrevious Resultsor Machine")
    print("5. Results")
    print("6. Quit")
    print("=================================================================")
    choice = input("Please make a selection: ")
    print("Loading Data...")
    pd.set_option('display.max_columns', 10)
    f = open("NSL_KDD-master\KDDTrain+.csv")
    cnfile = open("NSL_KDD-master\Field Names.csv")
    column_names = pd.read_csv(cnfile,  header=None)
    column_names_list = column_names[0].tolist()
    col_list = list(range(0, 42))
    column_names_list.append("lables")
    data = pd.read_csv(f, header=None, names=column_names_list, usecols=col_list)
    le = prep.LabelEncoder()
    encoded_services = le.fit_transform(data.service)

    # Categorical boolean mask
    categorical_feature_mask = data.dtypes == object
    # filter categorical columns using mask and turn it into a list
    categorical_cols = data.columns[categorical_feature_mask].tolist()

    print(data[categorical_cols].head(20))
    data[categorical_cols[0]] = le.fit_transform(data[categorical_cols[0]])
    protocols_map = dict(zip(le.classes_, le.transform(le.classes_)))
    data[categorical_cols[1]] = le.fit_transform(data[categorical_cols[1]])
    services_map = dict(zip(le.classes_, le.transform(le.classes_)))
    data[categorical_cols[2]] = le.fit_transform(data[categorical_cols[2]])
    flags_map = dict(zip(le.classes_, le.transform(le.classes_)))
    data[categorical_cols[3]] = le.fit_transform(data[categorical_cols[3]])
    labels_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print(data[categorical_cols].head(20))

    enc = prep.OneHotEncoder(categorical_features=categorical_feature_mask, sparse=False, )
    data_ohe = enc.fit_transform(data)
    ohe_df = pd.DataFrame.from_records(data=data_ohe)
    print(ohe_df.head(20))
    array = ohe_df.values
    X = array[:, 0:41]
    Y = array[:, 41]
#    print(X)
    print(X.shape)
    while choice != '6':
        if choice == '1':
            print("Something is happen1ing here, just kidding...")
        elif choice == '2':
            #feature_selection_menu(X, Y)
            print("Something is happen1ing here, just kidding...")
        elif choice == '3':
            print("Using selected features to run neural network algorithm...")
        elif choice == '4':
            print("Classifying using SVM with selected features...")
        elif choice == '5':
            print("Showing results...")
        elif choice == '6':
            print("Quitting, thanks for using the tool! :)")
            break
        else:
            print("Invalid Option!")

        print("+==============================================================+")
        print("Please select an option:                                       |")
        print("1. Load Data                                                   |")
        print("2. Select Features                                             |")
        print("3. Neural Network                                              |")
        print("4. Support VectPrevious Resultsor Machine                      |")
        print("5. Results                                                     |")
        print("6. Quit                                                        |")
        print("+==============================================================+")
        choice = input("Please make a selection: ")


def feature_selection_menu(data, labels):
    print("|----------------------------------------------|")
    print("| Please select a feature selection method:    |")
    print("|----------------------------------------------|")
    print("|1. K best                                     |")
    print("|2.                                            |")
    print("|3. Previous Menu                              |")
    print("|----------------------------------------------|")
    choice = '3'
    choice = input("Please make a selection: ")
    while(choice != '3'):
        if choice == '1':
            num_features = int(input("Please enter the number of features: "))
            use_fun = input("please enter the function to use: ")
            run_k_best(num_features, use_fun, data, labels)
            break
        elif choice == '3':
            print("Going back to previous menu...")
            break

    print("|----------------------------------------------|")
    print("|1. K best                                     |")
    print("|2.                                            |")
    print("|3. Previous Menu                              |")
    print("|----------------------------------------------|")
    choice = input("Please make a selection: ")


def run_k_best(num_features, use_fun, data, labels):
     features = fs.SelectKBest(getattr(fs, use_fun), num_features).fit_transform(data, labels)
     print(features)
     print(features.shape)



main()