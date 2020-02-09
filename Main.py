from Unigrams_module import *
from baseline_extract import *
from simple_features_module import *
from Cross_validation_module import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.svm import SVC



df = pd.read_csv('training.csv', names = ['line', 'character', 'gender'])
corpus = []
for index, row in df.iterrows():
    corpus.append(row.line)
gender = df.gender.values.tolist()

type = input("Please enter classification type ('gender'/'character'): ")

if type not in ['gender', 'character']:
    raise ValueError('Invalid input please select a valid classification type')

characters = []
characters_set = list(set(df.character))
for index, char in enumerate(characters_set):
    characters.append(characters_set[index][0] + characters_set[index][1:].lower())
simple_features = feature_extract.simple_features(df.line, characters)

###@@@@@@@@@@@@@@@@@@@@@
#Gender Frames
if type == 'gender':
    male_frame, male_model = ent_perp_frame.train(corpus, label = gender, target = 'male', type = type)
    female_frame, female_model = ent_perp_frame.train(corpus, label = gender, target = 'female', type = type)
    gender_final_frame = pd.concat([male_frame, female_frame, simple_features], axis =1)
    #print(gender_final_frame)


###@@@@@@@@@@@@@@@@@@@@@@

#character Frames

character = df.character.values.tolist()
if type == 'character':
    characters_frame = pd.DataFrame(simple_features)
    characters_models = Counter()
    #experiment = df.character
    for c in characters:
        c = c.upper()
        c_frame, c_model = ent_perp_frame.train(corpus, label=character, target=c, type=type)
        characters_models[c] = c_model
        characters_frame = pd.concat([characters_frame, c_frame], axis =1)
        #experiment = pd.concat([experiment, c_frame], axis =1)
    #experiment = experiment.drop(columns = ['character'])




y_gender_train = df.gender
y_char_train = df.character


if type == 'character':
    X_train, y_train = characters_frame, y_char_train
elif type == 'gender':
    X_train, y_train = gender_final_frame, y_gender_train

'Baselines'
if type == 'gender':
    tr_base_report = baseline.gender(y_train)
    ###ts_base_report = baseline.gender(y_test)
elif type == 'character':
    tr_base_report = baseline.characters(y_train, characters_set)
    ###ts_base_report = baseline.characters(y_test, characters_set)

print('Train Baseline:')
print(tr_base_report)
###print('----------------------------------------------------------------')
###print('Test Baseline:')
###print(ts_base_report)
print('----------------------------------------------------------------')
print('*****************************************************************')
print('----------------------------------------------------------------')

'Logistic Model'
if type == 'gender':
    logistic = LogisticRegression().fit(X_train, y_train)
elif type == 'character':
    logistic = LogisticRegression(solver  = 'lbfgs', multi_class = 'multinomial').fit(X_train, y_train)
tr_l_report = classification_report(y_train, logistic.predict(X_train))
###ts_l_report = classification_report(y_test, logistic.predict(X_test))
print('Logistic:')
print('Train:')
print(tr_l_report)
###print('----------------------------------------------------------------')
###print('Test')
###print(ts_l_report)
print('----------------------------------------------------------------')
print('*****************************************************************')
print('----------------------------------------------------------------')

'SVM Model'

svm_svc = SVC().fit(X_train, y_train)
tr_svc_report = classification_report(y_train, svm_svc.predict(X_train))
###ts_svc_report = classification_report(y_test, svm_svc.predict(X_test))
print('Support Vector Machine:')
print('Train:')
print(tr_svc_report)
###print('----------------------------------------------------------------')
###print('Test:')
###print(ts_svc_report)
print('----------------------------------------------------------------')
#<><><>><><><><>><><><><><><><><>><><><><<><><><><<><><><><><><><<><><><>><><><><><><><><><<><>><<><><><><><>#
'''CrossValidation'''
print('Cross Validation...')

lam = np.arange(0.1, 1.1, 0.1)
if type == 'character':
    cv_results, optimal_lambda = crossValidate(characters_frame, y_char_train, lam, folds = 5, type = type)
elif type == 'gender':
    cv_results, optimal_lambda = crossValidate(gender_final_frame, y_gender_train, lam, folds = 5, type =type)

#print(optimal_lambda)
print('The Cross Validation Results are: Precision: {}, Recall: {}, Fscore: {}'.format(cv_results[0], cv_results[1], cv_results[2]))

#<><><><><><><><>><<<><><><><<>><<>><<><<><><<><<><><><<><<><><<<><><><<>><><><><>><><><><><><><><><#
#Integration of test_data
df_test = pd.read_csv('test.csv', names = ['line', 'character', 'gender'])
corpus_test = []
for index, row in df_test.iterrows():
    corpus_test.append(row.line)
gender_test = df_test.gender.values.tolist()

simple_features_test = feature_extract.simple_features(df_test.line, characters)

if type == 'gender':
    male_test_frame = ent_perp_frame.test(corpus_test, gender_test,  male_model, target = 'male', type = type)
    female_test_frame = ent_perp_frame.test(corpus_test, gender_test,  female_model, target = 'female', type = type)
    gender_final_frame_test = pd.concat([male_test_frame, female_test_frame, simple_features_test], axis=1) #simple_features_test
    final_X = gender_final_frame_test
    final_y = df_test.gender
    #test_baseline_report = baseline.gender(final_y)
elif type == 'character':
    characters_frame_test = pd.DataFrame(simple_features_test)
    #experiment = df_test.character
    for c in characters:
        c = c.upper()
        c_frame = ent_perp_frame.test(corpus_test, df_test.character, characters_models[c], target=c, type=type)
        characters_frame_test = pd.concat([characters_frame_test, c_frame], axis=1)
        #experiment = pd.concat([experiment, c_frame], axis=1)
    #experiment = experiment.drop(columns = ['character'])

    final_X = characters_frame_test
    final_y = df_test.character
    #test_baseline_report = baseline.characters(final_y, characters_set)
#print('Test baseline:')
#print(test_baseline_report)


#testing model performance on test.csv data
print('Test.csv data model performance..')
test_prediction = logistic.predict(final_X)

logistic_report = classification_report(final_y, test_prediction)
#error_obs = pd.concat([df_test.line, final_y, pd.Series(test_prediction)], names = ['input', 'label', 'prediction'], axis =1)
#error_obs.to_csv('Observed_predictions_gender.csv')
#print(error_obs)
print('logistic model:')
print(logistic_report)
print('---------------------------------------------')
print('Support vector machine:')
svm_report = classification_report(final_y, svm_svc.predict(final_X))
print(svm_report)


