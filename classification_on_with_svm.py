import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay

##############################################################################################################################################
# 1.Split the data into training and testing (70-30) keeping both sets approximately balanced. Discard entries with missing information (NA).
# 2. Carrega os dados da tabela 'Heart.csv' sem os dados com 0 da coluna "AHD"

data_heart = pd.read_csv('Heart.csv')

# Transforma as palvras "Yes" e "No" em 1 e 0, respectivamente
data = data_heart.dropna()
data['AHD'] = data['AHD'].map({'Yes': 1, 'No': 0})


# Remove as entradas com "NA"
x_na = data.drop(['Unnamed: 0','AHD'], axis=1)
y_ahd = data['AHD']


age_chol = ['Age', 'Chol']
x_age_chol = x_na[age_chol]

# Dados divididos em um conjunto de treinamento e teste (70-30) 
x_train, x_test, y_train, y_test = train_test_split(x_age_chol, y_ahd, test_size=0.3, random_state=42, stratify=y_ahd)

# criação dos subplots
fig, sub = plt.subplots(1, 2, figsize=(10, 6))

# Configurações dos gráficos 
plot_params_train ={
    'x': x_train['Age'],
    'y': x_train['Chol'],
    'hue': y_train,
    'marker': 'o',
    's': 100,
    'alpha': 1,
    'ax': sub[0] 
}
##############################################################################################################################################
#3. Plot the training and testing sets using the chosen variables.
sns.scatterplot(**plot_params_train)
sub[0].set_title('Training')
sub[0].set_xlabel('Age')
sub[0].set_ylabel('Chol')

plot_params_test ={
    'x': x_test['Age'],
    'y': x_test['Chol'],
    'hue': y_test,
    'marker': 'o',
    's': 100,
    'alpha': 1,
    'ax': sub[1] 
}

# Gráfico para treinamento
sns.scatterplot(**plot_params_test)
sub[1].set_title('Testing')
sub[1].set_xlabel('Age')
sub[1].set_ylabel('Chol')

plt.tight_layout()
plt.show()

##############################################################################################################################################
# 4. Use the training set to fit an SVM with linear, polynomial, and radial kernels. Adjust the hyperparameters accordingly.
for kernel in ['linear', 'poly', 'rbf']:
    if kernel == 'linear':
        model = SVC(kernel=kernel, degree=3, C=1)
    elif kernel == 'poly':    
        model = SVC(kernel=kernel, gamma='scale', C=1)
    elif kernel == 'rbf':    
        model = SVC(kernel=kernel, C=1) 
           
    model.fit(x_train, y_train)
    x_train_pred = model.predict(x_train)
    accuracy = accuracy_score(y_train, x_train_pred)
    print(f" Accuracy (Train) - SVM {kernel.capitalize()}: {accuracy*100:.2f}%")

##############################################################################################################################################
# 5. Plot the maximum margin [2] of the training set for each kernel. 
def train_SVM_max(clf, x_model, y_model, title): 
    plot_params ={
    'x': x_model['Age'],
    'y': x_model['Chol'],
    'hue': y_model,
    'marker': 'o',
    's': 100,
    'alpha': 1
}
    plt.figure(figsize=(10, 6))
    sns.scatterplot(**plot_params)
    sub = plt.gca()
    xlim = sub.get_xlim()
    ylim = sub.get_ylim()
 

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    aux = np.vstack([xx.ravel(), yy.ravel()]).T
    aux1 = clf.decision_function(aux).reshape(xx.shape)
    
    #plt.scatter(x_train[y_train == 1][:, 0], x_train[y_train == 1][:, 1], label='Doente', color='r')
    #plt.scatter(x_train[y_train == 0][:, 0], x_train[y_train == 0][:, 1], label='Saudável', color='b')
    sub.contour(xx, yy, aux1, colors='g', levels=[-1, 0, 1], alpha=1, linestyles=['--','-','--'])
    sub.set_title(title)
    sub.set_xlabel('Age')
    sub.set_ylabel('Chol')
    plt.show()
    
  
for kernel in ['linear', 'poly', 'rbf']:
    if kernel == 'linear':
        model = SVC(kernel=kernel, degree=3, C=1)
        title = 'SVM Linear - Maximum Margin'
    elif kernel == 'poly':    
        model = SVC(kernel=kernel, gamma='scale', C=1)
        title = 'SVM Polynomial - Maximum Margin'
    elif kernel == 'rbf':    
        model = SVC(kernel=kernel, C=1)
        title = 'SVM Radial Kernels - Maximum Margin' 
           
    model.fit(x_train, y_train)
    train_SVM_max(model, x_train, y_train, title) 
    
##############################################################################################################################################
# 6. Plot the Region and Boundaries of the training each set for each kernel
 # SVM regularization parameter
# Padronizar os dados
standard_scaler = StandardScaler()
x_train_standard = standard_scaler.fit_transform(x_train)

models = (
    svm.SVC(kernel="linear", C=1),
    svm.LinearSVC(C=1, max_iter=10000),
    svm.SVC(kernel="rbf", gamma=0.7, C=1),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=1),
)

models = (clf.fit(x_train_standard, y_train) for clf in models)

# title for the plots
titles = (
    "SVC with linear kernel (Train)",
    "LinearSVC (linear kernel (Train))",
    "SVC with RBF kernel (Train)",
    "SVC with polynomial (degree 3) kernel (Train)",
)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2, figsize=(10, 6))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

x0, x1 = x_train_standard[:, 0], x_train_standard[:, 1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        x_train_standard,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax
    )
    ax.scatter(x0, x1, c=y_train, cmap=plt.cm.coolwarm, s=100, edgecolors="k")
    ax.set_xlabel('Age')
    ax.set_ylabel('Chol')
    ax.set_title(title)
    
plt.tight_layout()
plt.show()

##############################################################################################################################################
#7. Estimate the heart disease (AHD) on the test set and calculate the accuracy (percentage of success) for each kernel.
x_test_standard = standard_scaler.transform(x_test)

for kernel in ['linear', 'poly', 'rbf']:
    if kernel == 'linear':
        model = SVC(kernel=kernel, degree=3, C=1)
    elif kernel == 'poly':    
        model = SVC(kernel=kernel, gamma='scale', C=1)
    elif kernel == 'rbf':    
        model = SVC(kernel=kernel, C=1) 
     
    model.fit(x_test_standard, y_test)       
    x_test_pred = model.predict(x_test_standard)
    accuracy = accuracy_score(y_test, x_test_pred)
    print(f" Accuracy (Test) - SVM {kernel.capitalize()}: {accuracy*100:.2f}%")
    
##############################################################################################################################################
# 8. Present the plots of the regions and decision boundaries of the test set for each kernel.

models = (
    svm.SVC(kernel="linear", C=1),
    svm.LinearSVC(C=1, max_iter=10000),
    svm.SVC(kernel="rbf", gamma=0.7, C=1),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=1),
)

models = (clf.fit(x_test_standard, y_test) for clf in models)

# title for the plots
titles = (
    "SVC with linear kernel (Test)",
    "LinearSVC (linear kernel (Test))",
    "SVC with RBF kernel (Test)",
    "SVC with polynomial (degree 3) kernel (Test)",
)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2, figsize=(10, 6))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

xt0, xt1 = x_test_standard[:, 0], x_test_standard[:, 1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        x_test_standard,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax
    )
    ax.scatter(xt0, xt1, c=y_test, cmap=plt.cm.coolwarm, s=100, edgecolors="k")
    ax.set_xlabel('Age')
    ax.set_ylabel('Chol')
    ax.set_title(title)
    
plt.tight_layout()
plt.show()

