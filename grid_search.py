import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.init as nn_init
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from preprocessing import MSC, MeanCenter, Autoscale, trans2absr
import math
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchviz import make_dot
from SPanalysis import SpectralAnalysis
# Image display
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV
device = torch.device('cpu')
run_on_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('yes')
    run_on_gpu = True

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


dataset= 'dataset/Combind_H_canopy.xlsx' 
analysis = SpectralAnalysis(dataset,'PLSR')
analysis.preprocess_data()
def rwctocat(clss):
    if clss== 'Well watered':
        return 0
    elif clss== 'Mild stress':
        return 1
    else:
        return 2

X_train, y_train_reg, y_train_class, X_test, y_test_reg, y_test_class = analysis.X_train, analysis.y_train_reg.iloc[:,0],analysis.y_train_class,analysis.X_test,analysis.y_test_reg.iloc[:,0],analysis.y_test_class

y_train_class = y_train_class.astype('category')
y_test_class =  y_test_class.astype('category')
print(f'Training sample {X_train.shape[0]}')
print(f'Total spectra {X_train.shape[1]}')
print(f'Val sample {X_test.shape[0]}')
y_test_reg.shape


# grid searching

# Define the MLP model
class SpectralMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_prob, device):
        super(SpectralMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.bn1 = nn.BatchNorm1d(hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.bn2 = nn.BatchNorm1d(hidden_dim).to(device)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2).to(device)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2).to(device)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 2).to(device)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2).to(device)
        self.fc5 = nn.Linear(hidden_dim // 2, hidden_dim // 4).to(device)
        self.bn5 = nn.BatchNorm1d(hidden_dim // 4).to(device)
        self.fc6 = nn.Linear(hidden_dim // 4, hidden_dim // 4).to(device)
        self.bn6 = nn.BatchNorm1d(hidden_dim // 4).to(device)
        self.output = nn.Linear(hidden_dim // 4, output_dim).to(device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        # x = self.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x

# Define the custom PyTorch model wrapper
class SpectralMLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim = 130, hidden_dim=32, dropout_prob=0.2, learning_rate=0.001, weight_decay =0.0001, num_epochs =1000 ,batch_size=64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpectralMLP(input_dim, 1, hidden_dim, dropout_prob, device=self.device).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=4, verbose=False)
        self.train_loss = []
        self.val_loss = []

    def fit(self, X, y, X_val=None, y_val=None):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y if isinstance(y, np.ndarray) else y.values, dtype=torch.float32).view(-1, 1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val if isinstance(y_val, np.ndarray) else y_val.values, dtype=torch.float32).view(-1, 1).to(self.device)
            val_dataset = TensorDataset(X_val_tensor,y_val_tensor)
            valLoader = DataLoader(val_dataset,batch_size=self.batch_size,shuffle=True) 
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for x_batch, y_batch in loader:
                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = torch.sqrt(self.criterion(logits, y_batch))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(loader)
            self.train_loss.append(avg_loss)

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    running_val_loss = 0.0
                    for val_x_batch, val_y_batch in valLoader: 
                        val_logits = self.model(val_x_batch)
                        val_loss = torch.sqrt(self.criterion(val_logits, val_y_batch)).item()
                        running_val_loss += val_loss
                    avg_val_loss=running_val_loss/len(valLoader)
                    self.val_loss.append(avg_val_loss)
                self.model.train()

            self.scheduler.step(avg_loss)
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}' if X_val is not None and y_val is not None else f'Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}')

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        return predictions

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_dim': [32,64,128],
    'dropout_prob': [0.2,0.3,0.5,0.6],
    'learning_rate': [0.0001,0.001,0.0025,0.005,0.0075,0.01],
    'weight_decay': [0.0001,0.0005],
    'num_epochs': [1000,3000],
    'batch_size':[64,128,256]
}

# Create the GridSearchCV object
scorer = make_scorer(r2_score)
grid_search = GridSearchCV(estimator=SpectralMLPWrapper(input_dim=X_train.shape[1]),
                           param_grid=param_grid,
                           scoring=scorer,
                           cv=10,
                           verbose=2,
                           n_jobs=-1)



# Fit the model
grid_search.fit(X_train, y_train_reg, X_val=X_test, y_val=y_test_reg)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Best estimator
best_mlp = grid_search.best_estimator_

# Predict on test data
y_pred = best_mlp.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test_reg, y_pred)
print(f'R2 Score on test data: {r2}')

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(best_mlp.train_loss, label='Training Loss')
plt.plot(best_mlp.val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


from sklearn.metrics import r2_score

# Switch model to evaluation mode
best_mlp.model.eval()

# Generate some test data
# test_features = torch.randn(20, 177).to(device)
# test_labels = torch.randint(0, 3, (20,)).to(device)
# test_regression_targets = torch.randn(20, 1).to(device)

# Make predictions
with torch.no_grad():

    reg_outputs = best_mlp.model(torch.Tensor(X_test).to(device=device))
    
# Convert tensors to numpy arrays for plotting



y_actual = y_test_reg
y_pred = reg_outputs.detach().cpu().numpy()

y_pred = analysis.scalers["RWC"].inverse_transform(y_pred)
y_true_test = analysis.scalers["RWC"].inverse_transform(y_actual.values.reshape(-1, 1))

# Calculate R-squared value
r2 = r2_score(y_true_test, y_pred)

# Plot the results
plt.figure(figsize=(10, 5))
plt.scatter(y_true_test, y_pred, label='Predicted vs Actual')
plt.plot([y_true_test.min(), y_true_test.max()], [y_true_test.min(), y_true_test.max()], 'r--', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted Values (R2 Score: {r2:.2f}) \n dropout_prob: 0.4, hidden_dim: 128, learning_rate: 0.001, num_epochs: 100, weight_decay: 0.0001')
plt.legend()
plt.show()

print(f"R2 Score: {r2:.2f}")