import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import requests
from io import BytesIO
import zipfile

class BankMarketingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bank Marketing Analysis")

        # Download and extract the Bank Marketing dataset
        self.download_data()

        # Preprocess the data
        self.preprocess_data()

        # Train a Random Forest classifier
        self.train_classifier()

        # GUI components
        self.label = ttk.Label(root, text="Bank Marketing Analysis", font=("Helvetica", 16))
        self.label.pack(pady=10)

        # Buttons
        self.btn_distribution = ttk.Button(root, text="Show Distribution Plot", command=self.show_distribution)
        self.btn_distribution.pack(pady=5)

        self.btn_feature_importance = ttk.Button(root, text="Show Feature Importance Plot", command=self.show_feature_importance)
        self.btn_feature_importance.pack(pady=5)

        self.btn_confusion_matrix = ttk.Button(root, text="Show Confusion Matrix Heatmap", command=self.show_confusion_matrix)
        self.btn_confusion_matrix.pack(pady=5)

        # Exit button
        self.btn_exit = ttk.Button(root, text="Exit", command=root.destroy)
        self.btn_exit.pack(pady=10)

        # Canvas for displaying plots
        self.canvas = None

    def download_data(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
        response = requests.get(url)
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            with z.open('bank-additional/bank-additional-full.csv') as f:
                self.bank_data = pd.read_csv(f, sep=';')

    def preprocess_data(self):
        label_encoder = LabelEncoder()
        self.bank_data['y'] = label_encoder.fit_transform(self.bank_data['y'])
        self.bank_data = pd.get_dummies(self.bank_data)

        # Separate features and target variable
        self.X = self.bank_data.drop('y', axis=1)
        self.y = self.bank_data['y']

    def train_classifier(self):
        # Train a Random Forest classifier
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_classifier.fit(X_train, y_train)

    def clear_canvas(self):
        # Clear the canvas and redraw if it exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def show_distribution(self):
        self.clear_canvas()  # Clear previous plot
        plt.figure(figsize=(8, 5))
        sns.countplot(x='y', data=self.bank_data)
        plt.title('Distribution of the Target Variable (y)')
        self.show_plot()

    def show_feature_importance(self):
        self.clear_canvas()  # Clear previous plot
        plt.figure(figsize=(10, 6))
        feature_importance = pd.Series(self.rf_classifier.feature_importances_, index=self.X.columns)
        feature_importance.nlargest(10).plot(kind='barh')
        plt.title('Top 10 Most Important Features')
        self.show_plot()

    def show_confusion_matrix(self):
        self.clear_canvas()  # Clear previous plot
        # Make predictions on the test set
        y_pred = self.rf_classifier.predict(self.X)

        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.crosstab(self.y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True),
                    annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        self.show_plot()

    def show_plot(self):
        # Show the plot on the canvas
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = BankMarketingGUI(root)
    root.mainloop()
