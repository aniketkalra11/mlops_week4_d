import unittest
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

class TestModel(unittest.TestCase):
    restored_model= None
    base_model = None
    top_model_complete = 'iris_randomforest_model.joblib'
    smaple_path = 'samples'
    def setUp(self):
        from joblib import load
        self.restored_model = load(self.top_model_complete)

    def test_model(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score

        # Load the iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Make predictions using the restored model
        predictions = self.restored_model.predict(X)

        # Calculate accuracy
        accuracy = accuracy_score(y, predictions)

        # Assert that the accuracy is above a certain threshold
        self.assertGreater(accuracy, 0.9, "Model accuracy is below the expected threshold.")
        print('IRIS model test passed successfully with accuracy:', accuracy)
    
    def test_dataset(self):
        iris_data = pd.read_csv('generated_iris_data.csv')
        self.assertIsInstance(iris_data, pd.DataFrame, "The dataset should be a pandas DataFrame.")
        self.assertGreater(len(iris_data), 0, "The dataset should not be empty.")
        self.assertIn('sepal_length', iris_data.columns, "The dataset should contain 'sepal_length' column.")
        self.assertIn('sepal_width', iris_data.columns, "The dataset should contain 'sepal_width' column.")
        self.assertIn('petal_length', iris_data.columns, "The dataset should contain 'petal_length' column.")
        self.assertIn('petal_width', iris_data.columns, "The dataset should contain 'petal_width' column.")
        self.assertIn('species', iris_data.columns, "The dataset should contain 'species' column.") 
        print("Dataset test passed successfully.")


    def tearDown(self):
        # Clean up if necessary
        self.restored_model = None
        self.base_model = None
if __name__ == '__main__':
    print('Running tests for the Iris model and Dataset...')
    unittest.main()
    # This will run the tests when the script is executed directly