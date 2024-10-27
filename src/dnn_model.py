
from keras.models import Sequential # Venv?  y
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

class DNNModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            Dense(64, input_dim=input_shape, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
