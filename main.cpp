#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

class NeuralNetwork {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    vector<vector<double>> weights1;
    vector<double> biases1;
    vector<vector<double>> weights2;
    vector<double> biases2;
public:
    NeuralNetwork(int input, int hidden, int output) {
        inputSize = input;
        hiddenSize = hidden;
        outputSize = output;
        weights1 = vector<vector<double>>(hiddenSize, vector<double>(inputSize));
        biases1 = vector<double>(hiddenSize);
        weights2 = vector<vector<double>>(outputSize, vector<double>(hiddenSize));
        biases2 = vector<double>(outputSize);
        // Initialisation aléatoire des poids et biais
        srand(time(NULL));
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights1[i][j] = ((double)rand() / RAND_MAX) - 0.5;
            }
            biases1[i] = ((double)rand() / RAND_MAX) - 0.5;
        }
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights2[i][j] = ((double)rand() / RAND_MAX) - 0.5;
            }
            biases2[i] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    vector<double> forward(vector<double> inputs) {
        vector<double> hidden(hiddenSize);
        vector<double> outputs(outputSize);
        // Calcul de la couche cachée
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0;
            for (int j = 0; j < inputSize; j++) {
                sum += inputs[j] * weights1[i][j];
            }
            hidden[i] = max(0.0, sum + biases1[i]); // Fonction d'activation ReLU
        }
        // Calcul de la sortie
        for (int i = 0; i < outputSize; i++) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * weights2[i][j];
            }
            outputs[i] = sum + biases2[i];
        }
        return outputs;
    }
    void backward(vector<double> inputs, vector<double> targets, double learningRate) {
        // Forward pass pour calculer les sorties et les activations de la couche cachée
        vector<double> hidden(hiddenSize);
        vector<double> outputs(outputSize);
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0;
            for (int j = 0; j < inputSize; j++) {
                sum += inputs[j] * weights1[i][j];
            }
            hidden[i] = max(0.0, sum + biases1[i]); // Fonction d'activation ReLU
        }
        for (int i = 0; i < outputSize; i++) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * weights2[i][j];
            }
            outputs[i] = sum + biases2[i];
        }

        // Calcul des erreurs de la sortie
        vector<double> outputErrors(outputSize);
        for (int i = 0; i < outputSize; i++) {
            outputErrors[i] = outputs[i] - targets[i];
        }

        // Calcul des erreurs de la couche cachée
        vector<double> hiddenErrors(hiddenSize);
        for (int i = 0; i < hiddenSize; i++) {
            double error = 0;
            for (int j = 0; j < outputSize; j++) {
                error += outputErrors[j] * weights2[j][i];
            }
            hiddenErrors[i] = error * (hidden[i] > 0 ? 1 : 0); // Dérivée de la fonction d'activation ReLU
        }

        // Mise à jour des poids et biais de la deuxième couche
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights2[i][j] -= learningRate * outputErrors[i] * hidden[j];
            }
            biases2[i] -= learningRate * outputErrors[i];
        }

        // Mise à jour des poids et biais de la première couche
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights1[i][j] -= learningRate * hiddenErrors[i] * inputs[j];
            }
            biases1[i] -= learningRate * hiddenErrors[i];
        }
    }
};


int main() {
    NeuralNetwork nn(2, 3, 1); // Réseau de neurones avec 2 entrées, une couche cachée de 3 neurones et 1 sortie
    vector<double> inputs = {1, 0}; // Entrées
    vector<double> outputs = nn.forward(inputs); // Calcul de la sortie
    cout << "Output: " << outputs[0] << endl;
    return 0;
}
